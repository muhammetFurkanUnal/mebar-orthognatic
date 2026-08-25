"""
Run the full prompting experiment: iterate over the manifest CSV,
calling the LLM for each pending entry, saving responses, and
updating the manifest checkpoint.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from config import config
from prompting.exception import (
    LLMAuthenticationError,
    LLMRequestPreparationError,
    LLMTransportError,
    LLMProviderError,
    LLMRateLimitError,
    LLMProviderServerError,
    LLMResponseError,
    LLMInvalidJsonError,
)
from prompting.llm_handler import LLMCallClient
from prompting.manifest_builder import ManifestBuilder, ManifestEntry


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class LogRow:
    call_id: str
    model: str
    patient_name: str
    view: str
    educated: bool
    image_count: int
    pdf_count: int
    call_duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    success: bool
    cost_usd: float | None
    attempt_number: int
    is_retry: bool
    err_message: str
    err_type: str
    timestamp_utc: str = field(default_factory=utc_timestamp, init=False)


def generate_id() -> str:
    return f"call_{uuid4().hex}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "prompting" / "manifest.csv"
LOGS_CSV_PATH = PROJECT_ROOT / "prompting" / "outputs" / "logs.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_log_row(log_row: LogRow) -> None:
    import csv

    fieldnames = [
        "call_id", "model", "patient_name", "view", "educated",
        "image_count", "pdf_count", "call_duration_ms",
        "input_tokens", "output_tokens", "success", "cost_usd",
        "attempt_number", "is_retry", "err_message", "err_type",
        "timestamp_utc",
    ]
    file_exists = LOGS_CSV_PATH.exists()
    with open(LOGS_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: str(getattr(log_row, k)) for k in fieldnames})


def _write_result_disk(entry: ManifestEntry, payload: dict) -> None:
    Path(entry.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(entry.output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Process one manifest entry
# ---------------------------------------------------------------------------


def process_entry(
    entry: ManifestEntry,
    client: LLMCallClient,
    *,
    max_attempts: int = 5,
    rate_limit_max_sleep: int = 20,
) -> None:
    """
    Run the LLM call for one manifest entry with retries.

    If ``entry.done`` is already ``True`` this function returns immediately
    without making any API call.

    On success:
      - Write the LLM response JSON to entry.output_path.
      - Set entry.done = True.

    On terminal failure (auth, prep, provider error or limit exceeded):
      - Log the error but do *not* mark the entry as done.
      - Write the error info to entry.output_path.
    """

# -- Skip if already done -------------------------------------------------
    if entry.done:
        return
    # -- Read system prompt ------------------------------------------------
    try:
        system_prompt = Path(entry.system_prompt_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        err = f"System prompt file not found: {entry.system_prompt_path}"
        payload = {
            "error": True, "error_type": "SystemPromptNotFound", "error_message": err,
            "configuration_id": entry.configuration_id, "model": entry.model,
            "view": entry.view, "educated": entry.educated, "patient_name": entry.patient_name,
        }
        _write_result_disk(entry, payload)
        _append_log_row(LogRow(
            call_id=generate_id(), model=entry.model, patient_name=entry.patient_name,
            view=entry.view, educated=entry.educated, image_count=0, pdf_count=0,
            call_duration_ms=0, input_tokens=None, output_tokens=None, success=False,
            cost_usd=None, attempt_number=1, is_retry=False, err_message=err,
            err_type="SystemPromptNotFound",
        ))
        return

    # -- Patient image -----------------------------------------------------
    image_paths = []
    if entry.patient_image_path and Path(entry.patient_image_path).is_file():
        image_paths = [entry.patient_image_path]

    if not image_paths:
        err = f"Patient image not found: {entry.patient_image_path}"
        payload = {
            "error": True, "error_type": "PatientImageNotFound", "error_message": err,
            "configuration_id": entry.configuration_id, "model": entry.model,
            "view": entry.view, "educated": entry.educated, "patient_name": entry.patient_name,
        }
        _write_result_disk(entry, payload)
        _append_log_row(LogRow(
            call_id=generate_id(), model=entry.model, patient_name=entry.patient_name,
            view=entry.view, educated=entry.educated, image_count=0, pdf_count=0,
            call_duration_ms=0, input_tokens=None, output_tokens=None, success=False,
            cost_usd=None, attempt_number=1, is_retry=False, err_message=err,
            err_type="PatientImageNotFound",
        ))
        return

    # -- Append literature summary to system prompt if educated ------------
    ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
    LITERATURE_FILE = "arnett_summary.xml"

    file_paths = None
    pdf_count = 0
    if entry.educated:
        lit_path = ASSETS_DIR / LITERATURE_FILE
        if not lit_path.is_file():
            err = f"Educated prompt requires literature summary but file not found: {lit_path}"
            print(f"  [ERR] {err}", file=sys.stderr)
            payload = {
                "error": True, "error_type": "LiteratureSummaryNotFound", "error_message": err,
                "configuration_id": entry.configuration_id, "model": entry.model,
                "view": entry.view, "educated": entry.educated, "patient_name": entry.patient_name,
            }
            _write_result_disk(entry, payload)
            _append_log_row(LogRow(
                call_id=generate_id(), model=entry.model, patient_name=entry.patient_name,
                view=entry.view, educated=entry.educated, image_count=len(image_paths),
                pdf_count=0, call_duration_ms=0, input_tokens=None, output_tokens=None,
                success=False, cost_usd=None, attempt_number=1, is_retry=False,
                err_message=err, err_type="LiteratureSummaryNotFound",
            ))
            return
        lit_content = lit_path.read_text(encoding="utf-8")
        system_prompt += "\n\n<!-- LITERATURE SUMMARY -->\n" + lit_content
        pdf_count = 1

    # -- Retry loop ---------------------------------------------------------
    success = False
    error_message = ""
    error_type = ""
    current_attempt = 0
    rate_limit_exc_counter = 0
    inv_json_exc_counter = 0
    response_error_exc_counter = 0
    response_error_exc_limit = 1
    inv_json_exc_limit = 1
    call_id = ""

    for attempt in range(max_attempts):
        current_attempt = attempt + 1
        call_id = generate_id()

        try:
            client.llm_call(
                base_url=config.base_url,
                model_name=entry.model,
                api_key=config.API_KEY_OPENROUTER,
                system_prompt=system_prompt,
                image_paths=image_paths,
                file_paths=file_paths,
            )
        except LLMRequestPreparationError as exc:
            print(f"  [ERR] Request preparation: {exc}", file=sys.stderr)
            error_message = str(exc); error_type = "LLMRequestPreparationError"
            break
        except LLMAuthenticationError as exc:
            print(f"  [ERR] Auth: {exc}", file=sys.stderr)
            error_message = str(exc); error_type = "LLMAuthenticationError"
            break
        except LLMTransportError as exc:
            print(f"  [WARN] Transport: {exc}", file=sys.stderr)
            time.sleep(1.5)
            continue
        except LLMRateLimitError as exc:
            print(f"  [WARN] Rate limit: {exc}", file=sys.stderr)
            rate_limit_exc_counter += 1
            sleep_sec = min(2 ** (rate_limit_exc_counter + 1), rate_limit_max_sleep)
            print(f"  Sleeping {sleep_sec}s ...")
            time.sleep(sleep_sec)
            continue
        except LLMProviderServerError as exc:
            print(f"  [WARN] Server error: {exc}", file=sys.stderr)
            time.sleep(1.5)
            continue
        except LLMProviderError as exc:
            print(f"  [ERR] Provider: {exc}", file=sys.stderr)
            error_message = str(exc); error_type = "LLMProviderError"
            break
        except LLMResponseError as exc:
            print(f"  [WARN] Response error: {exc}", file=sys.stderr)
            response_error_exc_counter += 1
            if response_error_exc_counter > response_error_exc_limit:
                error_message = str(exc); error_type = "LLMResponseError"
                break
            continue
        except LLMInvalidJsonError as exc:
            print(f"  [WARN] Invalid JSON: {exc}", file=sys.stderr)
            inv_json_exc_counter += 1
            if inv_json_exc_counter > inv_json_exc_limit:
                error_message = str(exc); error_type = "LLMInvalidJsonError"
                break
            continue

        success = True
        break

    if not success and not error_type:
        error_type = "AttemptLimitReached"
        error_message = "All retry attempts failed due to non-fatal errors."

    api_resp = client.api_response

    # -- Build output payload ----------------------------------------------
    if success and api_resp and api_resp.content:
        # Try to parse LLM response; if it contains an "err" key then
        # the model self-reported a condition it could not handle
        # (e.g. ruler not legible, literature missing).  Treat as failure.
        model_err_type = None
        model_err_msg = None
        parsed = None
        try:
            parsed = json.loads(api_resp.content)
            if isinstance(parsed, dict) and "err" in parsed:
                model_err_type = "ModelReportedError"
                model_err_msg = parsed["err"]
        except (json.JSONDecodeError, TypeError):
            pass

        if model_err_type:
            success = False
            error_type = model_err_type
            error_message = model_err_msg or "Model reported an error in its response"

        payload = {
            "configuration_id": entry.configuration_id, "model": entry.model,
            "patient_name": entry.patient_name, "view": entry.view, "educated": entry.educated,
            "pdf_paths": file_paths,
            "response_content": api_resp.content,
            "input_tokens": api_resp.input_tokens, "output_tokens": api_resp.output_tokens,
            "total_tokens": api_resp.total_tokens, "cost_usd": api_resp.cost_usd,
            "finish_reason": api_resp.finish_reason, "api_duration_ms": api_resp.api_duration_ms,
            "attempt_number": current_attempt,
            "error": (not success),
            "error_type": error_type if not success else None,
            "error_message": error_message if not success else None,
        }
        if success:
            payload["response_parsed"] = parsed if model_err_type is None else None
        else:
            payload["response_parsed"] = None
    else:
        payload = {
            "configuration_id": entry.configuration_id, "model": entry.model,
            "patient_name": entry.patient_name, "view": entry.view, "educated": entry.educated,
            "pdf_paths": file_paths,
            "response_content": api_resp.content if api_resp else None,
            "input_tokens": api_resp.input_tokens if api_resp else None,
            "output_tokens": api_resp.output_tokens if api_resp else None,
            "total_tokens": api_resp.total_tokens if api_resp else None,
            "cost_usd": api_resp.cost_usd if api_resp else None,
            "finish_reason": api_resp.finish_reason if api_resp else None,
            "api_duration_ms": api_resp.api_duration_ms if api_resp else 0,
            "attempt_number": current_attempt, "error": True,
            "error_type": error_type, "error_message": error_message,
        }

    _write_result_disk(entry, payload)

    log_row = LogRow(
        call_id=call_id, model=entry.model, patient_name=entry.patient_name,
        view=entry.view, educated=entry.educated,
        image_count=len(image_paths), pdf_count=pdf_count,
        call_duration_ms=api_resp.api_duration_ms if api_resp else 0,
        input_tokens=api_resp.input_tokens if api_resp else None,
        output_tokens=api_resp.output_tokens if api_resp else None,
        success=success, cost_usd=api_resp.cost_usd if api_resp else None,
        attempt_number=current_attempt, is_retry=(current_attempt > 1),
        err_message=error_message, err_type=error_type,
    )
    _append_log_row(log_row)

    if success:
        entry.done = True

    print(f"  {'OK' if success else 'FAIL'} @ attempt {current_attempt} | "
          f"tokens in/out: {log_row.input_tokens}/{log_row.output_tokens} | "
          f"cost: {log_row.cost_usd}")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------


def run_test_mode(entries: list[ManifestEntry], manifest_path: Path) -> None:
    """
    Test mode: runs a small hand-picked subset of configurations.

    Picks:
      - 2 random patients with profile + uneducated (across different models)
      - 1 random patient with profile + educated
    All three use the FIRST model (gemini) for simplicity.

    Results are saved to disk exactly like full mode, and the manifest
    is checkpointed so test runs are tracked.  Entries are NOT marked
    done so they can be re-tested later if needed.
    """
    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)

    # Filter to profile-only, pending entries
    profile_entries = [e for e in entries if e.view == "profile" and not e.done]
    if not profile_entries:
        print("No pending profile entries available for test.", file=sys.stderr)
        return

    # Use only the first model for test (gemini)
    test_model = ManifestBuilder.MODELS[0]

    # Filter by model and education
    uneducated_pool = [
        e for e in profile_entries
        if not e.educated and e.model == test_model
    ]
    educated_pool = [
        e for e in profile_entries
        if e.educated and e.model == test_model
    ]

    if len(uneducated_pool) < 2:
        print(f"Not enough uneducated profile entries (need 2, found {len(uneducated_pool)}).", file=sys.stderr)
        return
    if not educated_pool:
        print(f"No educated profile entries available.", file=sys.stderr)
        return

    # Random selections
    selected_uneducated = random.sample(uneducated_pool, 2)
    selected_educated = random.sample(educated_pool, 1)

    selected = selected_uneducated + selected_educated

    print(f"Test model: {test_model}")
    print(f"Selected {len(selected)} test configurations:")
    for s in selected:
        print(f"  - {s.view} | educated={s.educated} | patient={s.patient_name} | "
              f"image_exists={Path(s.patient_image_path).is_file()}")

    # Build a test manifest path
    test_manifest = manifest_path.parent / "manifest_test.csv"
    print(f"\nTest manifest will be checkpointed to: {test_manifest}")

    # Mark test entries so they don't accidentally get picked up by full mode.
    # We clone them with new configuration_ids for isolation.
    test_entries: list[ManifestEntry] = []
    for original in selected:
        test_entry = ManifestEntry(
            educated=original.educated,
            view=original.view,
            system_prompt_path=original.system_prompt_path,
            model=original.model,
            patient_name=original.patient_name,
            patient_image_path=original.patient_image_path,
        )
        test_entry.output_path = ManifestBuilder._output_path(
            test_entry.model, test_entry.view, test_entry.educated, test_entry.configuration_id
        )
        test_entries.append(test_entry)

    # Save test manifest
    ManifestBuilder.save_csv(test_entries, test_manifest)

    print("\nRunning test configurations...")
    print("─" * 60)

    client = LLMCallClient()
    try:
        for idx, entry in enumerate(test_entries, start=1):
            model_short = entry.model_short
            print(f"\n[{idx}/{len(test_entries)}] "
                  f"model={model_short}, view={entry.view}, "
                  f"educated={entry.educated}, patient={entry.patient_name}")
            process_entry(entry, client)
            # Save checkpoint — note: entry.done will be True if successful
            ManifestBuilder.save_csv(test_entries, test_manifest)
            print(f"  Output: {entry.output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted during test. Checkpointing...", file=sys.stderr)
        ManifestBuilder.save_csv(test_entries, test_manifest)
        sys.exit(130)

    print("─" * 60)
    successes = sum(1 for e in test_entries if e.done)
    failures = len(test_entries) - successes
    print(f"Test complete: {successes} succeeded, {failures} failed")
    print(f"Test manifest saved to: {test_manifest}")
    print("NOTE: Test entries are NOT marked in the main manifest, so they can be re-tested.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Check for --test flag
    is_test = "--test" in sys.argv

    manifest_path = Path(MANIFEST_PATH)

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Building it now...", file=sys.stderr)
        builder = ManifestBuilder()
        entries = builder.build_all_entries()
        ManifestBuilder.save_csv(entries, manifest_path)
    else:
        entries = ManifestBuilder.load_csv(manifest_path)

    if is_test:
        run_test_mode(entries, manifest_path)
        sys.exit(0)

    pending = [e for e in entries if not e.done]
    if not pending:
        print("All configurations are already done. Nothing to run.")
        sys.exit(0)

    print(f"Total entries: {len(entries)}")
    print(f"Pending entries: {len(pending)}")
    print("Starting processing (Ctrl+C to checkpoint and exit)...")
    print("─" * 60)

    client = LLMCallClient()

    try:
        for idx, entry in enumerate(pending, start=1):
            model_short = entry.model_short
            print(f"[{idx}/{len(pending)}] model={model_short}, "
                  f"view={entry.view}, educated={entry.educated}, "
                  f"patient={entry.patient_name}")
            process_entry(entry, client)
            # Checkpoint after each entry
            ManifestBuilder.save_csv(entries, manifest_path)
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpointing manifest...", file=sys.stderr)
        ManifestBuilder.save_csv(entries, manifest_path)
        print("Manifest saved. You can resume later.", file=sys.stderr)
        sys.exit(130)

    print("─" * 60)
    remaining = sum(1 for e in entries if not e.done)
    print(f"Finished. Remaining pending: {remaining} / {len(entries)}")
    sys.exit(0)
