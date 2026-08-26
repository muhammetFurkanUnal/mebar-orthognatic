"""
Summarize experiment results stored under prompting/outputs/success/<model>/.

Reads every .json result file, groups them by prompt type
(view x education level) and writes one summary CSV per group:

  educated_front_summary.csv
  uneducated_front_summary.csv
  educated_profile_summary.csv
  uneducated_profile_summary.csv

Each row of a summary CSV corresponds to one result file (one LLM call).
Measurement fields are flattened from ``response_parsed``; failed calls
are included with empty measurement columns plus their error metadata.

Usage:
    python3 -m prompting.summarize_results
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "prompting" / "outputs" / "success"
SUMMARY_DIR = PROJECT_ROOT / "prompting" / "outputs" / "summaries"

# Base columns present in every summary CSV (measurement columns appended after)
BASE_COLUMNS = [
    "model_short",
    "model",
    "patient_name",
    "configuration_id",
    "error",
    "error_type",
    "error_message",
    "attempt_number",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "api_duration_ms",
    "finish_reason",
]

# Preferred measurement column order per view; any extra keys discovered
# in the data are appended alphabetically after these.
PREFERRED_COLUMNS: Dict[str, List[str]] = {
    "front": [
        "nasal_tip_deviation_right_mm",
        "chin_point_deviation_right_mm",
    ],
    "profile": [
        "midface",
        "upper_lip",
        "lower_lip",
        "chin_pogonion",
        "maxilla_advancement_mm",
        "maxilla_impaction_mm",
        "maxilla_rotation_right_mm",
        "mandible_advancement_mm",
        "mandible_impaction_mm",
        "mandible_rotation_right_mm",
        "genioplasty_reduction_mm",
        "genioplasty_lengthening_mm",
        "notes",
    ],
}


def load_result_files() -> List[Dict[str, Any]]:
    """Load every .json under RESULTS_DIR/<model>/. Returns list of payloads
    enriched with model_short."""
    payloads: List[Dict[str, Any]] = []
    for model_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        for json_file in sorted(model_dir.glob("*.json")):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping unreadable JSON {json_file}: {exc}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                print(f"[WARN] Skipping non-object JSON {json_file}", file=sys.stderr)
                continue
            payload["model_short"] = model_dir.name
            payload["_source_file"] = str(json_file)
            payloads.append(payload)
    return payloads


def flatten_measurements(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten response_parsed into a flat dict of measurement columns."""
    flat: Dict[str, Any] = {}
    parsed = payload.get("response_parsed")

    if isinstance(parsed, dict):
        view = payload.get("view")
        if view == "front":
            flat.update(parsed)
        elif view == "profile":
            s1 = parsed.get("section_1_diagnostic_analysis") or {}
            s2 = parsed.get("section_2_quantitative_treatment_plan") or {}
            s3 = parsed.get("section_3_additional_recommendations") or {}
            movements = s2.get("movements") or {}
            flat.update(s1)
            flat.update(movements)
            flat.update(s3)
        else:
            # Unknown view: dump everything top-level
            flat.update(parsed)
    else:
        # Model self-reported error stored in response_content (e.g. ruler not legible)
        content = payload.get("response_content")
        if isinstance(content, str):
            try:
                maybe = json.loads(content)
                if isinstance(maybe, dict) and "err" in maybe:
                    flat["model_err"] = maybe["err"]
            except json.JSONDecodeError:
                pass
    return flat


def build_rows(payloads: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group flattened rows by (view, educated)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for payload in payloads:
        view = payload.get("view")
        educated = payload.get("educated")
        if view not in ("front", "profile") or not isinstance(educated, bool):
            print(f"[WARN] Skipping payload with unknown type info: {payload.get('_source_file')}",
                  file=sys.stderr)
            continue

        row: Dict[str, Any] = {
            "model_short": payload.get("model_short", ""),
            "model": payload.get("model", ""),
            "patient_name": payload.get("patient_name", ""),
            "configuration_id": payload.get("configuration_id", ""),
            "error": payload.get("error", ""),
            "error_type": payload.get("error_type", ""),
            "error_message": payload.get("error_message", ""),
            "attempt_number": payload.get("attempt_number", ""),
            "input_tokens": payload.get("input_tokens", ""),
            "output_tokens": payload.get("output_tokens", ""),
            "total_tokens": payload.get("total_tokens", ""),
            "cost_usd": payload.get("cost_usd", ""),
            "api_duration_ms": payload.get("api_duration_ms", ""),
            "finish_reason": payload.get("finish_reason", ""),
        }
        row.update(flatten_measurements(payload))

        key = (view, educated)
        groups.setdefault(key, []).append(row)
    return groups


def column_order(rows: List[Dict[str, Any]], view: str) -> List[str]:
    """Determine final column order: preferred measurement columns first,
    then any extra discovered keys alphabetically."""
    discovered: set = set()
    for row in rows:
        discovered.update(row.keys())
    ordered = [c for c in PREFERRED_COLUMNS.get(view, []) if c in discovered]
    extras = sorted(discovered - set(ordered) - set(BASE_COLUMNS))
    return BASE_COLUMNS + extras + ordered


def write_summary_csv(rows: List[Dict[str, Any]], view: str, educated: bool) -> Path:
    """Write one group of rows to its summary CSV. Returns output path."""
    level = "educated" if educated else "uneducated"
    out_path = SUMMARY_DIR / f"{view}_{level}_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = column_order(rows, view)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Fill missing keys with empty string so DictWriter never raises
            complete = {c: row.get(c, "") for c in columns}
            writer.writerow(complete)
    return out_path


def main() -> None:
    if not RESULTS_DIR.is_dir():
        print(f"Results directory not found: {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    payloads = load_result_files()
    if not payloads:
        print("No result JSON files found. Nothing to summarize.")
        sys.exit(0)

    groups = build_rows(payloads)

    print(f"Loaded {len(payloads)} result files from {RESULTS_DIR}")
    print()
    written: List[Path] = []
    for (view, educated), rows in sorted(groups.items(), reverse=False):
        level = "educated" if educated else "uneducated"
        n_err = sum(1 for r in rows if str(r.get("error")).lower() == "true")
        out_path = write_summary_csv(rows, view, educated)
        written.append(out_path)
        print(f"  {view}_{level}_summary.csv: {len(rows)} rows "
              f"({len(rows) - n_err} ok, {n_err} error) -> {out_path}")
    print()
    print(f"Done. {len(written)} summary CSVs written to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
