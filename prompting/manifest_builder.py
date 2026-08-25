"""
Manifest Builder for the prompting experiment.

Generates a CSV manifest of every (model, view, educated, patient)
configuration that needs to be sent to the LLM API.  The CSV serves as
both the run plan and the checkpoint log (via the `done` column).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "calibrated"

PROMPTS_DIR = PROJECT_ROOT / "prompting" / "prompts"

OUTPUTS_DIR = PROJECT_ROOT / "prompting" / "outputs" / "success"

MODEL_SHORT_NAMES: dict[str, str] = {
    "google/gemini-3.1-pro-preview": "gemini",
    "anthropic/claude-opus-4.8": "claude",
    "openai/gpt-5.6-sol-pro": "gpt",
}

# Ensure output subdirectories exist
for short_name in MODEL_SHORT_NAMES.values():
    (OUTPUTS_DIR / short_name).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_short_name(model_full: str) -> str:
    """Return the short folder name for a full OpenRouter model string."""
    return MODEL_SHORT_NAMES[model_full]


def _generate_id() -> str:
    return f"cfg_{uuid4().hex}"


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One row of the manifest = one API call to make."""

    configuration_id: str = field(default_factory=_generate_id)
    educated: bool = False
    view: str = ""  # "front" | "profile"
    system_prompt_path: str = ""
    model: str = ""  # Full OpenRouter model string
    patient_name: str = ""
    patient_image_path: str = ""
    output_path: str = ""
    done: bool = False

    @property
    def model_short(self) -> str:
        return _model_short_name(self.model)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ManifestBuilder:
    """Build and persist the full manifest of configurations."""

    MODELS: List[str] = [
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4.8",
        "openai/gpt-5.6-sol-pro",
    ]

    VIEWS: List[str] = ["front", "profile"]
    EDUCATION_LEVELS: List[bool] = [True, False]

    def __init__(self, patients: Optional[List[str]] = None) -> None:
        """
        Parameters
        ----------
        patients : list of str, optional
            Patient image file names (e.g. "gamze oeksuez.png").
            If ``None``, they are auto-discovered from ``data/calibrated/profile/``.
        """
        self.patients: List[str] = patients or self._discover_patients()

    # ------------------------------------------------------------------
    # Patient discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_patients() -> List[str]:
        profile_dir = DATA_DIR / "profile"
        if not profile_dir.is_dir():
            return []
        return sorted(f.name for f in profile_dir.iterdir() if f.is_file())

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt_path(educated: bool, view: str) -> str:
        level = "educated" if educated else "uneducated"
        return str(PROMPTS_DIR / f"{level}_{view}.xml")

    @staticmethod
    def _patient_image_path(patient_name: str, view: str) -> str:
        return str(DATA_DIR / view / patient_name)

    # ------------------------------------------------------------------
    # Build full manifest
    # ------------------------------------------------------------------

    def build_all_entries(self) -> List[ManifestEntry]:
        """Generate every configuration and return it as a list of entries."""
        entries: List[ManifestEntry] = []
        for model in self.MODELS:
            for view in self.VIEWS:
                for educated in self.EDUCATION_LEVELS:
                    for patient_name in self.patients:
                        entry = ManifestEntry(
                            educated=educated,
                            view=view,
                            system_prompt_path=self._system_prompt_path(educated, view),
                            model=model,
                            patient_name=Path(patient_name).stem,
                            patient_image_path=self._patient_image_path(patient_name, view),
                        )
                        entry.output_path = self._output_path(model, view, educated, entry.configuration_id)
                        entries.append(entry)
        return entries

    @staticmethod
    def _output_path(model_full: str, view: str, educated: bool, configuration_id: str) -> str:
        short = _model_short_name(model_full)
        level = "educated" if educated else "uneducated"
        return str(OUTPUTS_DIR / short / f"{view}_{level}_{configuration_id}.json")

    # ------------------------------------------------------------------
    # CSV persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_csv(entries: List[ManifestEntry], path: str | Path) -> None:
        """Write the manifest to a CSV file (always overwrites)."""
        if not entries:
            raise ValueError("Cannot save an empty manifest.")

        field_names = list(asdict(entries[0]).keys())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for entry in entries:
                writer.writerow(asdict(entry))

    @staticmethod
    def load_csv(path: str | Path) -> List[ManifestEntry]:
        """Load a previously saved manifest CSV, restoring ``done`` state."""
        entries: List[ManifestEntry] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = ManifestEntry(
                    configuration_id=row["configuration_id"],
                    educated=row["educated"].strip().lower() == "true",
                    view=row["view"],
                    system_prompt_path=row["system_prompt_path"],
                    model=row["model"],
                    patient_name=row["patient_name"],
                    patient_image_path=row["patient_image_path"],
                    output_path=row["output_path"],
                    done=row["done"].strip().lower() == "true",
                )
                entries.append(entry)
        return entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    builder = ManifestBuilder()
    all_entries = builder.build_all_entries()

    csv_path = PROJECT_ROOT / "prompting" / "manifest.csv"
    ManifestBuilder.save_csv(all_entries, csv_path)
    print(f"Manifest saved to {csv_path}")
    print(f"Total configurations: {len(all_entries)}")
    sys.exit(0)
