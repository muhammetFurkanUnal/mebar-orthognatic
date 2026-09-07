"""
run_all.py
Executes the entire statistical analysis pipeline sequentially (Scripts 01 through 08).
"""

import sys
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "01_descriptive_and_normality.py",
    "02_continuous_agreement_vs_reference.py",
    "03_categorical_agreement_vs_reference.py",
    "04_condition_paired_comparisons.py",
    "05_multimodel_comparisons.py",
    "06_inter_rater_reliability_and_refusals.py",
    "07_resource_utilization.py",
    "08_mixed_effects_models.py",
    "09_generate_plots.py",
    "10_generate_report.py",
]


def run_pipeline():
    print("=" * 70)
    print("STARTING ORTHOGNATHIC AI STATISTICAL ANALYSIS PIPELINE")
    print("=" * 70)

    for script in SCRIPTS:
        script_path = CURRENT_DIR / script
        print(f"\n---> Running: {script}")
        res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip())
        else:
            print(f"Error in {script}:")
            print(res.stderr)
            sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL STATISTICAL SCRIPTS EXECUTED SUCCESSFULLY!")
    print(f"Outputs written to: {CURRENT_DIR / 'outputs'}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
