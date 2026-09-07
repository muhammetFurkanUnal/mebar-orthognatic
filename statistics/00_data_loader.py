"""
00_data_loader.py
Data loading and preprocessing utilities for the orthognathic AI analysis pipeline.
Handles European decimal commas, missing values, standardizes column names,
and structures datasets in both wide and long formats.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
SUMMARIES_DIR = PROJECT_ROOT / "prompting" / "outputs" / "summaries"
OUTPUTS_DIR = PROJECT_ROOT / "statistics" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_numeric(val):
    """Safely convert strings with commas or signs to float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace('"', '').replace("'", "")
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return np.nan
    # Handle Turkish/European comma format
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_str(val):
    """Normalize categorical strings."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    return s if s not in ["", "nan", "none", "null"] else np.nan


def load_front_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load frontal data for educated and uneducated conditions.
    Columns: model_short, patient_name, nasal_tip_deviation_right_mm, chin_point_deviation_right_mm
    """
    path_edu = REFERENCE_DIR / "karşılaştırmalar.xlsx - front_educated.csv"
    path_unedu = REFERENCE_DIR / "karşılaştırmalar.xlsx - front_uneducated.csv"

    df_edu = pd.read_csv(path_edu)
    df_unedu = pd.read_csv(path_unedu)

    num_cols = ["nasal_tip_deviation_right_mm", "chin_point_deviation_right_mm"]
    for c in num_cols:
        df_edu[c] = df_edu[c].apply(parse_numeric)
        df_unedu[c] = df_unedu[c].apply(parse_numeric)

    df_edu["patient_name"] = df_edu["patient_name"].str.strip()
    df_unedu["patient_name"] = df_unedu["patient_name"].str.strip()

    df_edu["condition"] = "educated"
    df_unedu["condition"] = "uneducated"
    df_edu["view"] = "front"
    df_unedu["view"] = "front"

    return df_edu, df_unedu


def load_profile_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load profile data for educated and uneducated conditions.
    Categorical: midface, upper_lip, lower_lip, chin_pogonion
    Continuous: maxilla_advancement_mm, maxilla_impaction_mm,
                mandible_advancement_mm(POG), mandible_impaction_mm(POG)
    """
    path_edu = REFERENCE_DIR / "karşılaştırmalar.xlsx - profile_educated.csv"
    path_unedu = REFERENCE_DIR / "karşılaştırmalar.xlsx - profile_uneducated.csv"

    df_edu = pd.read_csv(path_edu)
    df_unedu = pd.read_csv(path_unedu)

    # Standardize column names (replace spaces and parenthesis)
    rename_map = {
        "mandible_advancement_mm(POG)": "mandible_advancement_mm",
        "mandible_impaction_mm(POG)": "mandible_impaction_mm",
    }
    df_edu = df_edu.rename(columns=rename_map)
    df_unedu = df_unedu.rename(columns=rename_map)

    num_cols = [
        "maxilla_advancement_mm",
        "maxilla_impaction_mm",
        "mandible_advancement_mm",
        "mandible_impaction_mm",
    ]
    for c in num_cols:
        df_edu[c] = df_edu[c].apply(parse_numeric)
        df_unedu[c] = df_unedu[c].apply(parse_numeric)

    cat_cols = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    for c in cat_cols:
        df_edu[c] = df_edu[c].apply(clean_str)
        df_unedu[c] = df_unedu[c].apply(clean_str)

    df_edu["patient_name"] = df_edu["patient_name"].str.strip()
    df_unedu["patient_name"] = df_unedu["patient_name"].str.strip()

    df_edu["condition"] = "educated"
    df_unedu["condition"] = "uneducated"
    df_edu["view"] = "profile"
    df_unedu["view"] = "profile"

    return df_edu, df_unedu


def load_resource_data() -> pd.DataFrame:
    """
    Load and combine execution resource summaries from prompting/outputs/summaries/
    Columns include: model_short, model, patient_name, condition, view,
    input_tokens, output_tokens, total_tokens, cost_usd, api_duration_ms, error, error_type
    """
    files = {
        ("educated", "front"): SUMMARIES_DIR / "front_educated_summary.csv",
        ("uneducated", "front"): SUMMARIES_DIR / "front_uneducated_summary.csv",
        ("educated", "profile"): SUMMARIES_DIR / "profile_educated_summary.csv",
        ("uneducated", "profile"): SUMMARIES_DIR / "profile_uneducated_summary.csv",
    }

    dfs = []
    for (cond, view), fpath in files.items():
        if fpath.exists():
            df = pd.read_csv(fpath)
            df["condition"] = cond
            df["view"] = view
            df["patient_name"] = df["patient_name"].str.strip()
            num_cols = ["input_tokens", "output_tokens", "total_tokens", "cost_usd", "api_duration_ms"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "error" in df.columns:
                df["is_error"] = df["error"].astype(str).str.lower().isin(["true", "1"])
            else:
                df["is_error"] = False
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def build_unified_long_dataset() -> pd.DataFrame:
    """
    Build a long-format dataframe containing all continuous and categorical predictions
    aligned with reference ground truth where applicable.
    """
    front_edu, front_unedu = load_front_data()
    prof_edu, prof_unedu = load_profile_data()

    # Combine front
    front_all = pd.concat([front_edu, front_unedu], ignore_index=True)
    # Combine profile
    prof_all = pd.concat([prof_edu, prof_unedu], ignore_index=True)

    # Merge on patient_name, model_short, condition
    merged = pd.merge(
        prof_all,
        front_all[["patient_name", "model_short", "condition", "nasal_tip_deviation_right_mm", "chin_point_deviation_right_mm"]],
        on=["patient_name", "model_short", "condition"],
        how="outer"
    )
    return merged


if __name__ == "__main__":
    print("Testing 00_data_loader...")
    f_edu, f_unedu = load_front_data()
    p_edu, p_unedu = load_profile_data()
    res = load_resource_data()
    unified = build_unified_long_dataset()
    print(f"Frontal Educated: {f_edu.shape}, Uneducated: {f_unedu.shape}")
    print(f"Profile Educated: {p_edu.shape}, Uneducated: {p_unedu.shape}")
    print(f"Resource data: {res.shape}")
    print(f"Unified data: {unified.shape}")
    print("DataLoader test passed successfully!")
