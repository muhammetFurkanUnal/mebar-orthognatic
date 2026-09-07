"""
01_descriptive_and_normality.py
Calculates descriptive statistics (mean, SD, median, IQR, min, max, skewness, kurtosis)
and tests for normality (Shapiro-Wilk) for all continuous variables across models,
conditions (educated/uneducated), and views (frontal/profile).
Also computes frequency and percentage distributions for categorical soft tissue variables.

Outputs:
  - statistics/outputs/01_descriptive_continuous.csv
  - statistics/outputs/01_descriptive_categorical.csv
  - statistics/outputs/01_normality_tests.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import (
    load_front_data,
    load_profile_data,
    OUTPUTS_DIR,
)


def compute_continuous_descriptives():
    front_edu, front_unedu = load_front_data()
    prof_edu, prof_unedu = load_profile_data()

    all_data = pd.concat([front_edu, front_unedu, prof_edu, prof_unedu], ignore_index=True)

    # Merge rotation columns from summaries if available
    sum_edu_path = Path(__file__).resolve().parent.parent / "prompting" / "outputs" / "summaries" / "profile_educated_summary.csv"
    sum_unedu_path = Path(__file__).resolve().parent.parent / "prompting" / "outputs" / "summaries" / "profile_uneducated_summary.csv"
    
    prof_rot_list = []
    if sum_edu_path.exists():
        d_se = pd.read_csv(sum_edu_path)[["model_short", "patient_name", "maxilla_rotation_right_mm", "mandible_rotation_right_mm"]].copy()
        d_se["condition"] = "educated"
        d_se["view"] = "profile"
        prof_rot_list.append(d_se)
    if sum_unedu_path.exists():
        d_su = pd.read_csv(sum_unedu_path)[["model_short", "patient_name", "maxilla_rotation_right_mm", "mandible_rotation_right_mm"]].copy()
        d_su["condition"] = "uneducated"
        d_su["view"] = "profile"
        prof_rot_list.append(d_su)

    if prof_rot_list:
        rot_all = pd.concat(prof_rot_list, ignore_index=True)
        all_data = pd.merge(all_data, rot_all, on=["model_short", "patient_name", "condition", "view"], how="left")

    continuous_vars = {
        "nasal_tip_deviation_right_mm": "front",
        "chin_point_deviation_right_mm": "front",
        "maxilla_advancement_mm": "profile",
        "maxilla_impaction_mm": "profile",
        "mandible_advancement_mm": "profile",
        "mandible_impaction_mm": "profile",
        "maxilla_rotation_right_mm": "profile",
        "mandible_rotation_right_mm": "profile",
    }

    desc_records = []
    norm_records = []

    for var_name, expected_view in continuous_vars.items():
        sub_df = all_data[all_data["view"] == expected_view]
        # Group by model_short and condition
        for (model, cond), group in sub_df.groupby(["model_short", "condition"]):
            vals = group[var_name].dropna().values
            n = len(vals)

            if n == 0:
                continue

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            median_val = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr_val = q75 - q25
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            skew_val = float(stats.skew(vals, bias=False)) if n > 2 else np.nan
            kurt_val = float(stats.kurtosis(vals, bias=False)) if n > 3 else np.nan

            desc_records.append({
                "view": expected_view,
                "variable": var_name,
                "model": model,
                "condition": cond,
                "n": n,
                "mean": round(mean_val, 3),
                "sd": round(sd_val, 3),
                "median": round(median_val, 3),
                "q25": round(q25, 3),
                "q75": round(q75, 3),
                "iqr": round(iqr_val, 3),
                "min": round(min_val, 3),
                "max": round(max_val, 3),
                "skewness": round(skew_val, 3) if not np.isnan(skew_val) else np.nan,
                "kurtosis": round(kurt_val, 3) if not np.isnan(kurt_val) else np.nan,
                "formatted_mean_sd": f"{mean_val:.2f} ± {sd_val:.2f}",
                "formatted_median_iqr": f"{median_val:.2f} ({iqr_val:.2f})"
            })

            # Shapiro-Wilk Test (requires at least 3 values and non-constant data)
            if n >= 3 and np.ptp(vals) > 1e-8:
                w_stat, p_val = stats.shapiro(vals)
                is_normal = bool(p_val >= 0.05)
            else:
                w_stat, p_val, is_normal = np.nan, np.nan, False

            norm_records.append({
                "view": expected_view,
                "variable": var_name,
                "model": model,
                "condition": cond,
                "n": n,
                "shapiro_w": round(float(w_stat), 4) if not np.isnan(w_stat) else np.nan,
                "shapiro_p": round(float(p_val), 5) if not np.isnan(p_val) else np.nan,
                "is_normal": is_normal
            })

    df_desc = pd.DataFrame(desc_records)
    df_norm = pd.DataFrame(norm_records)

    df_desc.to_csv(OUTPUTS_DIR / "01_descriptive_continuous.csv", index=False)
    df_norm.to_csv(OUTPUTS_DIR / "01_normality_tests.csv", index=False)
    print(f"Saved 01_descriptive_continuous.csv ({len(df_desc)} rows)")
    print(f"Saved 01_normality_tests.csv ({len(df_norm)} rows)")


def compute_categorical_descriptives():
    prof_edu, prof_unedu = load_profile_data()
    all_prof = pd.concat([prof_edu, prof_unedu], ignore_index=True)

    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    categories = ["hypoplastic", "normognathic", "hyperplastic"]

    cat_records = []

    for var_name in cat_vars:
        for (model, cond), group in all_prof.groupby(["model_short", "condition"]):
            total_n = len(group)
            counts = group[var_name].value_counts(dropna=False).to_dict()

            for cat in categories:
                cnt = counts.get(cat, 0)
                pct = (cnt / total_n * 100) if total_n > 0 else 0.0
                cat_records.append({
                    "variable": var_name,
                    "model": model,
                    "condition": cond,
                    "category": cat,
                    "count": cnt,
                    "total": total_n,
                    "percentage": round(pct, 2),
                    "formatted_n_pct": f"{cnt} ({pct:.1f}%)"
                })

            # Check missing / unclassified
            missing_cnt = sum(counts.get(k, 0) for k in counts if k not in categories)
            if missing_cnt > 0:
                pct = (missing_cnt / total_n * 100)
                cat_records.append({
                    "variable": var_name,
                    "model": model,
                    "condition": cond,
                    "category": "missing_or_refusal",
                    "count": missing_cnt,
                    "total": total_n,
                    "percentage": round(pct, 2),
                    "formatted_n_pct": f"{missing_cnt} ({pct:.1f}%)"
                })

    df_cat = pd.DataFrame(cat_records)
    df_cat.to_csv(OUTPUTS_DIR / "01_descriptive_categorical.csv", index=False)
    print(f"Saved 01_descriptive_categorical.csv ({len(df_cat)} rows)")


if __name__ == "__main__":
    print("Running 01_descriptive_and_normality...")
    compute_continuous_descriptives()
    compute_categorical_descriptives()
    print("Completed 01_descriptive_and_normality.")
