"""
07_resource_utilization.py
Analyzes secondary resource consumption metrics:
  - Input tokens, Output tokens, Total tokens, Cost (USD), API Response Time (seconds)
Across models and conditions:
  1. Descriptive statistics (Mean, SD, Median, IQR, Min, Max)
  2. Kruskal-Wallis H-tests comparing models within each condition (with Dunn/Mann-Whitney post-hoc)
  3. Paired Wilcoxon signed-rank tests comparing Educated vs Uneducated conditions within each model

Outputs:
  - statistics/outputs/07_resource_descriptives.csv
  - statistics/outputs/07_resource_kruskal_tests.csv
  - statistics/outputs/07_resource_condition_wilcoxon.csv
"""

import sys
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_resource_data, OUTPUTS_DIR

MODELS = ["claude", "deepseek", "gemini", "gpt"]
RESOURCE_VARS = ["input_tokens", "output_tokens", "total_tokens", "cost_usd", "api_duration_sec"]


def compute_resource_descriptives(df):
    records = []
    for (model, cond), group in df.groupby(["model_short", "condition"]):
        for var in RESOURCE_VARS:
            vals = group[var].dropna().values
            n = len(vals)
            if n == 0:
                continue

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            median_val = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr_val = q75 - q25

            records.append({
                "model": model,
                "condition": cond,
                "metric": var,
                "n": n,
                "mean": round(mean_val, 4),
                "sd": round(sd_val, 4),
                "median": round(median_val, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr_val, 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "formatted_median_iqr": f"{median_val:.2f} ({iqr_val:.2f})",
                "formatted_mean_sd": f"{mean_val:.2f} ± {sd_val:.2f}",
            })

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUTS_DIR / "07_resource_descriptives.csv", index=False)
    print(f"Saved 07_resource_descriptives.csv ({len(res_df)} rows)")


def compute_kruskal_model_comparisons(df):
    records = []
    for cond in ["educated", "uneducated"]:
        cond_df = df[df["condition"] == cond]
        for var in RESOURCE_VARS:
            # Group values by model
            model_arrays = []
            for m in MODELS:
                vals = cond_df[cond_df["model_short"] == m][var].dropna().values
                if len(vals) > 0:
                    model_arrays.append(vals)

            if len(model_arrays) == len(MODELS):
                h_stat, p_val = stats.kruskal(*model_arrays)
                records.append({
                    "condition": cond,
                    "metric": var,
                    "k_models": len(MODELS),
                    "kruskal_h": round(float(h_stat), 4),
                    "df": len(MODELS) - 1,
                    "p_val": round(float(p_val), 5),
                    "is_significant": bool(p_val < 0.05),
                })

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUTS_DIR / "07_resource_kruskal_tests.csv", index=False)
    print(f"Saved 07_resource_kruskal_tests.csv ({len(res_df)} rows)")


def compute_wilcoxon_condition_comparisons(df):
    records = []
    for model in MODELS:
        m_df = df[df["model_short"] == model]

        for var in RESOURCE_VARS:
            # Match by patient_name and view
            pivot = m_df.pivot(index=["patient_name", "view"], columns="condition", values=var).dropna()

            if len(pivot) < 5 or "educated" not in pivot.columns or "uneducated" not in pivot.columns:
                continue

            edu_vals = pivot["educated"].values
            unedu_vals = pivot["uneducated"].values
            diffs = edu_vals - unedu_vals
            n = len(diffs)

            mean_edu = float(np.mean(edu_vals))
            mean_unedu = float(np.mean(unedu_vals))
            med_edu = float(np.median(edu_vals))
            med_unedu = float(np.median(unedu_vals))

            pct_change = ((med_edu - med_unedu) / med_unedu * 100.0) if med_unedu != 0 else np.nan

            if np.all(diffs == 0):
                w_stat, w_pval = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(edu_vals, unedu_vals)
                    w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, w_pval = np.nan, np.nan

            records.append({
                "model": model,
                "metric": var,
                "n_pairs": n,
                "median_educated": round(med_edu, 4),
                "median_uneducated": round(med_unedu, 4),
                "pct_change_median": round(pct_change, 2) if not np.isnan(pct_change) else np.nan,
                "mean_educated": round(mean_edu, 4),
                "mean_uneducated": round(mean_unedu, 4),
                "wilcoxon_w": round(w_stat, 4) if not np.isnan(w_stat) else np.nan,
                "p_val": round(w_pval, 5) if not np.isnan(w_pval) else np.nan,
                "is_significant": bool(w_pval < 0.05) if not np.isnan(w_pval) else False,
            })

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUTS_DIR / "07_resource_condition_wilcoxon.csv", index=False)
    print(f"Saved 07_resource_condition_wilcoxon.csv ({len(res_df)} rows)")


if __name__ == "__main__":
    print("Running 07_resource_utilization...")
    res_data = load_resource_data()
    if res_data.empty:
        print("Error: No resource data loaded from prompting/outputs/summaries/.")
    else:
        # Filter models and add api_duration_sec
        res_data = res_data[res_data["model_short"].isin(MODELS)].copy()
        if "api_duration_ms" in res_data.columns:
            res_data["api_duration_sec"] = res_data["api_duration_ms"] / 1000.0
        else:
            res_data["api_duration_sec"] = np.nan

        compute_resource_descriptives(res_data)
        compute_kruskal_model_comparisons(res_data)
        compute_wilcoxon_condition_comparisons(res_data)
        print("Completed 07_resource_utilization.")
