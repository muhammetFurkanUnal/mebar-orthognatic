"""
02_continuous_agreement_vs_reference.py
Evaluates agreement and accuracy between AI model predictions and the Arnett reference plan
for continuous surgical movements (maxilla advancement/impaction, mandible advancement/impaction).

Computes:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Bland-Altman analysis (Bias, SD, 95% Limits of Agreement [LoA], 95% CIs)
  - Intraclass Correlation Coefficient (ICC, two-way mixed, single rater, absolute agreement - ICC(2,1))
  - Lin's Concordance Correlation Coefficient (CCC - rho_c) and 95% CI

Outputs:
  - statistics/outputs/02_continuous_vs_reference_metrics.csv
  - statistics/outputs/02_bland_altman_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_profile_data, OUTPUTS_DIR


def lins_ccc(x, y):
    """Calculate Lin's Concordance Correlation Coefficient and its 95% CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    sd_x = np.sqrt(var_x)
    sd_y = np.sqrt(var_y)

    if sd_x == 0 or sd_y == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    rho = cov_xy / (sd_x * sd_y)
    # Bias correction factor (accuracy)
    v = sd_x / sd_y
    u = (mean_x - mean_y) / np.sqrt(sd_x * sd_y)
    c_b = 2 / (v + (1 / v) + (u ** 2))
    rho_c = rho * c_b

    # Asymptotic standard error using Fisher's z transform
    z = np.arctanh(np.clip(rho_c, -0.9999, 0.9999))
    # Approximation of SE for z
    se_z = np.sqrt(1 / (n - 2))
    z_low = z - 1.96 * se_z
    z_high = z + 1.96 * se_z
    ci_low = np.tanh(z_low)
    ci_high = np.tanh(z_high)

    return float(rho_c), float(ci_low), float(ci_high), float(c_b), float(rho)


def compute_continuous_agreement():
    prof_edu, prof_unedu = load_profile_data()

    movement_vars = [
        "maxilla_advancement_mm",
        "maxilla_impaction_mm",
        "mandible_advancement_mm",
        "mandible_impaction_mm",
    ]

    models = ["claude", "deepseek", "gemini", "gpt"]
    conditions = [("educated", prof_edu), ("uneducated", prof_unedu)]

    metrics_records = []
    ba_records = []

    for cond_name, df_cond in conditions:
        ref_df = df_cond[df_cond["model_short"] == "reference"].copy()
        ref_dict = ref_df.set_index("patient_name")

        for model in models:
            m_df = df_cond[df_cond["model_short"] == model].copy()
            m_dict = m_df.set_index("patient_name")

            # Common patients
            common_patients = [p for p in ref_dict.index if p in m_dict.index]

            for var_name in movement_vars:
                paired_data = []
                for p in common_patients:
                    ref_val = ref_dict.loc[p, var_name]
                    ai_val = m_dict.loc[p, var_name]
                    if pd.notna(ref_val) and pd.notna(ai_val):
                        paired_data.append((p, float(ai_val), float(ref_val)))

                if len(paired_data) < 3:
                    continue

                p_ids = [d[0] for d in paired_data]
                ai_vals = np.array([d[1] for d in paired_data])
                ref_vals = np.array([d[2] for d in paired_data])
                n = len(ai_vals)

                diff = ai_vals - ref_vals  # AI - Reference (Error)
                abs_diff = np.abs(diff)

                mae = float(np.mean(abs_diff))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                bias = float(np.mean(diff))
                sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

                # Bland-Altman LoA
                lower_loa = bias - 1.96 * sd_diff
                upper_loa = bias + 1.96 * sd_diff

                # 95% CIs for Bland-Altman
                t_val = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
                se_bias = sd_diff / np.sqrt(n)
                ci_bias_low = bias - t_val * se_bias
                ci_bias_high = bias + t_val * se_bias

                se_loa = np.sqrt(3 * (sd_diff ** 2) / n)
                ci_lower_loa_low = lower_loa - t_val * se_loa
                ci_lower_loa_high = lower_loa + t_val * se_loa
                ci_upper_loa_low = upper_loa - t_val * se_loa
                ci_upper_loa_high = upper_loa + t_val * se_loa

                # Intraclass Correlation Coefficient ICC(2,1) using Pingouin
                # Create long dataframe for pingouin
                icc_df = pd.DataFrame({
                    "patient": p_ids * 2,
                    "rater": ["ai"] * n + ["reference"] * n,
                    "score": np.concatenate([ai_vals, ref_vals])
                })
                try:
                    icc_res = pg.intraclass_corr(data=icc_df, targets="patient", raters="rater", ratings="score")
                    # ICC(A,1) is two-way random, single rater, absolute agreement
                    row_matches = icc_res[icc_res["Type"].str.contains(r"\(A,1\)|ICC2\b", regex=True)]
                    if not row_matches.empty:
                        icc2 = row_matches.iloc[0]
                        icc_val = float(icc2["ICC"])
                        ci_col = "CI95" if "CI95" in icc2 else "CI95%"
                        icc_ci_low = float(icc2[ci_col][0])
                        icc_ci_high = float(icc2[ci_col][1])
                        icc_pval = float(icc2["pval"])
                    else:
                        icc_val, icc_ci_low, icc_ci_high, icc_pval = np.nan, np.nan, np.nan, np.nan
                except Exception as e:
                    icc_val, icc_ci_low, icc_ci_high, icc_pval = np.nan, np.nan, np.nan, np.nan

                # Lin's Concordance Correlation Coefficient
                ccc_val, ccc_ci_low, ccc_ci_high, cb_val, pearson_rho = lins_ccc(ai_vals, ref_vals)

                # Append to records
                metrics_records.append({
                    "model": model,
                    "condition": cond_name,
                    "variable": var_name,
                    "n": n,
                    "mae": round(mae, 3),
                    "rmse": round(rmse, 3),
                    "bias": round(bias, 3),
                    "sd_diff": round(sd_diff, 3),
                    "icc_2_1": round(icc_val, 4) if not np.isnan(icc_val) else np.nan,
                    "icc_ci95_low": round(icc_ci_low, 4) if not np.isnan(icc_ci_low) else np.nan,
                    "icc_ci95_high": round(icc_ci_high, 4) if not np.isnan(icc_ci_high) else np.nan,
                    "icc_pval": round(icc_pval, 5) if not np.isnan(icc_pval) else np.nan,
                    "lin_ccc": round(ccc_val, 4) if not np.isnan(ccc_val) else np.nan,
                    "lin_ccc_ci95_low": round(ccc_ci_low, 4) if not np.isnan(ccc_ci_low) else np.nan,
                    "lin_ccc_ci95_high": round(ccc_ci_high, 4) if not np.isnan(ccc_ci_high) else np.nan,
                    "accuracy_cb": round(cb_val, 4) if not np.isnan(cb_val) else np.nan,
                    "pearson_rho": round(pearson_rho, 4) if not np.isnan(pearson_rho) else np.nan,
                })

                ba_records.append({
                    "model": model,
                    "condition": cond_name,
                    "variable": var_name,
                    "n": n,
                    "bias": round(bias, 3),
                    "ci_bias_low": round(ci_bias_low, 3),
                    "ci_bias_high": round(ci_bias_high, 3),
                    "sd_diff": round(sd_diff, 3),
                    "lower_loa": round(lower_loa, 3),
                    "ci_lower_loa_low": round(ci_lower_loa_low, 3),
                    "ci_lower_loa_high": round(ci_lower_loa_high, 3),
                    "upper_loa": round(upper_loa, 3),
                    "ci_upper_loa_low": round(ci_upper_loa_low, 3),
                    "ci_upper_loa_high": round(ci_upper_loa_high, 3),
                })

    df_metrics = pd.DataFrame(metrics_records)
    df_ba = pd.DataFrame(ba_records)

    df_metrics.to_csv(OUTPUTS_DIR / "02_continuous_vs_reference_metrics.csv", index=False)
    df_ba.to_csv(OUTPUTS_DIR / "02_bland_altman_summary.csv", index=False)
    print(f"Saved 02_continuous_vs_reference_metrics.csv ({len(df_metrics)} rows)")
    print(f"Saved 02_bland_altman_summary.csv ({len(df_ba)} rows)")


if __name__ == "__main__":
    print("Running 02_continuous_agreement_vs_reference...")
    compute_continuous_agreement()
    print("Completed 02_continuous_agreement_vs_reference.")
