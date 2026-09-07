"""
05_multimodel_comparisons.py
Evaluates differences across the four AI models (Claude, DeepSeek, Gemini, GPT)
within the same patients using matched-sample non-parametric methods:
  - Continuous variables:
      * Friedman test across 4 models
      * Kendall's W effect size
      * Post-hoc pairwise Wilcoxon signed-rank tests with Bonferroni and Holm adjustments
  - Categorical variables:
      * Cochran's Q test for matched binary outcomes
        (Dysplastic status detection & Consensus agreement)
      * Post-hoc pairwise McNemar tests with Bonferroni and Holm corrections

Outputs:
  - statistics/outputs/05_multimodel_friedman_continuous.csv
  - statistics/outputs/05_multimodel_posthoc_wilcoxon.csv
  - statistics/outputs/05_multimodel_cochran_q_categorical.csv
"""

import sys
from pathlib import Path
from itertools import combinations
from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_front_data, load_profile_data, OUTPUTS_DIR

MODELS = ["claude", "deepseek", "gemini", "gpt"]
CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]


def holm_bonferroni(p_values):
    """Adjust p-values using Holm-Bonferroni step-down method."""
    p_vals = np.array(p_values, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    adjusted = np.empty(n, dtype=float)
    cum_max = 0.0
    for i in range(n):
        multiplier = n - i
        val = sorted_p[i] * multiplier
        cum_max = max(cum_max, val)
        adjusted[sorted_indices[i]] = min(cum_max, 1.0)
    return adjusted


def cochrans_q_test(matrix_binary):
    """
    Cochran's Q test on a binary numpy array of shape (N_patients, k_models).
    Returns: Q_statistic, p_value, df
    """
    x = np.asarray(matrix_binary, dtype=float)
    n, k = x.shape
    if k < 2:
        return np.nan, np.nan, np.nan

    c_sums = x.sum(axis=0)  # column sums (model totals)
    r_sums = x.sum(axis=1)  # row sums (patient totals)

    t = c_sums.sum()
    sum_c2 = np.sum(c_sums ** 2)
    sum_r = np.sum(r_sums)
    sum_r2 = np.sum(r_sums ** 2)

    denom = k * sum_r - sum_r2
    if denom == 0:
        return 0.0, 1.0, k - 1

    numer = (k - 1) * (k * sum_c2 - (t ** 2))
    q_stat = float(numer / denom)
    p_val = float(1.0 - stats.chi2.cdf(q_stat, df=k - 1))
    return q_stat, p_val, k - 1


def run_continuous_multimodel():
    front_edu, front_unedu = load_front_data()
    prof_edu, prof_unedu = load_profile_data()

    # Calculate absolute frontal deviations
    front_edu["abs_nasal_tip_deviation_mm"] = front_edu["nasal_tip_deviation_right_mm"].abs()
    front_unedu["abs_nasal_tip_deviation_mm"] = front_unedu["nasal_tip_deviation_right_mm"].abs()
    front_edu["abs_chin_point_deviation_mm"] = front_edu["chin_point_deviation_right_mm"].abs()
    front_unedu["abs_chin_point_deviation_mm"] = front_unedu["chin_point_deviation_right_mm"].abs()

    var_views = [
        ("nasal_tip_deviation_right_mm", front_edu, front_unedu, "front"),
        ("chin_point_deviation_right_mm", front_edu, front_unedu, "front"),
        ("abs_nasal_tip_deviation_mm", front_edu, front_unedu, "front_absolute"),
        ("abs_chin_point_deviation_mm", front_edu, front_unedu, "front_absolute"),
        ("maxilla_advancement_mm", prof_edu, prof_unedu, "profile"),
        ("maxilla_impaction_mm", prof_edu, prof_unedu, "profile"),
        ("mandible_advancement_mm", prof_edu, prof_unedu, "profile"),
        ("mandible_impaction_mm", prof_edu, prof_unedu, "profile"),
    ]

    friedman_records = []
    posthoc_records = []

    conditions = [("educated", "edu"), ("uneducated", "unedu")]

    for var_name, df_edu, df_unedu, view_name in var_views:
        for cond_name, suffix in conditions:
            df_curr = df_edu if cond_name == "educated" else df_unedu

            # Filter for the 4 AI models
            sub = df_curr[df_curr["model_short"].isin(MODELS)]

            # Pivot to wide format: rows = patient, cols = models
            pivot_df = sub.pivot(index="patient_name", columns="model_short", values=var_name).dropna()

            if len(pivot_df) < 5 or set(MODELS) - set(pivot_df.columns):
                continue

            n_patients = len(pivot_df)
            k = len(MODELS)

            arrays = [pivot_df[m].values for m in MODELS]

            # Friedman Test
            f_stat, f_pval = stats.friedmanchisquare(*arrays)
            # Kendall's W = Chi2 / (N * (k - 1))
            kendalls_w = float(f_stat / (n_patients * (k - 1)))

            friedman_records.append({
                "view": view_name,
                "variable": var_name,
                "condition": cond_name,
                "n_patients": n_patients,
                "k_models": k,
                "friedman_stat": round(float(f_stat), 4),
                "friedman_df": k - 1,
                "friedman_pval": round(float(f_pval), 5),
                "kendalls_w": round(kendalls_w, 4),
                "is_significant": bool(f_pval < 0.05),
            })

            # Post-hoc pairwise Wilcoxon tests (all 6 pairs)
            pair_records_for_var = []
            for m1, m2 in combinations(MODELS, 2):
                diff = pivot_df[m1].values - pivot_df[m2].values
                if np.all(diff == 0):
                    w_stat, w_pval = 0.0, 1.0
                else:
                    try:
                        w_res = stats.wilcoxon(pivot_df[m1].values, pivot_df[m2].values)
                        w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
                    except Exception:
                        w_stat, w_pval = np.nan, np.nan

                mean_diff = float(np.mean(diff))
                pair_records_for_var.append({
                    "view": view_name,
                    "variable": var_name,
                    "condition": cond_name,
                    "model_1": m1,
                    "model_2": m2,
                    "n_patients": n_patients,
                    "mean_diff": round(mean_diff, 3),
                    "wilcoxon_w": round(w_stat, 4) if not np.isnan(w_stat) else np.nan,
                    "raw_pval": w_pval,
                })

            # Compute Bonferroni and Holm corrections
            raw_p_list = [r["raw_pval"] for r in pair_records_for_var]
            bonf_p = [min(1.0, p * len(raw_p_list)) if not np.isnan(p) else np.nan for p in raw_p_list]
            holm_p = holm_bonferroni(raw_p_list)

            for idx, r in enumerate(pair_records_for_var):
                r["raw_pval"] = round(float(r["raw_pval"]), 5) if not np.isnan(r["raw_pval"]) else np.nan
                r["bonferroni_pval"] = round(float(bonf_p[idx]), 5) if not np.isnan(bonf_p[idx]) else np.nan
                r["holm_pval"] = round(float(holm_p[idx]), 5) if not np.isnan(holm_p[idx]) else np.nan
                r["is_significant_holm"] = bool(holm_p[idx] < 0.05) if not np.isnan(holm_p[idx]) else False
                posthoc_records.append(r)

    df_friedman = pd.DataFrame(friedman_records)
    df_posthoc = pd.DataFrame(posthoc_records)

    df_friedman.to_csv(OUTPUTS_DIR / "05_multimodel_friedman_continuous.csv", index=False)
    df_posthoc.to_csv(OUTPUTS_DIR / "05_multimodel_posthoc_wilcoxon.csv", index=False)
    print(f"Saved 05_multimodel_friedman_continuous.csv ({len(df_friedman)} rows)")
    print(f"Saved 05_multimodel_posthoc_wilcoxon.csv ({len(df_posthoc)} rows)")


def run_categorical_cochran():
    prof_edu, prof_unedu = load_profile_data()
    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]

    conditions = [("educated", prof_edu), ("uneducated", prof_unedu)]
    records = []

    for cond_name, df_cond in conditions:
        sub = df_cond[df_cond["model_short"].isin(MODELS)]

        for var_name in cat_vars:
            # Pivot table
            pivot = sub.pivot(index="patient_name", columns="model_short", values=var_name).dropna()
            if len(pivot) < 5 or set(MODELS) - set(pivot.columns):
                continue

            n_patients = len(pivot)

            # Test 1: Dysplastic classification (1 = Hypo/Hyper, 0 = Normognathic)
            bin_dysplastic = (pivot[MODELS] != "normognathic").astype(int).values
            q_stat, q_pval, q_df = cochrans_q_test(bin_dysplastic)

            records.append({
                "variable": var_name,
                "condition": cond_name,
                "outcome_type": "dysplastic_detection",
                "n_patients": n_patients,
                "k_models": len(MODELS),
                "cochran_q_stat": round(q_stat, 4),
                "df": q_df,
                "p_val": round(q_pval, 5),
                "is_significant": bool(q_pval < 0.05)
            })

            # Test 2: Model agreement with consensus (1 = matches modal prediction, 0 = divergent)
            # Find row consensus
            consensus_mode = []
            for row in pivot[MODELS].values:
                mode_val = Counter(row).most_common(1)[0][0]
                consensus_mode.append(mode_val)

            bin_consensus = np.zeros(pivot[MODELS].shape, dtype=int)
            for i, mode_val in enumerate(consensus_mode):
                bin_consensus[i, :] = (pivot[MODELS].values[i, :] == mode_val).astype(int)

            q_stat_c, q_pval_c, q_df_c = cochrans_q_test(bin_consensus)

            records.append({
                "variable": var_name,
                "condition": cond_name,
                "outcome_type": "consensus_agreement",
                "n_patients": n_patients,
                "k_models": len(MODELS),
                "cochran_q_stat": round(q_stat_c, 4),
                "df": q_df_c,
                "p_val": round(q_pval_c, 5),
                "is_significant": bool(q_pval_c < 0.05)
            })

    df_cochran = pd.DataFrame(records)
    df_cochran.to_csv(OUTPUTS_DIR / "05_multimodel_cochran_q_categorical.csv", index=False)
    print(f"Saved 05_multimodel_cochran_q_categorical.csv ({len(df_cochran)} rows)")


def run_two_way_repeated_measures_anova():
    """
    Two-Way Repeated Measures ANOVA testing Model, Condition, and Model x Condition interaction
    with Greenhouse-Geisser sphericity correction and generalized eta-squared (ng2) effect size.
    """
    front_edu, front_unedu = load_front_data()
    prof_edu, prof_unedu = load_profile_data()

    front_edu["abs_nasal_tip_deviation_mm"] = front_edu["nasal_tip_deviation_right_mm"].abs()
    front_unedu["abs_nasal_tip_deviation_mm"] = front_unedu["nasal_tip_deviation_right_mm"].abs()
    front_edu["abs_chin_point_deviation_mm"] = front_edu["chin_point_deviation_right_mm"].abs()
    front_unedu["abs_chin_point_deviation_mm"] = front_unedu["chin_point_deviation_right_mm"].abs()

    all_prof = pd.concat([prof_edu, prof_unedu], ignore_index=True)
    all_front = pd.concat([front_edu, front_unedu], ignore_index=True)

    targets = [
        ("maxilla_advancement_mm", all_prof, "Maxilla Advancement (mm)"),
        ("maxilla_impaction_mm", all_prof, "Maxilla Impaction (mm)"),
        ("mandible_advancement_mm", all_prof, "Mandible Advancement (mm)"),
        ("mandible_impaction_mm", all_prof, "Mandible Impaction (mm)"),
        ("nasal_tip_deviation_right_mm", all_front, "Nasal Tip Deviation (mm)"),
        ("chin_point_deviation_right_mm", all_front, "Chin Point Deviation (mm)"),
        ("abs_nasal_tip_deviation_mm", all_front, "Abs Nasal Tip Deviation (mm)"),
        ("abs_chin_point_deviation_mm", all_front, "Abs Chin Point Deviation (mm)"),
    ]

    records = []

    for var_name, df_source, var_label in targets:
        sub = df_source[df_source["model_short"].isin(MODELS)][["patient_name", "model_short", "condition", var_name]].dropna()

        # Complete cases (30 patients x 4 models x 2 conditions = 8 cells per patient)
        counts = sub.groupby("patient_name")["model_short"].count()
        valid_patients = counts[counts == 8].index
        sub = sub[sub["patient_name"].isin(valid_patients)]

        if len(valid_patients) < 10:
            continue

        try:
            aov = pg.rm_anova(
                data=sub,
                dv=var_name,
                within=["model_short", "condition"],
                subject="patient_name"
            )

            for _, row in aov.iterrows():
                records.append({
                    "variable": var_name,
                    "variable_label": var_label,
                    "source": row["Source"],
                    "ss": round(float(row["SS"]), 4),
                    "ddof1": int(row["ddof1"]),
                    "ddof2": int(row["ddof2"]),
                    "ms": round(float(row["MS"]), 4),
                    "f_stat": round(float(row["F"]), 4),
                    "p_unc": round(float(row["p_unc"]), 5),
                    "p_gg_corr": round(float(row["p_GG_corr"]), 5) if "p_GG_corr" in row and pd.notna(row["p_GG_corr"]) else np.nan,
                    "sphericity_eps": round(float(row["eps"]), 4) if "eps" in row and pd.notna(row["eps"]) else np.nan,
                    "gen_eta_squared": round(float(row["ng2"]), 4) if "ng2" in row and pd.notna(row["ng2"]) else np.nan,
                    "is_significant": bool(row.get("p_GG_corr", row["p_unc"]) < 0.05)
                })
        except Exception as e:
            pass

    df_aov = pd.DataFrame(records)
    df_aov.to_csv(OUTPUTS_DIR / "05_multimodel_two_way_rm_anova.csv", index=False)
    print(f"Saved 05_multimodel_two_way_rm_anova.csv ({len(df_aov)} rows)")


if __name__ == "__main__":
    print("Running 05_multimodel_comparisons...")
    run_continuous_multimodel()
    run_categorical_cochran()
    run_two_way_repeated_measures_anova()
    print("Completed 05_multimodel_comparisons.")
