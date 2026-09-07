"""
04_condition_paired_comparisons.py
Evaluates the paired effect of the prompting condition (Educated vs Uneducated)
within the same model and patient:
  - Continuous variables:
      * Normality test on paired differences (Shapiro-Wilk)
      * Paired t-test (t-stat, p, mean diff, 95% CI, Cohen's dz)
      * Wilcoxon signed-rank test (W-stat, p, median diff, rank-biserial r)
  - Categorical variables:
      * Stuart-Maxwell test of marginal homogeneity for 3x3 tables
      * McNemar test with continuity correction on collapsed binary table (Normognathic vs Dysplastic)
      * Shift dynamics (unchanged, increased, decreased)

Outputs:
  - statistics/outputs/04_condition_continuous_paired_tests.csv
  - statistics/outputs/04_condition_categorical_paired_tests.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_front_data, load_profile_data, OUTPUTS_DIR

MODELS = ["claude", "deepseek", "gemini", "gpt"]
CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]


def stuart_maxwell_test(table_3x3):
    """
    Stuart-Maxwell test for marginal homogeneity in a 3x3 paired table.
    table_3x3: numpy array of shape (3, 3) where rows = Uneducated, cols = Educated.
    Returns: chi2_stat, p_value, df
    """
    n = np.asarray(table_3x3, dtype=float)
    if n.shape != (3, 3):
        return np.nan, np.nan, np.nan

    # Marginal sums
    row_sums = n.sum(axis=1)
    col_sums = n.sum(axis=0)

    # Differences between marginals
    d = row_sums - col_sums

    # Covariance matrix V
    v = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                v[i, i] = row_sums[i] + col_sums[i] - 2 * n[i, i]
            else:
                v[i, j] = -(n[i, j] + n[j, i])

    # Reduce to 2x2 by dropping the last category
    d_sub = d[:2]
    v_sub = v[:2, :2]

    # Check if there are differences
    if np.allclose(d_sub, 0):
        return 0.0, 1.0, 2

    try:
        # Invert submatrix
        v_inv = np.linalg.inv(v_sub)
        chi2_stat = float(d_sub.T @ v_inv @ d_sub)
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=2))
        return chi2_stat, p_val, 2
    except np.linalg.LinAlgError:
        # Singular matrix, fallback to pseudo-inverse
        try:
            v_inv = np.linalg.pinv(v_sub)
            chi2_stat = float(d_sub.T @ v_inv @ d_sub)
            p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=2))
            return chi2_stat, p_val, 2
        except Exception:
            return np.nan, np.nan, np.nan


def mcnemar_binary_test(b, c):
    """
    McNemar's test with continuity correction for discordant pairs b and c.
    chi2 = (|b - c| - 1)^2 / (b + c)
    """
    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0
    chi2 = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p_val)


def run_continuous_paired_tests():
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

    records = []

    for var_name, df_edu, df_unedu, view_name in var_views:
        for model in MODELS:
            sub_edu = df_edu[df_edu["model_short"] == model].set_index("patient_name")
            sub_unedu = df_unedu[df_unedu["model_short"] == model].set_index("patient_name")

            common_pts = [p for p in sub_edu.index if p in sub_unedu.index]

            pairs = []
            for p in common_pts:
                v_edu = sub_edu.loc[p, var_name]
                v_unedu = sub_unedu.loc[p, var_name]
                if pd.notna(v_edu) and pd.notna(v_unedu):
                    pairs.append((float(v_edu), float(v_unedu)))

            if len(pairs) < 3:
                continue

            arr_edu = np.array([p[0] for p in pairs])
            arr_unedu = np.array([p[1] for p in pairs])
            diffs = arr_edu - arr_unedu
            n = len(diffs)

            mean_diff = float(np.mean(diffs))
            sd_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
            median_diff = float(np.median(diffs))
            iqr_diff = float(np.percentile(diffs, 75) - np.percentile(diffs, 25))

            # Normality of differences
            if np.ptp(diffs) > 1e-8:
                sw_stat, sw_pval = stats.shapiro(diffs)
                diff_is_normal = bool(sw_pval >= 0.05)
            else:
                sw_stat, sw_pval, diff_is_normal = np.nan, np.nan, False

            # Paired t-test
            t_res = stats.ttest_rel(arr_edu, arr_unedu)
            t_stat = float(t_res.statistic)
            t_pval = float(t_res.pvalue)

            # 95% CI of mean difference
            t_crit = stats.t.ppf(0.975, df=n - 1)
            se_diff = sd_diff / np.sqrt(n)
            ci_low = mean_diff - t_crit * se_diff
            ci_high = mean_diff + t_crit * se_diff

            # Cohen's dz
            cohens_dz = (mean_diff / sd_diff) if sd_diff > 1e-8 else 0.0

            # Wilcoxon signed-rank test
            if np.all(diffs == 0):
                w_stat, w_pval = 0.0, 1.0
                rank_biserial = 0.0
            else:
                try:
                    w_res = stats.wilcoxon(arr_edu, arr_unedu, alternative="two-sided")
                    w_stat = float(w_res.statistic)
                    w_pval = float(w_res.pvalue)
                    # Non-zero diff count
                    n_nonzero = np.count_nonzero(diffs)
                    max_w = n_nonzero * (n_nonzero + 1) / 2
                    rank_biserial = float(1.0 - (2.0 * w_stat / max_w)) if max_w > 0 else 0.0
                except Exception:
                    w_stat, w_pval, rank_biserial = np.nan, np.nan, np.nan

            recommended_test = "paired_t" if diff_is_normal else "wilcoxon"
            primary_p = t_pval if diff_is_normal else w_pval

            records.append({
                "view": view_name,
                "variable": var_name,
                "model": model,
                "n": n,
                "mean_edu": round(float(np.mean(arr_edu)), 3),
                "mean_unedu": round(float(np.mean(arr_unedu)), 3),
                "mean_difference": round(mean_diff, 3),
                "sd_difference": round(sd_diff, 3),
                "ci95_diff_low": round(ci_low, 3),
                "ci95_diff_high": round(ci_high, 3),
                "median_difference": round(median_diff, 3),
                "iqr_difference": round(iqr_diff, 3),
                "shapiro_p_diff": round(float(sw_pval), 5) if not np.isnan(sw_pval) else np.nan,
                "diff_is_normal": diff_is_normal,
                "paired_t_stat": round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                "paired_t_pval": round(t_pval, 5) if not np.isnan(t_pval) else np.nan,
                "cohens_dz": round(cohens_dz, 4),
                "wilcoxon_w": round(w_stat, 4) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_pval": round(w_pval, 5) if not np.isnan(w_pval) else np.nan,
                "rank_biserial_r": round(rank_biserial, 4) if not np.isnan(rank_biserial) else np.nan,
                "recommended_test": recommended_test,
                "primary_pval": round(primary_p, 5) if not np.isnan(primary_p) else np.nan,
            })

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUTPUTS_DIR / "04_condition_continuous_paired_tests.csv", index=False)
    print(f"Saved 04_condition_continuous_paired_tests.csv ({len(df_res)} rows)")


def run_categorical_paired_tests():
    prof_edu, prof_unedu = load_profile_data()
    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]

    records = []

    for var_name in cat_vars:
        for model in MODELS:
            sub_edu = prof_edu[prof_edu["model_short"] == model].set_index("patient_name")
            sub_unedu = prof_unedu[prof_unedu["model_short"] == model].set_index("patient_name")

            common_pts = [p for p in sub_edu.index if p in sub_unedu.index]

            pairs = []
            for p in common_pts:
                v_edu = sub_edu.loc[p, var_name]
                v_unedu = sub_unedu.loc[p, var_name]
                if pd.notna(v_edu) and pd.notna(v_unedu) and v_edu in CATEGORIES and v_unedu in CATEGORIES:
                    pairs.append((v_unedu, v_edu))  # row = unedu, col = edu

            n_pairs = len(pairs)
            if n_pairs < 5:
                continue

            # Construct 3x3 table
            t3x3 = np.zeros((3, 3), dtype=int)
            cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
            for u, e in pairs:
                t3x3[cat_idx[u], cat_idx[e]] += 1

            # Stuart-Maxwell Test
            sm_stat, sm_pval, sm_df = stuart_maxwell_test(t3x3)

            # McNemar Test on collapsed Normognathic vs Dysplastic (Hypo + Hyper)
            # Binary: 0 = Normognathic, 1 = Dysplastic
            # b: Unedu=Normal, Edu=Dysplastic
            # c: Unedu=Dysplastic, Edu=Normal
            b = sum(1 for u, e in pairs if u == "normognathic" and e != "normognathic")
            c = sum(1 for u, e in pairs if u != "normognathic" and e == "normognathic")
            mcn_stat, mcn_pval = mcnemar_binary_test(b, c)

            # Shift dynamics:
            # - same: u == e
            # - increased ordinal value: idx(e) > idx(u)
            # - decreased ordinal value: idx(e) < idx(u)
            unchanged = sum(1 for u, e in pairs if cat_idx[u] == cat_idx[e])
            increased = sum(1 for u, e in pairs if cat_idx[e] > cat_idx[u])
            decreased = sum(1 for u, e in pairs if cat_idx[e] < cat_idx[u])

            records.append({
                "variable": var_name,
                "model": model,
                "n_pairs": n_pairs,
                "unchanged_n": unchanged,
                "unchanged_pct": round(unchanged / n_pairs * 100, 2),
                "increased_severity_n": increased,
                "decreased_severity_n": decreased,
                "stuart_maxwell_chi2": round(sm_stat, 4) if not np.isnan(sm_stat) else np.nan,
                "stuart_maxwell_df": sm_df if not np.isnan(sm_df) else np.nan,
                "stuart_maxwell_pval": round(sm_pval, 5) if not np.isnan(sm_pval) else np.nan,
                "mcnemar_discordant_b": b,
                "mcnemar_discordant_c": c,
                "mcnemar_chi2": round(mcn_stat, 4) if not np.isnan(mcn_stat) else np.nan,
                "mcnemar_pval": round(mcn_pval, 5) if not np.isnan(mcn_pval) else np.nan,
            })

    df_cat = pd.DataFrame(records)
    df_cat.to_csv(OUTPUTS_DIR / "04_condition_categorical_paired_tests.csv", index=False)
    print(f"Saved 04_condition_categorical_paired_tests.csv ({len(df_cat)} rows)")


if __name__ == "__main__":
    print("Running 04_condition_paired_comparisons...")
    run_continuous_paired_tests()
    run_categorical_paired_tests()
    print("Completed 04_condition_paired_comparisons.")
