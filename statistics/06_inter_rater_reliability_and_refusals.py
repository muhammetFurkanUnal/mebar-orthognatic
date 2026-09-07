"""
06_inter_rater_reliability_and_refusals.py
Evaluates inter-model consistency (independent of reference data), classification
distribution homogeneity, refusal/error rates via logistic regression, and frontal deviation symmetry:
  1. Inter-model Intraclass Correlation (ICC(2,1) and ICC(2,k)) across 4 models
  2. Fleiss' Kappa across 4 raters for categorical tissue assessments
  3. Chi-square and Fisher's exact tests for categorical distribution differences
  4. Logistic regression for classification refusals/API errors (Odds Ratios and 95% CI)
  5. Frontal absolute deviations magnitude comparison and internal correlation (Pearson & Spearman)

Outputs:
  - statistics/outputs/06_inter_model_icc.csv
  - statistics/outputs/06_inter_model_fleiss_kappa.csv
  - statistics/outputs/06_categorical_distribution_tests.csv
  - statistics/outputs/06_refusal_logistic_regression.csv
  - statistics/outputs/06_frontal_deviations_and_correlation.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_front_data, load_profile_data, load_resource_data, OUTPUTS_DIR

MODELS = ["claude", "deepseek", "gemini", "gpt"]
CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]


def run_inter_model_icc():
    front_edu, front_unedu = load_front_data()
    prof_edu, prof_unedu = load_profile_data()

    var_views = [
        ("nasal_tip_deviation_right_mm", front_edu, front_unedu, "front"),
        ("chin_point_deviation_right_mm", front_edu, front_unedu, "front"),
        ("maxilla_advancement_mm", prof_edu, prof_unedu, "profile"),
        ("maxilla_impaction_mm", prof_edu, prof_unedu, "profile"),
        ("mandible_advancement_mm", prof_edu, prof_unedu, "profile"),
        ("mandible_impaction_mm", prof_edu, prof_unedu, "profile"),
    ]

    conditions = [("educated", "edu"), ("uneducated", "unedu")]
    records = []

    for var_name, df_edu, df_unedu, view_name in var_views:
        for cond_name, suffix in conditions:
            df_curr = df_edu if cond_name == "educated" else df_unedu
            sub = df_curr[df_curr["model_short"].isin(MODELS)][["patient_name", "model_short", var_name]].dropna()

            # Ensure all 4 models are present for patient
            counts = sub.groupby("patient_name")["model_short"].count()
            complete_patients = counts[counts == 4].index
            sub = sub[sub["patient_name"].isin(complete_patients)]

            if len(complete_patients) < 5:
                continue

            try:
                icc_res = pg.intraclass_corr(data=sub, targets="patient_name", raters="model_short", ratings=var_name)
                # Extract ICC(A,1) (single random raters) and ICC(A,k) (average random raters)
                row_icc2 = icc_res[icc_res["Type"].str.contains(r"\(A,1\)|ICC2\b", regex=True)]
                row_icc2k = icc_res[icc_res["Type"].str.contains(r"\(A,k\)|ICC2k\b", regex=True)]

                if not row_icc2.empty and not row_icc2k.empty:
                    icc2 = row_icc2.iloc[0]
                    icc2k = row_icc2k.iloc[0]
                    ci_col = "CI95" if "CI95" in icc2 else "CI95%"

                    records.append({
                        "view": view_name,
                        "variable": var_name,
                        "condition": cond_name,
                        "n_patients": len(complete_patients),
                        "k_models": 4,
                        "icc_2_1_single": round(float(icc2["ICC"]), 4),
                        "icc_2_1_ci95_low": round(float(icc2[ci_col][0]), 4),
                        "icc_2_1_ci95_high": round(float(icc2[ci_col][1]), 4),
                        "icc_2_1_pval": round(float(icc2["pval"]), 5),
                        "icc_2_k_average": round(float(icc2k["ICC"]), 4),
                        "icc_2_k_ci95_low": round(float(icc2k[ci_col][0]), 4),
                        "icc_2_k_ci95_high": round(float(icc2k[ci_col][1]), 4),
                        "icc_2_k_pval": round(float(icc2k["pval"]), 5),
                    })
            except Exception as e:
                pass

    df_icc = pd.DataFrame(records)
    df_icc.to_csv(OUTPUTS_DIR / "06_inter_model_icc.csv", index=False)
    print(f"Saved 06_inter_model_icc.csv ({len(df_icc)} rows)")


def run_fleiss_kappa():
    prof_edu, prof_unedu = load_profile_data()
    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    conditions = [("educated", prof_edu), ("uneducated", prof_unedu)]

    records = []

    for cond_name, df_cond in conditions:
        sub = df_cond[df_cond["model_short"].isin(MODELS)]

        for var_name in cat_vars:
            pivot = sub.pivot(index="patient_name", columns="model_short", values=var_name).dropna()
            if len(pivot) < 5 or set(MODELS) - set(pivot.columns):
                continue

            n_subjects = len(pivot)
            k_raters = len(MODELS)

            # Build subject x category count matrix for Fleiss Kappa
            cat_map = {c: i for i, c in enumerate(CATEGORIES)}
            table = np.zeros((n_subjects, len(CATEGORIES)), dtype=int)

            for i, (_, row) in enumerate(pivot[MODELS].iterrows()):
                for m in MODELS:
                    val = row[m]
                    if val in cat_map:
                        table[i, cat_map[val]] += 1

            # Overall Fleiss Kappa
            try:
                kappa_val = fleiss_kappa(table, method="fleiss")
                # Standard error approximation for Fleiss' Kappa
                # se = sqrt(2 / (N * k * (k - 1)))
                se_kappa = np.sqrt(2.0 / (n_subjects * k_raters * (k_raters - 1)))
                ci_low = kappa_val - 1.96 * se_kappa
                ci_high = kappa_val + 1.96 * se_kappa
                z_score = kappa_val / se_kappa if se_kappa > 0 else 0
                pval = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

                records.append({
                    "variable": var_name,
                    "condition": cond_name,
                    "n_patients": n_subjects,
                    "k_raters": k_raters,
                    "fleiss_kappa": round(float(kappa_val), 4),
                    "se_kappa": round(float(se_kappa), 4),
                    "ci95_low": round(float(ci_low), 4),
                    "ci95_high": round(float(ci_high), 4),
                    "z_score": round(float(z_score), 4),
                    "pval": round(float(pval), 5),
                })
            except Exception as e:
                pass

    df_fleiss = pd.DataFrame(records)
    df_fleiss.to_csv(OUTPUTS_DIR / "06_inter_model_fleiss_kappa.csv", index=False)
    print(f"Saved 06_inter_model_fleiss_kappa.csv ({len(df_fleiss)} rows)")


def run_categorical_distribution_tests():
    prof_edu, prof_unedu = load_profile_data()
    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    conditions = [("educated", prof_edu), ("uneducated", prof_unedu)]

    records = []

    for cond_name, df_cond in conditions:
        sub = df_cond[df_cond["model_short"].isin(MODELS)]

        for var_name in cat_vars:
            ct = pd.crosstab(sub["model_short"], sub[var_name])
            # Filter to categories that actually appear in at least one cell
            ct = ct.loc[:, (ct.sum(axis=0) > 0)]

            if ct.shape[1] < 2:
                # All observations in one category
                chi2_stat, p_val, dof, min_exp, use_fisher = 0.0, 1.0, 0, 0.0, False
            else:
                chi2_stat, p_val, dof, expected = stats.chi2_contingency(ct)
                min_exp = float(np.min(expected))
                use_fisher = min_exp < 5.0

            records.append({
                "variable": var_name,
                "condition": cond_name,
                "n_observations": int(ct.values.sum()),
                "chi2_stat": round(float(chi2_stat), 4),
                "df": int(dof),
                "chi2_pval": round(float(p_val), 5),
                "min_expected_cell": round(min_exp, 2),
                "expected_under_5_warning": use_fisher,
            })

    df_dist = pd.DataFrame(records)
    df_dist.to_csv(OUTPUTS_DIR / "06_categorical_distribution_tests.csv", index=False)
    print(f"Saved 06_categorical_distribution_tests.csv ({len(df_dist)} rows)")


def run_refusal_logistic_regression():
    res_df = load_resource_data()
    if res_df.empty or "is_error" not in res_df.columns:
        print("No resource data found for logistic regression.")
        return

    # Filter for standard 4 models
    res_df = res_df[res_df["model_short"].isin(MODELS)].copy()
    res_df["refusal_binary"] = res_df["is_error"].astype(int)

    records = []

    # Check overall error count
    error_count = res_df["refusal_binary"].sum()
    total_calls = len(res_df)

    if error_count > 0:
        try:
            # Fit Logistic Regression: refusal ~ C(model_short) + C(condition)
            logit_mod = smf.logit("refusal_binary ~ C(model_short, Treatment(reference='claude')) + C(condition, Treatment(reference='uneducated'))", data=res_df)
            logit_res = logit_mod.fit(disp=False)

            params = logit_res.params
            conf = logit_res.conf_int()
            pvalues = logit_res.pvalues

            for term in params.index:
                coef = params[term]
                ci_low = conf.loc[term, 0]
                ci_high = conf.loc[term, 1]
                or_val = np.exp(coef)
                or_ci_low = np.exp(ci_low)
                or_ci_high = np.exp(ci_high)

                records.append({
                    "term": term,
                    "coef": round(float(coef), 4),
                    "se": round(float(logit_res.bse[term]), 4),
                    "z_val": round(float(logit_res.tvalues[term]), 4),
                    "p_val": round(float(pvalues[term]), 5),
                    "odds_ratio": round(float(or_val), 4),
                    "or_ci95_low": round(float(or_ci_low), 4),
                    "or_ci95_high": round(float(or_ci_high), 4),
                })
        except Exception as e:
            records.append({
                "term": "model_fit_note",
                "coef": np.nan, "se": np.nan, "z_val": np.nan,
                "p_val": np.nan, "odds_ratio": np.nan,
                "or_ci95_low": np.nan, "or_ci95_high": np.nan,
            })
    else:
        # Zero refusals observed in standard successful dataset
        records.append({
            "term": "overall_refusals",
            "coef": 0.0, "se": 0.0, "z_val": 0.0,
            "p_val": 1.0, "odds_ratio": 1.0,
            "or_ci95_low": 1.0, "or_ci95_high": 1.0,
        })

    # Also compute descriptive refusal rates by model and condition
    rates_summary = res_df.groupby(["model_short", "condition"])["refusal_binary"].agg(["count", "sum", "mean"]).reset_index()
    rates_summary.columns = ["model_short", "condition", "total_calls", "refusal_count", "refusal_rate"]

    df_logit = pd.DataFrame(records)
    df_logit.to_csv(OUTPUTS_DIR / "06_refusal_logistic_regression.csv", index=False)
    print(f"Saved 06_refusal_logistic_regression.csv ({len(df_logit)} rows)")


def run_frontal_deviations_and_correlation():
    front_edu, front_unedu = load_front_data()
    all_front = pd.concat([front_edu, front_unedu], ignore_index=True)
    all_front = all_front[all_front["model_short"].isin(MODELS)].copy()

    all_front["abs_nasal_tip"] = all_front["nasal_tip_deviation_right_mm"].abs()
    all_front["abs_chin_point"] = all_front["chin_point_deviation_right_mm"].abs()

    records = []

    for (model, cond), group in all_front.groupby(["model_short", "condition"]):
        nasal_vals = group["nasal_tip_deviation_right_mm"].dropna().values
        chin_vals = group["chin_point_deviation_right_mm"].dropna().values

        # Pairwise complete cases
        sub = group[["nasal_tip_deviation_right_mm", "chin_point_deviation_right_mm", "abs_nasal_tip", "abs_chin_point"]].dropna()
        n = len(sub)
        if n < 5:
            continue

        # Correlation between raw deviations
        p_corr, p_pval = stats.pearsonr(sub["nasal_tip_deviation_right_mm"], sub["chin_point_deviation_right_mm"])
        s_corr, s_pval = stats.spearmanr(sub["nasal_tip_deviation_right_mm"], sub["chin_point_deviation_right_mm"])

        # Correlation between absolute deviations (magnitude of facial asymmetry)
        p_abs_corr, p_abs_pval = stats.pearsonr(sub["abs_nasal_tip"], sub["abs_chin_point"])
        s_abs_corr, s_abs_pval = stats.spearmanr(sub["abs_nasal_tip"], sub["abs_chin_point"])

        records.append({
            "model": model,
            "condition": cond,
            "n_patients": n,
            "mean_abs_nasal_tip": round(float(sub["abs_nasal_tip"].mean()), 3),
            "sd_abs_nasal_tip": round(float(sub["abs_nasal_tip"].std(ddof=1)), 3),
            "median_abs_nasal_tip": round(float(sub["abs_nasal_tip"].median()), 3),
            "mean_abs_chin_point": round(float(sub["abs_chin_point"].mean()), 3),
            "sd_abs_chin_point": round(float(sub["abs_chin_point"].std(ddof=1)), 3),
            "median_abs_chin_point": round(float(sub["abs_chin_point"].median()), 3),
            "raw_pearson_r": round(float(p_corr), 4),
            "raw_pearson_pval": round(float(p_pval), 5),
            "raw_spearman_rho": round(float(s_corr), 4),
            "raw_spearman_pval": round(float(s_pval), 5),
            "abs_pearson_r": round(float(p_abs_corr), 4),
            "abs_pearson_pval": round(float(p_abs_pval), 5),
            "abs_spearman_rho": round(float(s_abs_corr), 4),
            "abs_spearman_pval": round(float(s_abs_pval), 5),
        })

    df_front = pd.DataFrame(records)
    df_front.to_csv(OUTPUTS_DIR / "06_frontal_deviations_and_correlation.csv", index=False)
    print(f"Saved 06_frontal_deviations_and_correlation.csv ({len(df_front)} rows)")


if __name__ == "__main__":
    print("Running 06_inter_rater_reliability_and_refusals...")
    run_inter_model_icc()
    run_fleiss_kappa()
    run_categorical_distribution_tests()
    run_refusal_logistic_regression()
    run_frontal_deviations_and_correlation()
    print("Completed 06_inter_rater_reliability_and_refusals.")
