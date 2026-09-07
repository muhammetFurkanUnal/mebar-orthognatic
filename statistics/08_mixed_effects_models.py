"""
08_mixed_effects_models.py
Unified Mixed-Effects Modeling Framework:
  1. Linear Mixed-Effects Model (LMM):
     - Models continuous measurements and absolute error against Arnett reference
     - Fixed effects: Model + Condition + Model * Condition
     - Random effect: Random intercept for Patient ID (1 | Patient)
     - Produces parameter estimates, 95% CIs, and Type III Wald test ANOVA table
  2. Cumulative Link Mixed Model (CLMM / Ordinal Logistic):
     - Models ordinal soft tissue classification (hypoplastic < normognathic < hyperplastic)
     - Fixed effects: Model + Condition
     - Clustered covariance by Patient ID
     - Produces log-odds, Odds Ratios (OR) and 95% CIs
  3. Generalized Linear Mixed Model (GLMM / Binary GEE):
     - Models binary clinical acceptability (|Error| <= 2.0 mm)
     - Binomial family with Logit link, clustered by Patient ID

Outputs:
  - statistics/outputs/08_lmm_continuous_results.csv
  - statistics/outputs/08_lmm_anova_table.csv
  - statistics/outputs/08_clmm_ordinal_results.csv
  - statistics/outputs/08_glmm_binary_results.csv
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_profile_data, load_front_data, OUTPUTS_DIR

MODELS = ["claude", "deepseek", "gemini", "gpt"]
CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]


def prepare_long_analysis_dataframe():
    """Merge profile and front data into a single long dataframe with reference errors."""
    prof_edu, prof_unedu = load_profile_data()
    front_edu, front_unedu = load_front_data()

    # Separate reference values from profile data
    prof_all = pd.concat([prof_edu, prof_unedu], ignore_index=True)
    ref_prof = prof_all[prof_all["model_short"] == "reference"].copy()
    ref_dict = ref_prof.groupby("patient_name").first()

    # Filter for the 4 AI models
    ai_prof = prof_all[prof_all["model_short"].isin(MODELS)].copy()

    # Compute absolute error vs Arnett reference
    movements = ["maxilla_advancement_mm", "maxilla_impaction_mm", "mandible_advancement_mm", "mandible_impaction_mm"]
    for m in movements:
        err_col = f"abs_err_{m}"
        ai_prof[err_col] = np.nan
        for idx, row in ai_prof.iterrows():
            p_name = row["patient_name"]
            if p_name in ref_dict.index:
                ref_val = ref_dict.loc[p_name, m]
                ai_val = row[m]
                if pd.notna(ref_val) and pd.notna(ai_val):
                    ai_prof.loc[idx, err_col] = abs(float(ai_val) - float(ref_val))

    # Merge front data
    front_all = pd.concat([front_edu, front_unedu], ignore_index=True)
    ai_front = front_all[front_all["model_short"].isin(MODELS)].copy()

    merged = pd.merge(
        ai_prof,
        ai_front[["patient_name", "model_short", "condition", "nasal_tip_deviation_right_mm", "chin_point_deviation_right_mm"]],
        on=["patient_name", "model_short", "condition"],
        how="left"
    )

    merged["abs_nasal_tip"] = merged["nasal_tip_deviation_right_mm"].abs()
    merged["abs_chin_point"] = merged["chin_point_deviation_right_mm"].abs()

    # Encode categorical variables as ordinal integers (1, 2, 3)
    cat_map = {"hypoplastic": 1, "normognathic": 2, "hyperplastic": 3}
    for c in ["midface", "upper_lip", "lower_lip", "chin_pogonion"]:
        merged[f"{c}_ord"] = merged[c].map(cat_map)

    return merged


def run_lmm_models(df):
    """Fit Linear Mixed Models on continuous outcomes and errors."""
    outcomes = [
        ("abs_err_maxilla_advancement_mm", "Maxilla Advancement Absolute Error"),
        ("abs_err_mandible_advancement_mm", "Mandible Advancement Absolute Error"),
        ("abs_err_maxilla_impaction_mm", "Maxilla Impaction Absolute Error"),
        ("abs_err_mandible_impaction_mm", "Mandible Impaction Absolute Error"),
        ("maxilla_advancement_mm", "Maxilla Advancement (mm)"),
        ("mandible_advancement_mm", "Mandible Advancement (mm)"),
        ("abs_nasal_tip", "Frontal Nasal Tip Absolute Deviation (mm)"),
        ("abs_chin_point", "Frontal Chin Point Absolute Deviation (mm)"),
    ]

    param_records = []
    anova_records = []

    for outcome_col, outcome_label in outcomes:
        sub = df[["patient_name", "model_short", "condition", outcome_col]].dropna().copy()
        if len(sub) < 20:
            continue

        formula = f"{outcome_col} ~ C(model_short, Treatment(reference='claude')) * C(condition, Treatment(reference='uneducated'))"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod = smf.mixedlm(formula, data=sub, groups=sub["patient_name"])
                res = mod.fit(reml=True)

                # Parameter estimates
                params = res.params
                bse = res.bse
                pvalues = res.pvalues
                conf = res.conf_int()

                for term in params.index:
                    param_records.append({
                        "outcome": outcome_col,
                        "outcome_label": outcome_label,
                        "term": term,
                        "coef": round(float(params[term]), 4),
                        "se": round(float(bse[term]), 4) if term in bse and pd.notna(bse[term]) else np.nan,
                        "z_val": round(float(res.tvalues[term]), 4) if term in res.tvalues and pd.notna(res.tvalues[term]) else np.nan,
                        "p_val": round(float(pvalues[term]), 5) if term in pvalues and pd.notna(pvalues[term]) else np.nan,
                        "ci95_low": round(float(conf.loc[term, 0]), 4) if term in conf.index else np.nan,
                        "ci95_high": round(float(conf.loc[term, 1]), 4) if term in conf.index else np.nan,
                    })

                # Hypothesis tests for Model main effect, Condition main effect, and Interaction
                # Model effect (3 df)
                model_terms = [t for t in params.index if "C(model_short" in t and ":" not in t]
                if model_terms:
                    try:
                        r_matrix = np.zeros((len(model_terms), len(params)))
                        for i, term in enumerate(model_terms):
                            idx = list(params.index).index(term)
                            r_matrix[i, idx] = 1.0
                        wt = res.wald_test(r_matrix)
                        anova_records.append({
                            "outcome": outcome_col,
                            "outcome_label": outcome_label,
                            "effect": "Model Main Effect",
                            "chi2_stat": round(float(wt.statistic), 4),
                            "df": len(model_terms),
                            "p_val": round(float(wt.pvalue), 5),
                            "is_significant": bool(wt.pvalue < 0.05),
                        })
                    except Exception:
                        pass

                # Condition effect (1 df)
                cond_terms = [t for t in params.index if "C(condition" in t and ":" not in t]
                if cond_terms:
                    try:
                        r_matrix = np.zeros((len(cond_terms), len(params)))
                        for i, term in enumerate(cond_terms):
                            idx = list(params.index).index(term)
                            r_matrix[i, idx] = 1.0
                        wt = res.wald_test(r_matrix)
                        anova_records.append({
                            "outcome": outcome_col,
                            "outcome_label": outcome_label,
                            "effect": "Condition Main Effect",
                            "chi2_stat": round(float(wt.statistic), 4),
                            "df": len(cond_terms),
                            "p_val": round(float(wt.pvalue), 5),
                            "is_significant": bool(wt.pvalue < 0.05),
                        })
                    except Exception:
                        pass

                # Interaction effect (3 df)
                inter_terms = [t for t in params.index if ":" in t]
                if inter_terms:
                    try:
                        r_matrix = np.zeros((len(inter_terms), len(params)))
                        for i, term in enumerate(inter_terms):
                            idx = list(params.index).index(term)
                            r_matrix[i, idx] = 1.0
                        wt = res.wald_test(r_matrix)
                        anova_records.append({
                            "outcome": outcome_col,
                            "outcome_label": outcome_label,
                            "effect": "Model x Condition Interaction",
                            "chi2_stat": round(float(wt.statistic), 4),
                            "df": len(inter_terms),
                            "p_val": round(float(wt.pvalue), 5),
                            "is_significant": bool(wt.pvalue < 0.05),
                        })
                    except Exception:
                        pass

            except Exception as e:
                pass

    df_params = pd.DataFrame(param_records)
    df_anova = pd.DataFrame(anova_records)

    df_params.to_csv(OUTPUTS_DIR / "08_lmm_continuous_results.csv", index=False)
    df_anova.to_csv(OUTPUTS_DIR / "08_lmm_anova_table.csv", index=False)
    print(f"Saved 08_lmm_continuous_results.csv ({len(df_params)} rows)")
    print(f"Saved 08_lmm_anova_table.csv ({len(df_anova)} rows)")


def run_clmm_models(df):
    """Fit Ordinal Logistic Models with clustered standard errors by patient."""
    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    records = []

    for var in cat_vars:
        col = f"{var}_ord"
        sub = df[["patient_name", "model_short", "condition", col]].dropna().copy()
        # Ensure at least 2 distinct levels present
        if sub[col].nunique() < 2 or len(sub) < 20:
            continue

        sub[col] = sub[col].astype(int)
        formula = f"{col} ~ C(model_short, Treatment(reference='claude')) + C(condition, Treatment(reference='uneducated'))"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod = OrderedModel.from_formula(formula, data=sub, distr="logit")
                res = mod.fit(disp=False, cov_type="cluster", cov_kwds={"groups": sub["patient_name"]})

                params = res.params
                bse = res.bse
                pvals = res.pvalues
                conf = res.conf_int()

                for term in params.index:
                    coef = float(params[term])
                    se = float(bse[term]) if term in bse else np.nan
                    z_val = float(res.tvalues[term]) if term in res.tvalues else np.nan
                    pval = float(pvals[term]) if term in pvals else np.nan
                    ci_low = float(conf.loc[term, 0]) if term in conf.index else np.nan
                    ci_high = float(conf.loc[term, 1]) if term in conf.index else np.nan
                    # Compute Odds Ratio for slope coefficients
                    or_val = np.exp(coef) if "/" not in term else np.nan
                    or_low = np.exp(ci_low) if "/" not in term and pd.notna(ci_low) else np.nan
                    or_high = np.exp(ci_high) if "/" not in term and pd.notna(ci_high) else np.nan

                    records.append({
                        "variable": var,
                        "term": term,
                        "coef": round(coef, 4),
                        "se": round(se, 4) if not np.isnan(se) else np.nan,
                        "z_val": round(z_val, 4) if not np.isnan(z_val) else np.nan,
                        "p_val": round(pval, 5) if not np.isnan(pval) else np.nan,
                        "odds_ratio": round(or_val, 4) if not np.isnan(or_val) else np.nan,
                        "or_ci95_low": round(or_low, 4) if not np.isnan(or_low) else np.nan,
                        "or_ci95_high": round(or_high, 4) if not np.isnan(or_high) else np.nan,
                    })
            except Exception as e:
                pass

    df_clmm = pd.DataFrame(records)
    df_clmm.to_csv(OUTPUTS_DIR / "08_clmm_ordinal_results.csv", index=False)
    print(f"Saved 08_clmm_ordinal_results.csv ({len(df_clmm)} rows)")


def run_glmm_binary_models(df):
    """Fit Binary GLMM / GEE models for clinical error acceptability (<= 2.0 mm)."""
    error_vars = [
        ("abs_err_maxilla_advancement_mm", "Maxilla Advancement Clinical Agreement (<=2mm)"),
        ("abs_err_mandible_advancement_mm", "Mandible Advancement Clinical Agreement (<=2mm)"),
    ]

    records = []

    for err_col, label in error_vars:
        sub = df[["patient_name", "model_short", "condition", err_col]].dropna().copy()
        if len(sub) < 20:
            continue

        # Clinical success defined as within 2.0 mm of reference plan
        bin_col = f"{err_col}_success"
        sub[bin_col] = (sub[err_col] <= 2.0).astype(int)

        # Check variation in outcome
        if sub[bin_col].nunique() < 2:
            continue

        formula = f"{bin_col} ~ C(model_short, Treatment(reference='claude')) + C(condition, Treatment(reference='uneducated'))"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod = smf.gee(formula, groups="patient_name", data=sub, family=sm.families.Binomial())
                res = mod.fit()

                params = res.params
                bse = res.bse
                pvals = res.pvalues
                conf = res.conf_int()

                for term in params.index:
                    coef = float(params[term])
                    se = float(bse[term]) if term in bse else np.nan
                    z_val = float(res.tvalues[term]) if term in res.tvalues else np.nan
                    pval = float(pvals[term]) if term in pvals else np.nan
                    ci_low = float(conf.loc[term, 0]) if term in conf.index else np.nan
                    ci_high = float(conf.loc[term, 1]) if term in conf.index else np.nan
                    or_val = np.exp(coef)
                    or_low = np.exp(ci_low) if pd.notna(ci_low) else np.nan
                    or_high = np.exp(ci_high) if pd.notna(ci_high) else np.nan

                    records.append({
                        "outcome": label,
                        "term": term,
                        "coef": round(coef, 4),
                        "se": round(se, 4) if not np.isnan(se) else np.nan,
                        "z_val": round(z_val, 4) if not np.isnan(z_val) else np.nan,
                        "p_val": round(pval, 5) if not np.isnan(pval) else np.nan,
                        "odds_ratio": round(or_val, 4),
                        "or_ci95_low": round(or_low, 4) if not np.isnan(or_low) else np.nan,
                        "or_ci95_high": round(or_high, 4) if not np.isnan(or_high) else np.nan,
                    })
            except Exception as e:
                pass

    df_glmm = pd.DataFrame(records)
    df_glmm.to_csv(OUTPUTS_DIR / "08_glmm_binary_results.csv", index=False)
    print(f"Saved 08_glmm_binary_results.csv ({len(df_glmm)} rows)")


if __name__ == "__main__":
    print("Running 08_mixed_effects_models...")
    unified_df = prepare_long_analysis_dataframe()
    run_lmm_models(unified_df)
    run_clmm_models(unified_df)
    run_glmm_binary_models(unified_df)
    print("Completed 08_mixed_effects_models.")
