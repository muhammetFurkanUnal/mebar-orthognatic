"""
03_categorical_agreement_vs_reference.py
Evaluates diagnostic agreement for categorical soft tissue variables
(midface, upper_lip, lower_lip, chin_pogonion) using:
  - Weighted Cohen's Kappa (linear and quadratic for ordinal: hypoplastic < normognathic < hyperplastic)
  - Confusion Matrices (3x3)
  - Overall Accuracy (% exact match)
  - Diagnostic metrics per class: Sensitivity, Specificity, PPV (Precision), NPV, F1-score

Handles both direct comparison against reference ground truth (if populated)
and model-consensus benchmark (majority voting among models) when clinical reference labels are absent.

Outputs:
  - statistics/outputs/03_categorical_vs_reference_kappa.csv
  - statistics/outputs/03_categorical_confusion_matrices.csv
  - statistics/outputs/03_categorical_diagnostic_metrics.csv
"""

import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_profile_data, OUTPUTS_DIR

CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]
CAT_VARS = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
MODELS = ["claude", "deepseek", "gemini", "gpt"]


def compute_binary_metrics_for_class(cm, class_idx):
    """
    Given a 3x3 confusion matrix and target class index, compute:
    TP, FP, FN, TN, Sensitivity, Specificity, PPV, NPV, F1
    """
    tp = cm[class_idx, class_idx]
    fn = np.sum(cm[class_idx, :]) - tp
    fp = np.sum(cm[:, class_idx]) - tp
    tn = np.sum(cm) - (tp + fp + fn)

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": round(float(sens), 4) if not np.isnan(sens) else np.nan,
        "specificity": round(float(spec), 4) if not np.isnan(spec) else np.nan,
        "ppv": round(float(ppv), 4) if not np.isnan(ppv) else np.nan,
        "npv": round(float(npv), 4) if not np.isnan(npv) else np.nan,
        "f1": round(float(f1), 4) if not np.isnan(f1) else np.nan,
    }


def compute_categorical_agreement():
    prof_edu, prof_unedu = load_profile_data()

    kappa_records = []
    cm_records = []
    diag_records = []

    conditions = [("educated", prof_edu), ("uneducated", prof_unedu)]

    for cond_name, df_cond in conditions:
        # Check if reference has categorical data
        ref_df = df_cond[df_cond["model_short"] == "reference"]

        # Also create a consensus benchmark (majority vote) across the 4 models
        ai_df = df_cond[df_cond["model_short"].isin(MODELS)]

        consensus_dict = {}
        for p, p_group in ai_df.groupby("patient_name"):
            consensus_dict[p] = {}
            for v in CAT_VARS:
                vals = [x for x in p_group[v].dropna().tolist() if x in CATEGORIES]
                if vals:
                    majority = Counter(vals).most_common(1)[0][0]
                    consensus_dict[p][v] = majority
                else:
                    consensus_dict[p][v] = np.nan

        # Test both against Ground Truth (if available) and Consensus
        benchmark_sources = []
        has_ref_cats = ref_df[CAT_VARS].dropna(how="all").shape[0] > 0
        if has_ref_cats:
            ref_indexed = ref_df.set_index("patient_name")
            benchmark_sources.append(("ground_truth_reference", ref_indexed))

        benchmark_sources.append(("model_consensus", pd.DataFrame.from_dict(consensus_dict, orient="index")))

        for bench_type, bench_df in benchmark_sources:
            for model in MODELS:
                m_df = df_cond[df_cond["model_short"] == model].set_index("patient_name")

                for var_name in CAT_VARS:
                    y_true = []
                    y_pred = []

                    for p in bench_df.index:
                        if p in m_df.index:
                            t_val = bench_df.loc[p, var_name] if var_name in bench_df.columns else np.nan
                            p_val = m_df.loc[p, var_name] if var_name in m_df.columns else np.nan

                            if pd.notna(t_val) and pd.notna(p_val) and t_val in CATEGORIES and p_val in CATEGORIES:
                                y_true.append(t_val)
                                y_pred.append(p_val)

                    n_pairs = len(y_true)
                    if n_pairs < 5:
                        continue

                    # Unweighted, Linear weighted, and Quadratic weighted Kappa
                    unweighted_k = cohen_kappa_score(y_true, y_pred, labels=CATEGORIES)
                    linear_k = cohen_kappa_score(y_true, y_pred, labels=CATEGORIES, weights="linear")
                    quad_k = cohen_kappa_score(y_true, y_pred, labels=CATEGORIES, weights="quadratic")

                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
                    exact_match_pct = (np.trace(cm) / n_pairs) * 100

                    kappa_records.append({
                        "benchmark_type": bench_type,
                        "condition": cond_name,
                        "model": model,
                        "variable": var_name,
                        "n": n_pairs,
                        "exact_agreement_pct": round(exact_match_pct, 2),
                        "unweighted_kappa": round(float(unweighted_k), 4) if not np.isnan(unweighted_k) else np.nan,
                        "linear_weighted_kappa": round(float(linear_k), 4) if not np.isnan(linear_k) else np.nan,
                        "quadratic_weighted_kappa": round(float(quad_k), 4) if not np.isnan(quad_k) else np.nan,
                    })

                    # Record Confusion Matrix cells
                    for i, true_label in enumerate(CATEGORIES):
                        for j, pred_label in enumerate(CATEGORIES):
                            cm_records.append({
                                "benchmark_type": bench_type,
                                "condition": cond_name,
                                "model": model,
                                "variable": var_name,
                                "true_category": true_label,
                                "pred_category": pred_label,
                                "count": int(cm[i, j])
                            })

                    # Diagnostic metrics per category
                    for idx, cat_label in enumerate(CATEGORIES):
                        bin_metrics = compute_binary_metrics_for_class(cm, idx)
                        diag_records.append({
                            "benchmark_type": bench_type,
                            "condition": cond_name,
                            "model": model,
                            "variable": var_name,
                            "category": cat_label,
                            **bin_metrics
                        })

    df_kappa = pd.DataFrame(kappa_records)
    df_cm = pd.DataFrame(cm_records)
    df_diag = pd.DataFrame(diag_records)

    df_kappa.to_csv(OUTPUTS_DIR / "03_categorical_vs_reference_kappa.csv", index=False)
    df_cm.to_csv(OUTPUTS_DIR / "03_categorical_confusion_matrices.csv", index=False)
    df_diag.to_csv(OUTPUTS_DIR / "03_categorical_diagnostic_metrics.csv", index=False)

    print(f"Saved 03_categorical_vs_reference_kappa.csv ({len(df_kappa)} rows)")
    print(f"Saved 03_categorical_confusion_matrices.csv ({len(df_cm)} rows)")
    print(f"Saved 03_categorical_diagnostic_metrics.csv ({len(df_diag)} rows)")


if __name__ == "__main__":
    print("Running 03_categorical_agreement_vs_reference...")
    compute_categorical_agreement()
    print("Completed 03_categorical_agreement_vs_reference.")
