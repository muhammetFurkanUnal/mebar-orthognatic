"""
09_generate_plots.py
Generates publication-quality figures (300 DPI) for the orthognathic AI analysis:
  1. fig1_normality_qq_histograms.png: Distribution histograms & Q-Q plots
  2. fig2_bland_altman_grid.png: Bland-Altman analysis panels (AI vs Arnett Reference)
  3. fig3_confusion_matrices_heatmap.png: Diagnostic confusion matrix heatmaps
  4. fig4_condition_paired_comparison.png: Educated vs Uneducated paired shift plots
  5. fig5_model_accuracy_mae_rmse.png: MAE and RMSE comparison across models & conditions
  6. fig6_resource_tradeoffs.png: Token usage, cost (USD), and response time analysis
  7. fig7_frontal_asymmetry_correlation.png: Frontal nasal tip vs chin point deviation scatter

Outputs saved to: statistics/outputs/plots/
"""

import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_profile_data, load_front_data, load_resource_data, OUTPUTS_DIR

PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})
PALETTE = {"claude": "#D95F02", "deepseek": "#1F78B4", "gemini": "#33A02C", "gpt": "#7570B3", "reference": "#000000"}
MODELS = ["claude", "deepseek", "gemini", "gpt"]
CATEGORIES = ["hypoplastic", "normognathic", "hyperplastic"]


def plot_fig1_normality_and_histograms():
    """Figure 1: Histograms and Q-Q plots for key continuous variables."""
    prof_edu, _ = load_profile_data()
    front_edu, _ = load_front_data()

    variables = [
        ("maxilla_advancement_mm", prof_edu, "Maxilla Advancement (mm, Profile)"),
        ("mandible_advancement_mm", prof_edu, "Mandible Advancement (mm, Profile)"),
        ("nasal_tip_deviation_right_mm", front_edu, "Nasal Tip Deviation (mm, Frontal)"),
        ("chin_point_deviation_right_mm", front_edu, "Chin Point Deviation (mm, Frontal)"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)
    fig.suptitle("Figure 1: Distribution Histograms & Normal Q-Q Plots (Educated Condition)", fontsize=14, fontweight="bold")

    for i, (var_name, df_source, title_text) in enumerate(variables):
        # AI values
        sub = df_source[df_source["model_short"].isin(MODELS)][var_name].dropna().values
        # Histogram + KDE
        ax_hist = axes[i, 0]
        sns.histplot(sub, kde=True, ax=ax_hist, color="#2b5c8f", bins=15, edgecolor="white")
        w_stat, p_val = stats.shapiro(sub)
        ax_hist.set_title(f"{title_text}\n(Shapiro-Wilk W={w_stat:.3f}, p={p_val:.4f})", fontsize=10)
        ax_hist.set_xlabel("Measurement (mm)", fontsize=9)
        ax_hist.set_ylabel("Frequency", fontsize=9)
        ax_hist.grid(True)

        # Q-Q Plot
        ax_qq = axes[i, 1]
        (osm, osr), (slope, intercept, r) = stats.probplot(sub, dist="norm", fit=True)
        ax_qq.scatter(osm, osr, color="#e6550d", alpha=0.7, s=24)
        ax_qq.plot(osm, slope * np.array(osm) + intercept, color="#333333", linestyle="--", linewidth=1.5)
        ax_qq.set_title(f"Q-Q Plot: {var_name} (r={r:.3f})", fontsize=10)
        ax_qq.set_xlabel("Theoretical Quantiles", fontsize=9)
        ax_qq.set_ylabel("Sample Quantiles", fontsize=9)
        ax_qq.grid(True)

    out_path = PLOTS_DIR / "fig1_normality_qq_histograms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig2_bland_altman_grid():
    """Figure 2: 4x2 Bland-Altman panels comparing each AI model vs Arnett Reference for Jaw Movements."""
    prof_edu, _ = load_profile_data()
    ref_df = prof_edu[prof_edu["model_short"] == "reference"].set_index("patient_name")

    fig, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    fig.suptitle("Figure 2: Bland–Altman Plots (AI Models vs. Arnett Reference Plan - Educated)", fontsize=14, fontweight="bold")

    movements = [("maxilla_advancement_mm", "Maxillary Advancement (mm)", 0),
                 ("mandible_advancement_mm", "Mandibular Advancement (mm)", 1)]

    for row_idx, model in enumerate(MODELS):
        m_df = prof_edu[prof_edu["model_short"] == model].set_index("patient_name")
        common_pts = [p for p in ref_df.index if p in m_df.index]

        for var_col, var_title, col_idx in movements:
            ax = axes[row_idx, col_idx]

            pairs = []
            for p in common_pts:
                r_v = ref_df.loc[p, var_col]
                m_v = m_df.loc[p, var_col]
                if pd.notna(r_v) and pd.notna(m_v):
                    pairs.append((float(m_v), float(r_v)))

            if len(pairs) < 3:
                continue

            ai_vals = np.array([p[0] for p in pairs])
            ref_vals = np.array([p[1] for p in pairs])

            means = (ai_vals + ref_vals) / 2.0
            diffs = ai_vals - ref_vals  # AI - Reference

            bias = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1))
            upper_loa = bias + 1.96 * sd
            lower_loa = bias - 1.96 * sd

            # Scatter points
            ax.scatter(means, diffs, color=PALETTE[model], alpha=0.75, s=36, edgecolor="black", linewidth=0.5)

            # Horizontal lines
            ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
            ax.axhline(bias, color="red", linestyle="-", linewidth=1.5, label=f"Bias: {bias:.2f}")
            ax.axhline(upper_loa, color="blue", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {upper_loa:.2f}")
            ax.axhline(lower_loa, color="blue", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {lower_loa:.2f}")

            # Shaded LoA area
            ax.fill_between([means.min() - 2, means.max() + 2], lower_loa, upper_loa, color="#e6f2ff", alpha=0.3)

            ax.set_title(f"{model.upper()} | {var_title}", fontsize=11, fontweight="semibold")
            ax.set_xlabel("Mean of AI and Reference (mm)", fontsize=9)
            ax.set_ylabel("Difference (AI - Ref) [mm]", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
            ax.grid(True)

    out_path = PLOTS_DIR / "fig2_bland_altman_grid.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig3_confusion_matrices():
    """Figure 3: 4x4 heatmap grid of confusion matrices across 4 models and 4 anatomical regions."""
    cm_df_path = OUTPUTS_DIR / "03_categorical_confusion_matrices.csv"
    if not cm_df_path.exists():
        return

    cm_df = pd.read_csv(cm_df_path)
    # Filter for educated condition and consensus benchmark
    sub = cm_df[(cm_df["condition"] == "educated") & (cm_df["benchmark_type"] == "model_consensus")]

    cat_vars = ["midface", "upper_lip", "lower_lip", "chin_pogonion"]
    fig, axes = plt.subplots(4, 4, figsize=(15, 14), constrained_layout=True)
    fig.suptitle("Figure 3: Diagnostic Confusion Matrices vs. Consensus Benchmark (Educated)", fontsize=14, fontweight="bold")

    labels_abbr = ["Hypo", "Normo", "Hyper"]

    for i, var_name in enumerate(cat_vars):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            cell_data = sub[(sub["variable"] == var_name) & (sub["model"] == model)]

            # Reconstruct 3x3 matrix
            matrix = np.zeros((3, 3), dtype=int)
            for _, r in cell_data.iterrows():
                ti = CATEGORIES.index(r["true_category"])
                pj = CATEGORIES.index(r["pred_category"])
                matrix[ti, pj] = int(r["count"])

            total_n = np.sum(matrix)
            accuracy = (np.trace(matrix) / total_n * 100) if total_n > 0 else 0

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=labels_abbr,
                yticklabels=labels_abbr if j == 0 else False,
                ax=ax,
            )

            ax.set_title(f"{model.upper()} | {var_name}\n(Acc: {accuracy:.1f}%)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"True / Benchmark\n({var_name})", fontsize=9, fontweight="bold")
            if i == 3:
                ax.set_xlabel("AI Prediction", fontsize=9)

    out_path = PLOTS_DIR / "fig3_confusion_matrices_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig4_condition_paired_comparison():
    """Figure 4: Slope/Paired-shift plot comparing Educated vs Uneducated outputs across patients."""
    prof_edu, prof_unedu = load_profile_data()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Figure 4: Paired Shift between Educated vs. Uneducated Conditions (By Model)", fontsize=14, fontweight="bold")

    test_vars = [
        ("maxilla_advancement_mm", "Maxillary Advancement (mm)", axes[0, 0]),
        ("mandible_advancement_mm", "Mandibular Advancement (mm)", axes[0, 1]),
        ("maxilla_impaction_mm", "Maxillary Impaction (mm)", axes[1, 0]),
        ("mandible_impaction_mm", "Mandibular Impaction (mm)", axes[1, 1]),
    ]

    for var_col, var_title, ax in test_vars:
        box_data = []
        labels = []
        palette_list = []

        for m in MODELS:
            sub_edu = prof_edu[prof_edu["model_short"] == m].set_index("patient_name")[var_col].dropna()
            sub_unedu = prof_unedu[prof_unedu["model_short"] == m].set_index("patient_name")[var_col].dropna()

            common = [p for p in sub_edu.index if p in sub_unedu.index]
            for p in common:
                v_u = sub_unedu.loc[p]
                v_e = sub_edu.loc[p]
                box_data.append({"Model": m.upper(), "Condition": "Uneducated", "Value": v_u, "Patient": p})
                box_data.append({"Model": m.upper(), "Condition": "Educated", "Value": v_e, "Patient": p})

        plot_df = pd.DataFrame(box_data)
        sns.boxplot(
            data=plot_df,
            x="Model",
            y="Value",
            hue="Condition",
            palette={"Uneducated": "#bdc9e1", "Educated": "#02818a"},
            ax=ax,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "5"}
        )

        ax.set_title(var_title, fontsize=11, fontweight="semibold")
        ax.set_ylabel("Movement (mm)", fontsize=9)
        ax.set_xlabel("")
        ax.grid(True)
        if var_col != "maxilla_advancement_mm":
            ax.get_legend().remove()
        else:
            ax.legend(title="Condition", loc="upper right")

    out_path = PLOTS_DIR / "fig4_condition_paired_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig5_accuracy_mae_rmse():
    """Figure 5: Bar chart with error bars showing MAE & RMSE across models and conditions."""
    metrics_path = OUTPUTS_DIR / "02_continuous_vs_reference_metrics.csv"
    if not metrics_path.exists():
        return

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Figure 5: Surgical Movement Prediction Error vs. Reference Plan (MAE & RMSE)", fontsize=14, fontweight="bold")

    # MAE Plot
    sns.barplot(
        data=df,
        x="variable",
        y="mae",
        hue="model",
        ax=axes[0],
        palette=PALETTE,
        edgecolor="black"
    )
    axes[0].set_title("Mean Absolute Error (MAE in mm) - Lower is Better", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("MAE (mm)", fontsize=10)
    axes[0].set_xlabel("")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["Max Adv", "Max Imp", "Mand Adv", "Mand Imp"], rotation=15)
    axes[0].axhline(2.0, color="red", linestyle="--", linewidth=1.5, label="2.0 mm Clinical Threshold")
    axes[0].grid(True)
    axes[0].legend(title="Model", loc="upper right")

    # RMSE Plot
    sns.barplot(
        data=df,
        x="variable",
        y="rmse",
        hue="model",
        ax=axes[1],
        palette=PALETTE,
        edgecolor="black"
    )
    axes[1].set_title("Root Mean Square Error (RMSE in mm) - Lower is Better", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMSE (mm)", fontsize=10)
    axes[1].set_xlabel("")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["Max Adv", "Max Imp", "Mand Adv", "Mand Imp"], rotation=15)
    axes[1].grid(True)
    axes[1].legend(title="Model", loc="upper right")

    out_path = PLOTS_DIR / "fig5_model_accuracy_mae_rmse.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig6_resource_tradeoffs():
    """Figure 6: Token usage, Cost ($), and API response time comparisons across models."""
    res_df = load_resource_data()
    if res_df.empty:
        return

    res_df = res_df[res_df["model_short"].isin(MODELS)].copy()
    if "api_duration_ms" in res_df.columns:
        res_df["duration_sec"] = res_df["api_duration_ms"] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("Figure 6: Resource Utilization & Trade-offs (Tokens, Cost, Latency)", fontsize=14, fontweight="bold")

    # 1. Total Tokens
    sns.boxplot(
        data=res_df,
        x="model_short",
        y="total_tokens",
        hue="condition",
        palette={"uneducated": "#ece7f2", "educated": "#2b8cbe"},
        ax=axes[0],
        showfliers=False
    )
    axes[0].set_title("Total Tokens per Patient Prompt", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("Tokens", fontsize=10)
    axes[0].set_xlabel("Model", fontsize=10)
    axes[0].grid(True)

    # 2. Cost (USD)
    sns.barplot(
        data=res_df,
        x="model_short",
        y="cost_usd",
        hue="condition",
        palette={"uneducated": "#fee8c8", "educated": "#e34a33"},
        ax=axes[1],
        estimator=np.mean,
        errorbar=None,
        edgecolor="black"
    )
    axes[1].set_title("Mean Cost per Call (USD)", fontsize=11, fontweight="semibold")
    axes[1].set_ylabel("Cost ($ USD)", fontsize=10)
    axes[1].set_xlabel("Model", fontsize=10)
    axes[1].grid(True)

    # 3. Latency / Response Time (Seconds)
    sns.boxplot(
        data=res_df,
        x="model_short",
        y="duration_sec",
        hue="condition",
        palette={"uneducated": "#e5f5f9", "educated": "#2ca25f"},
        ax=axes[2],
        showfliers=False
    )
    axes[2].set_title("API Response Duration (Seconds)", fontsize=11, fontweight="semibold")
    axes[2].set_ylabel("Seconds", fontsize=10)
    axes[2].set_xlabel("Model", fontsize=10)
    axes[2].grid(True)

    out_path = PLOTS_DIR / "fig6_resource_tradeoffs.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


def plot_fig7_frontal_correlation():
    """Figure 7: Scatter plot with regression line of Nasal Tip vs Chin Point Deviations."""
    front_edu, _ = load_front_data()
    sub = front_edu[front_edu["model_short"].isin(MODELS)].dropna(subset=["nasal_tip_deviation_right_mm", "chin_point_deviation_right_mm"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for m in MODELS:
        m_sub = sub[sub["model_short"] == m]
        ax.scatter(
            m_sub["nasal_tip_deviation_right_mm"],
            m_sub["chin_point_deviation_right_mm"],
            color=PALETTE[m],
            label=m.upper(),
            s=40,
            alpha=0.8,
            edgecolor="white"
        )

    # Overall regression line
    sns.regplot(
        data=sub,
        x="nasal_tip_deviation_right_mm",
        y="chin_point_deviation_right_mm",
        scatter=False,
        ax=ax,
        color="#333333",
        line_kws={"linestyle": "--", "linewidth": 1.5, "label": "Linear Fit"}
    )

    r_val, p_val = stats.pearsonr(sub["nasal_tip_deviation_right_mm"], sub["chin_point_deviation_right_mm"])
    rho_val, rho_p = stats.spearmanr(sub["nasal_tip_deviation_right_mm"], sub["chin_point_deviation_right_mm"])

    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.0)

    ax.set_title(
        f"Figure 7: Facial Asymmetry Internal Consistency (Frontal View)\n"
        f"Pearson r = {r_val:.3f} (p = {p_val:.4f}) | Spearman ρ = {rho_val:.3f} (p = {rho_p:.4f})",
        fontsize=11,
        fontweight="bold"
    )
    ax.set_xlabel("Nasal Tip Deviation to Right (mm)", fontsize=10)
    ax.set_ylabel("Chin Point Deviation to Right (mm)", fontsize=10)
    ax.legend(title="AI Model", loc="upper left", frameon=True)
    ax.grid(True)

    out_path = PLOTS_DIR / "fig7_frontal_asymmetry_correlation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES (300 DPI)")
    print("=" * 60)
    plot_fig1_normality_and_histograms()
    plot_fig2_bland_altman_grid()
    plot_fig3_confusion_matrices()
    plot_fig4_condition_paired_comparison()
    plot_fig5_accuracy_mae_rmse()
    plot_fig6_resource_tradeoffs()
    plot_fig7_frontal_correlation()
    print("=" * 60)
    print(f"ALL PLOTS GENERATED SUCCESSFULLY in {PLOTS_DIR}")
    print("=" * 60)
