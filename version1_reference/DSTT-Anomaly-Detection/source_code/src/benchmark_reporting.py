from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from .thresholds import (
    METHOD_FIXED_MEAN_SIGMA,
    METHOD_FIXED_PERCENTILE,
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_NAMES,
    METHOD_ROLLING_MEAN_3SIGMA,
)

METHOD_ORDER = [
    METHOD_FIXED_PERCENTILE,
    METHOD_FIXED_MEAN_SIGMA,
    METHOD_ROLLING_MEAN_3SIGMA,
    METHOD_MOVING_MEDIAN_MAD,
]
BASELINES = [METHOD_FIXED_PERCENTILE, METHOD_FIXED_MEAN_SIGMA, METHOD_ROLLING_MEAN_3SIGMA]
METRIC_DIRECTIONS = {
    "f1_score": "higher",
    "pr_auc": "higher",
    "nab_style_score": "higher",
    "event_recall": "higher",
    "false_positive_rate": "lower",
    "high_variance_fpr": "lower",
    "mean_detection_delay_fraction": "lower",
}


def _safe_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return math.nan, math.nan
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return math.nan, math.nan


def _safe_paired_ttest(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return math.nan, math.nan
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    if np.std(differences, ddof=1) < 1e-12:
        return (math.inf if float(np.mean(differences)) > 0 else -math.inf), 0.0
    result = ttest_rel(differences, np.zeros_like(differences))
    return float(result.statistic), float(result.pvalue)


def _bootstrap_mean_ci(
    values: Iterable[float], *, seed: int = 42, n_bootstrap: int = 5000
) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction, preserving missing p-values."""
    result = [math.nan] * len(p_values)
    valid = [(i, float(p)) for i, p in enumerate(p_values) if np.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    m = len(valid)
    for rank, (original_index, p_value) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * p_value)
        running = max(running, adjusted)
        result[original_index] = running
    return result


def dataset_level_means(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset/method, averaging repeated seeds and window sizes.

    This avoids treating three seeds or three sequence lengths from the same NAB
    file as independent datasets in paired statistical comparisons.
    """
    group_cols = ["dataset", "domain", "method", "method_name"]
    metric_cols = [name for name in METRIC_DIRECTIONS if name in metrics.columns]
    metric_cols += [
        name for name in ["test_anomaly_events", "test_positive_points", "high_variance_normal_points"]
        if name in metrics.columns and name not in metric_cols
    ]
    grouped = metrics.groupby(group_cols, as_index=False, dropna=False)
    return grouped[metric_cols].mean(numeric_only=True)


def _pairwise_table(dataset_means: pd.DataFrame) -> pd.DataFrame:
    pending: list[dict] = []

    pivot_frames = {
        metric: dataset_means.pivot(index=["dataset", "domain"], columns="method", values=metric)
        for metric in METRIC_DIRECTIONS
        if metric in dataset_means.columns
    }
    events = dataset_means.groupby(["dataset", "domain"])["test_anomaly_events"].max() if "test_anomaly_events" in dataset_means.columns else None
    high_var_points = dataset_means.groupby(["dataset", "domain"])["high_variance_normal_points"].max() if "high_variance_normal_points" in dataset_means.columns else None

    for metric, pivot in pivot_frames.items():
        if METHOD_MOVING_MEDIAN_MAD not in pivot.columns:
            continue
        for baseline in BASELINES:
            if baseline not in pivot.columns:
                continue
            paired = pivot[[METHOD_MOVING_MEDIAN_MAD, baseline]].dropna()
            if metric in {"f1_score", "nab_style_score", "event_recall"} and events is not None:
                eligible = events[events > 0].index
                paired = paired.loc[paired.index.intersection(eligible)]
            if metric == "high_variance_fpr" and high_var_points is not None:
                eligible = high_var_points[high_var_points > 0].index
                paired = paired.loc[paired.index.intersection(eligible)]

            proposed_values = paired[METHOD_MOVING_MEDIAN_MAD].to_numpy(dtype=float)
            baseline_values = paired[baseline].to_numpy(dtype=float)
            direction = METRIC_DIRECTIONS[metric]
            delta = proposed_values - baseline_values if direction == "higher" else baseline_values - proposed_values
            if metric == "high_variance_fpr":
                stat, p_value = _safe_paired_ttest(delta)
                test_name = "Paired t-test (two-sided; proposal protocol)"
            else:
                stat, p_value = _safe_wilcoxon(delta)
                test_name = "Wilcoxon signed-rank (two-sided)"
            ci_low, ci_high = _bootstrap_mean_ci(delta, seed=42 + len(pending))
            row = {
                "metric": metric,
                "comparison": f"{METHOD_NAMES[METHOD_MOVING_MEDIAN_MAD]} vs {METHOD_NAMES[baseline]}",
                "baseline_method": baseline,
                "n_datasets": int(len(delta)),
                "mean_improvement_directional": float(np.mean(delta)) if len(delta) else math.nan,
                "median_improvement_directional": float(np.median(delta)) if len(delta) else math.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "proposed_better_datasets": int(np.sum(delta > 0)) if len(delta) else 0,
                "baseline_better_datasets": int(np.sum(delta < 0)) if len(delta) else 0,
                "ties": int(np.sum(np.isclose(delta, 0))) if len(delta) else 0,
                "statistical_test": test_name,
                "test_statistic": stat,
                "p_value": p_value,
            }
            if metric == "high_variance_fpr":
                valid_base = baseline_values > 0
                relative = ((baseline_values[valid_base] - proposed_values[valid_base]) / baseline_values[valid_base])
                row["mean_relative_fpr_reduction"] = float(np.mean(relative)) if len(relative) else math.nan
                row["datasets_with_at_least_15pct_relative_reduction"] = int(np.sum(relative >= 0.15)) if len(relative) else 0
                row["datasets_with_positive_baseline_fpr"] = int(np.sum(valid_base))
            pending.append(row)

    for metric in METRIC_DIRECTIONS:
        indices = [i for i, row in enumerate(pending) if row["metric"] == metric]
        adjusted = _holm_adjust([pending[i]["p_value"] for i in indices])
        for i, value in zip(indices, adjusted):
            pending[i]["p_value_holm"] = value
            pending[i]["significant_holm_0_05"] = bool(np.isfinite(value) and value < 0.05)
    return pd.DataFrame(pending)


def _domain_table(dataset_means: pd.DataFrame) -> pd.DataFrame:
    pivot = dataset_means.pivot(index=["dataset", "domain"], columns="method", values="f1_score")
    required = [METHOD_MOVING_MEDIAN_MAD, METHOD_ROLLING_MEAN_3SIGMA]
    if any(method not in pivot.columns for method in required):
        return pd.DataFrame(columns=["domain", "datasets", "mean_f1_improvement", "ci95_low", "ci95_high"])

    paired = pivot[required].dropna().reset_index()
    if "test_anomaly_events" in dataset_means.columns:
        events = dataset_means.groupby(["dataset", "domain"])["test_anomaly_events"].max().rename("events")
        paired = paired.merge(events.reset_index(), on=["dataset", "domain"], how="left")
        paired = paired[paired["events"] > 0]
    paired["proposed_minus_rolling_mean_f1"] = (
        paired[METHOD_MOVING_MEDIAN_MAD] - paired[METHOD_ROLLING_MEAN_3SIGMA]
    )
    rows: list[dict] = []
    for domain, group in paired.groupby("domain"):
        deltas = group["proposed_minus_rolling_mean_f1"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(deltas, seed=2026 + len(rows))
        stat, p_value = _safe_wilcoxon(deltas)
        rows.append({
            "domain": domain,
            "datasets": int(len(deltas)),
            "mean_f1_improvement": float(np.mean(deltas)) if len(deltas) else math.nan,
            "median_f1_improvement": float(np.median(deltas)) if len(deltas) else math.nan,
            "std_f1_improvement": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else math.nan,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "positive_dataset_count": int(np.sum(deltas > 0)),
            "wilcoxon_statistic": stat,
            "p_value_exploratory": p_value,
            "interpretation": "Descriptive; small domain sample size limits inference.",
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_f1_improvement", ascending=False).reset_index(drop=True)
        result["improvement_rank"] = np.arange(1, len(result) + 1)
    return result


def _write_plots(dataset_means: pd.DataFrame, window_means: pd.DataFrame, domain: pd.DataFrame, out: Path) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    for metric, title, ylabel, filename in [
        ("f1_score", "NAB test F1 by threshold method", "F1-score", "benchmark_f1_by_method.png"),
        ("false_positive_rate", "NAB test false-positive rate by threshold method", "False-positive rate", "benchmark_fpr_by_method.png"),
        ("pr_auc", "NAB test PR-AUC by threshold method", "Average precision / PR-AUC", "benchmark_pr_auc_by_method.png"),
        ("nab_style_score", "NAB-style test score by threshold method", "Normalized test score", "benchmark_nab_style_by_method.png"),
        ("high_variance_fpr", "False-positive rate during high-variance periods", "High-variance FPR", "benchmark_high_variance_fpr.png"),
    ]:
        if metric not in dataset_means.columns:
            continue
        rows = []
        for method in METHOD_ORDER:
            values = dataset_means.loc[dataset_means["method"] == method, metric].dropna().to_numpy(dtype=float)
            rows.append((np.mean(values) if len(values) else np.nan,
                         np.std(values, ddof=1) if len(values) > 1 else 0.0,
                         len(values)))
        means = [r[0] for r in rows]
        stds = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([METHOD_NAMES[m] for m in METHOD_ORDER], means, yerr=stds, capsize=4)
        ax.set_title(title + " (mean +/- SD across datasets)")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=17)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)

    if not window_means.empty and "f1_score" in window_means.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHOD_ORDER:
            group = window_means[window_means["method"] == method].sort_values("window_size")
            if group.empty:
                continue
            ax.plot(group["window_size"], group["mean_f1_score"], marker="o", label=METHOD_NAMES[method])
        ax.set_title("F1-score sensitivity to LSTM sequence length")
        ax.set_xlabel("Sequence length W")
        ax.set_ylabel("Mean F1-score across datasets")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "f1_by_sequence_length.png", dpi=160)
        plt.close(fig)

    if not domain.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(domain["domain"], domain["mean_f1_improvement"],
               yerr=domain["std_f1_improvement"].fillna(0), capsize=4)
        ax.axhline(0.0, linewidth=1)
        ax.set_title("Proposed method F1 improvement by NAB domain")
        ax.set_ylabel("Proposed - rolling mean + 3 sigma F1")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "domain_f1_improvement.png", dpi=160)
        plt.close(fig)


def _make_rq_report(
    dataset_means: pd.DataFrame,
    method_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    domain: pd.DataFrame,
    hypothesis_outcomes: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = [
        "# Research-question results generated from the held-out NAB experiment",
        "",
        "> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.",
        "> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.",
        "",
        "## RQ1 - Baseline performance",
        "How do fixed percentile, fixed mean + k sigma, and rolling mean + 3 sigma perform on the selected NAB files?",
        "",
        "| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in method_summary.iterrows():
        lines.append(
            f"| {row['method_name']} | {row.get('f1_score_mean', math.nan):.4f} (n={int(row.get('f1_score_count', 0))}) | "
            f"{row.get('false_positive_rate_mean', math.nan):.4f} | "
            f"{row.get('pr_auc_mean', math.nan):.4f} | {row.get('nab_style_score_mean', math.nan):.2f} (n={int(row.get('nab_style_score_count', 0))}) |"
        )
    lines += ["", "## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?", ""]
    rq2 = pairwise[pairwise["metric"].isin(["f1_score", "nab_style_score", "pr_auc"])] if not pairwise.empty else pairwise
    if rq2.empty:
        lines.append("Not enough paired, evaluable datasets to calculate the comparisons.")
    else:
        lines += [
            "| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, row in rq2.iterrows():
            lines.append(
                f"| {row['metric']} | {METHOD_NAMES.get(row['baseline_method'], row['baseline_method'])} | "
                f"{int(row['n_datasets'])} | {row['mean_improvement_directional']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row.get('p_value_holm', math.nan):.4g} |"
            )
    lines += ["", "## RQ3 - False alarms during high-variance periods", ""]
    rq3 = pairwise[pairwise["metric"] == "high_variance_fpr"] if not pairwise.empty else pairwise
    if rq3.empty:
        lines.append("No datasets had enough normal high-variance test points for a paired comparison.")
    else:
        lines += [
            "| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in rq3.iterrows():
            relative_reduction = row.get('mean_relative_fpr_reduction', math.nan)
            relative_text = f"{relative_reduction:.1%}" if np.isfinite(relative_reduction) else "N/A"
            adjusted_p = row.get('p_value_holm', math.nan)
            p_text = f"{adjusted_p:.4g}" if np.isfinite(adjusted_p) else "N/A"
            lines.append(
                f"| {METHOD_NAMES.get(row['baseline_method'], row['baseline_method'])} | {int(row['n_datasets'])} | "
                f"{row['mean_improvement_directional']:.4f} | {relative_text} | "
                f"{int(row.get('datasets_with_at_least_15pct_relative_reduction', 0))} | {p_text} |"
            )
        lines.append("")
        lines.append("For H2's relative-reduction criterion, inspect `high_variance_fpr_by_dataset.csv` and the paired comparison output; percentage reductions are only defined when the baseline high-variance FPR is greater than zero.")
    lines += ["", "## RQ4 - Which NAB domain benefits most?", ""]
    if domain.empty:
        lines.append("No domain had enough paired test results to calculate a domain comparison.")
    else:
        lines += [
            "| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in domain.iterrows():
            lines.append(
                f"| {row['domain']} | {int(row['datasets'])} | {row['mean_f1_improvement']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{int(row['positive_dataset_count'])} | {int(row['improvement_rank'])} |"
            )
        lines += [
            "",
            "Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.",
        ]
    if hypothesis_outcomes is not None and not hypothesis_outcomes.empty:
        lines += ["", "## Proposal hypothesis checks", "", "| Hypothesis | Outcome | Evidence count | Evaluated / configured | p-value |", "|---|---|---:|---:|---:|"]
        for _, row in hypothesis_outcomes.iterrows():
            count = row.get("observed_count", math.nan)
            evaluable = row.get("evaluable_datasets", row.get("configured_datasets", math.nan))
            configured = row.get("configured_datasets", math.nan)
            p_value = row.get("p_value", math.nan)
            count_text = str(int(count)) if np.isfinite(count) else "N/A"
            eval_text = str(int(evaluable)) if np.isfinite(evaluable) else "N/A"
            config_text = str(int(configured)) if np.isfinite(configured) else "N/A"
            p_text = f"{p_value:.4g}" if np.isfinite(p_value) else "N/A"
            lines.append(f"| {row['hypothesis']} | {row['outcome']} | {count_text} | {eval_text} / {config_text} | {p_text} |")
    lines += [
        "",
        "## Interpretation rules",
        "- The validation partition selects threshold hyperparameters; only the test partition is used for the final reported performance.",
        "- Paired statistical comparisons use one averaged observation per dataset, not each seed/window as an independent sample.",
        "- A file with no complete labeled test event is not evidence for anomaly-F1 superiority or event-level NAB-style score.",
        "- `Not supported / inconclusive` means the pre-defined criterion was not met or the number of evaluable files was insufficient; it is not proof that the methods are equivalent.",
        "- State the number of evaluable files alongside every aggregate result.",
        "",
    ]
    return "\n".join(lines)


def write_benchmark_outputs(
    metrics: pd.DataFrame,
    output_dir: Path,
    hypothesis_outcomes: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate benchmark tables, pairwise method comparisons, and RQ report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_means = dataset_level_means(metrics)
    dataset_means.to_csv(output_dir / "per_dataset_comparison.csv", index=False)

    metric_cols = [name for name in METRIC_DIRECTIONS if name in dataset_means.columns]
    method_summary = (
        dataset_means.groupby(["method", "method_name"], as_index=False)[metric_cols]
        .agg(["mean", "std", "count"])
    )
    # Flatten a multi-index produced by pandas aggregation into conventional CSV fields.
    method_summary.columns = [
        "_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column)
        for column in method_summary.columns
    ]
    method_order_map = {method: index for index, method in enumerate(METHOD_ORDER)}
    method_summary["_order"] = method_summary["method"].map(method_order_map)
    method_summary = method_summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    method_summary.to_csv(output_dir / "benchmark_method_summary.csv", index=False)

    pairwise = _pairwise_table(dataset_means)
    pairwise.to_csv(output_dir / "pairwise_proposed_vs_baselines.csv", index=False)

    # Average over seeds only; retain W so sequence-length effects remain visible.
    run_metric_cols = [name for name in METRIC_DIRECTIONS if name in metrics.columns]
    window_means = (
        metrics.groupby(["window_size", "method", "method_name"], as_index=False)[run_metric_cols]
        .mean(numeric_only=True)
        .rename(columns={name: f"mean_{name}" for name in run_metric_cols})
    )
    window_means.to_csv(output_dir / "comparison_by_window_size.csv", index=False)

    seed_variability = (
        metrics.groupby(["dataset", "domain", "window_size", "method", "method_name"], as_index=False)
        .agg(
            f1_mean_across_seeds=("f1_score", "mean"),
            f1_std_across_seeds=("f1_score", "std"),
            fpr_mean_across_seeds=("false_positive_rate", "mean"),
            fpr_std_across_seeds=("false_positive_rate", "std"),
            nab_style_mean_across_seeds=("nab_style_score", "mean"),
            nab_style_std_across_seeds=("nab_style_score", "std"),
            seed_runs=("seed", "count"),
        )
    )
    seed_variability.to_csv(output_dir / "seed_variability.csv", index=False)

    domain = _domain_table(dataset_means)
    domain.to_csv(output_dir / "domain_comparison.csv", index=False)

    # Dataset-level improvements for H2; the baseline FPR must be positive for
    # a relative percentage reduction to be defined.
    h2_pivot = dataset_means.pivot(index=["dataset", "domain"], columns="method", values="high_variance_fpr") if "high_variance_fpr" in dataset_means.columns else pd.DataFrame()
    if not h2_pivot.empty and METHOD_MOVING_MEDIAN_MAD in h2_pivot and METHOD_ROLLING_MEAN_3SIGMA in h2_pivot:
        h2 = h2_pivot[[METHOD_MOVING_MEDIAN_MAD, METHOD_ROLLING_MEAN_3SIGMA]].reset_index()
        h2 = h2.rename(columns={METHOD_MOVING_MEDIAN_MAD: "proposed_high_variance_fpr", METHOD_ROLLING_MEAN_3SIGMA: "baseline_high_variance_fpr"})
        h2["absolute_fpr_reduction"] = h2["baseline_high_variance_fpr"] - h2["proposed_high_variance_fpr"]
        h2["relative_fpr_reduction"] = np.where(
            h2["baseline_high_variance_fpr"] > 0,
            h2["absolute_fpr_reduction"] / h2["baseline_high_variance_fpr"],
            np.nan,
        )
    else:
        h2 = pd.DataFrame()
    h2.to_csv(output_dir / "high_variance_fpr_by_dataset.csv", index=False)

    _write_plots(dataset_means, window_means, domain, output_dir)
    report = _make_rq_report(dataset_means, method_summary, pairwise, domain, hypothesis_outcomes)
    (output_dir / "research_question_results.md").write_text(report, encoding="utf-8")

    return {
        "dataset_means": dataset_means,
        "method_summary": method_summary,
        "pairwise": pairwise,
        "window_means": window_means,
        "seed_variability": seed_variability,
        "domain": domain,
        "high_variance": h2,
    }
