from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from .nab_scoring import nab_style_test_score
from .thresholds import METHOD_MOVING_MEDIAN_MAD, METHOD_ROLLING_MEAN_3SIGMA


def rolling_past_std(values: np.ndarray, indices: np.ndarray, lookback: int) -> np.ndarray:
    """Causal standard deviation using up to `lookback` observations strictly before t."""
    values = np.asarray(values, dtype=float)
    output = np.zeros(len(indices), dtype=float)
    for j, end_index in enumerate(np.asarray(indices, dtype=int)):
        history = values[max(0, end_index - lookback) : end_index]
        output[j] = float(np.std(history, ddof=0)) if len(history) >= 2 else 0.0
    return output


def high_variance_mask(
    all_values: np.ndarray,
    train_end: int,
    test_indices: np.ndarray,
    lookback: int,
    percentile: float = 75.0,
) -> np.ndarray:
    """Flag test rows with trailing volatility >= a train-only percentile cutoff."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    train_indices = np.arange(1, train_end, dtype=int)
    train_volatility = rolling_past_std(all_values, train_indices, lookback)
    cutoff = float(np.percentile(train_volatility, percentile)) if len(train_volatility) else 0.0
    test_volatility = rolling_past_std(all_values, test_indices, lookback)
    return test_volatility >= cutoff


def event_level_metrics(
    timestamps: Iterable,
    predictions: np.ndarray,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    """Evaluate only complete labeled events within the test partition."""
    times = pd.DatetimeIndex(timestamps)
    predictions = np.asarray(predictions, dtype=np.int8)
    detected_events = 0
    delay_fractions: list[float] = []
    early_credits: list[float] = []

    for start, end in windows:
        indices = np.flatnonzero((times >= start) & (times <= end))
        if len(indices) == 0:
            continue
        detections = indices[predictions[indices] == 1]
        if len(detections):
            detected_events += 1
            first_detection_time = times[detections[0]]
            duration = max((end - start).total_seconds(), 1.0)
            delay = float((first_detection_time - start).total_seconds() / duration)
            delay = float(np.clip(delay, 0.0, 1.0))
            delay_fractions.append(delay)
            early_credits.append(1.0 - delay)

    considered_events = len(windows)
    return {
        "test_anomaly_events": float(considered_events),
        "detected_anomaly_events": float(detected_events),
        "event_recall": float(detected_events / considered_events) if considered_events else math.nan,
        "mean_detection_delay_fraction": float(np.mean(delay_fractions)) if delay_fractions else math.nan,
        "mean_early_detection_credit": float(np.mean(early_credits)) if early_credits else math.nan,
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    test_timestamps: Iterable,
    test_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    high_variance_points: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths do not match")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_true and y_pred must contain only 0/1 values")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) else math.nan
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)
    pr_auc = (
        float(average_precision_score(y_true, anomaly_scores))
        if len(np.unique(y_true)) == 2
        else math.nan
    )

    high_variance_points = np.asarray(high_variance_points, dtype=bool)
    if len(high_variance_points) != len(y_true):
        raise ValueError("high_variance_points and y_true lengths do not match")
    normal_high_variance = high_variance_points & (y_true == 0)
    if normal_high_variance.any():
        high_variance_fpr = float(np.mean(y_pred[normal_high_variance] == 1))
    else:
        high_variance_fpr = math.nan

    event_metrics = event_level_metrics(test_timestamps, y_pred, test_windows)
    nab_metrics = nab_style_test_score(test_timestamps, y_pred, test_windows)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "high_variance_fpr": high_variance_fpr,
        "high_variance_normal_points": int(normal_high_variance.sum()),
        **event_metrics,
        **nab_metrics,
    }


def _safe_wilcoxon(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 2:
        return math.nan, math.nan
    if np.allclose(first - second, 0.0):
        return 0.0, 1.0
    result = wilcoxon(first, second, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def _safe_ttest(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 2:
        return math.nan, math.nan
    differences = first - second
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    if np.std(differences, ddof=1) < 1e-12:
        # Exact constant non-zero paired differences imply an infinite t statistic.
        return (math.inf if float(np.mean(differences)) > 0 else -math.inf), 0.0
    result = ttest_rel(first, second)
    return float(result.statistic), float(result.pvalue)


def _dataset_method_means(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Average each dataset-method result over sequence lengths and random seeds."""
    fields = ["f1_score", "false_positive_rate", "high_variance_fpr", "pr_auc", "nab_style_score"]
    available = [field for field in fields if field in metrics_frame.columns]
    grouped = metrics_frame.groupby(["dataset", "domain", "method", "method_name"], as_index=False)
    means = grouped[available].mean(numeric_only=True)
    # These fields describe the shared dataset/test partition, not a method result.
    context_fields = [field for field in ["test_anomaly_events", "test_positive_points"] if field in metrics_frame.columns]
    if context_fields:
        context = grouped[context_fields].max(numeric_only=True)
        means = means.merge(context, on=["dataset", "domain", "method", "method_name"], how="left")
    return means


def build_statistical_outputs(
    metrics_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build statistical tests, hypothesis decision table, and domain effects."""
    required = {"dataset", "domain", "method", "method_name", "f1_score", "high_variance_fpr"}
    missing = required.difference(metrics_frame.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")

    dataset_means = _dataset_method_means(metrics_frame)
    proposed = METHOD_MOVING_MEDIAN_MAD
    baseline = METHOD_ROLLING_MEAN_3SIGMA
    pivot_f1 = dataset_means.pivot(index=["dataset", "domain"], columns="method", values="f1_score")
    pivot_fpr = dataset_means.pivot(index=["dataset", "domain"], columns="method", values="high_variance_fpr")

    # H1: count the direction of change over all configured files; inferential test
    # is restricted to files with usable, non-missing test anomaly labels.
    paired_f1 = pivot_f1[[proposed, baseline]].dropna()
    # Files without any labeled anomaly in the held-out test split cannot
    # establish anomaly-F1 superiority. Keep them in the configured denominator
    # but exclude them from directional counts and inferential testing.
    if "test_anomaly_events" in dataset_means.columns:
        event_counts = dataset_means.groupby(["dataset", "domain"])['test_anomaly_events'].max()
        eligible_keys = event_counts[event_counts > 0].index
        eligible_index = pd.MultiIndex.from_tuples(eligible_keys, names=["dataset", "domain"])
        paired_f1 = paired_f1.loc[paired_f1.index.intersection(eligible_index)]
    f1_differences = paired_f1[proposed] - paired_f1[baseline]
    h1_improved = int((f1_differences > 0).sum())
    total_datasets = int(metrics_frame["dataset"].nunique())
    n_f1_evaluable = int(len(paired_f1))
    stat_h1, p_h1 = _safe_wilcoxon(
        paired_f1[proposed].to_numpy(), paired_f1[baseline].to_numpy()
    ) if len(paired_f1) else (math.nan, math.nan)

    # H2: high-variance false-positive rate comparison.
    paired_fpr = pivot_fpr[[proposed, baseline]].dropna()
    stat_h2, p_h2 = _safe_ttest(
        paired_fpr[baseline].to_numpy(), paired_fpr[proposed].to_numpy()
    ) if len(paired_fpr) else (math.nan, math.nan)
    valid_relative = paired_fpr[baseline] > 0
    relative_reduction = (
        (paired_fpr.loc[valid_relative, baseline] - paired_fpr.loc[valid_relative, proposed])
        / paired_fpr.loc[valid_relative, baseline]
    )
    mean_relative_reduction = float(relative_reduction.mean()) if len(relative_reduction) else math.nan
    h2_count = int((relative_reduction >= 0.15).sum())

    # H3: descriptive domain ranking, not a high-powered significance test.
    domain_source = dataset_means
    if "test_anomaly_events" in domain_source.columns:
        domain_source = domain_source[domain_source["test_anomaly_events"] > 0]
    domain_effects = domain_source.pivot_table(
        index=["dataset", "domain"], columns="method", values="f1_score", aggfunc="mean"
    ).reset_index()
    domain_effects["proposed_minus_rolling_mean_f1"] = domain_effects[proposed] - domain_effects[baseline]
    domain_summary = (
        domain_effects.groupby("domain", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_f1_improvement=("proposed_minus_rolling_mean_f1", "mean"),
            std_f1_improvement=("proposed_minus_rolling_mean_f1", "std"),
            positive_dataset_count=("proposed_minus_rolling_mean_f1", lambda x: int((x > 0).sum())),
        )
        .sort_values("mean_f1_improvement", ascending=False)
        .reset_index(drop=True)
    )
    domain_summary["improvement_rank"] = np.arange(1, len(domain_summary) + 1)
    aws_rank = (
        int(domain_summary.loc[domain_summary["domain"] == "AWS CloudWatch", "improvement_rank"].iloc[0])
        if (domain_summary["domain"] == "AWS CloudWatch").any() else math.nan
    )

    statistical_tests = pd.DataFrame([
        {
            "hypothesis": "H1: Proposed F1 vs rolling mean + 3 sigma",
            "test": "Wilcoxon signed-rank (two-sided)",
            "n_datasets_tested": n_f1_evaluable,
            "datasets_improved_proposed": h1_improved,
            "datasets_configured": total_datasets,
            "mean_f1_difference_proposed_minus_baseline": float(f1_differences.mean()),
            "statistic": stat_h1,
            "p_value": p_h1,
            "alpha": 0.05,
        },
        {
            "hypothesis": "H2: High-variance FPR reduction vs rolling mean + 3 sigma",
            "test": "Paired t-test (two-sided)",
            "n_datasets_tested": int(len(paired_fpr)),
            "datasets_improved_proposed": int((paired_fpr[proposed] < paired_fpr[baseline]).sum()),
            "datasets_configured": int(len(pivot_fpr)),
            "mean_fpr_difference_proposed_minus_baseline": float((paired_fpr[proposed] - paired_fpr[baseline]).mean()) if len(paired_fpr) else math.nan,
            "mean_relative_fpr_reduction": mean_relative_reduction,
            "datasets_with_at_least_15pct_reduction": h2_count,
            "statistic": stat_h2,
            "p_value": p_h2,
            "alpha": 0.05,
        },
    ])

    h1_supported = (
        total_datasets == 10
        and n_f1_evaluable == 10
        and h1_improved >= 7
        and np.isfinite(p_h1)
        and p_h1 < 0.05
    )
    h2_supported = (
        np.isfinite(mean_relative_reduction)
        and mean_relative_reduction >= 0.15
        and np.isfinite(p_h2)
        and p_h2 < 0.05
    )
    aws_dataset_count = (
        int(domain_summary.loc[domain_summary["domain"] == "AWS CloudWatch", "datasets"].iloc[0])
        if (domain_summary["domain"] == "AWS CloudWatch").any() else 0
    )
    # H3 is an exploratory domain-ranking claim. Do not mark it supported when
    # there is only one domain, or when fewer than two AWS files are evaluable.
    h3_supported = bool(
        aws_rank == 1
        and domain_summary["domain"].nunique() >= 2
        and aws_dataset_count >= 2
    )
    hypothesis_outcomes = pd.DataFrame([
        {
            "hypothesis": "H1: Median + MAD improves F1 on at least 7 of 10 datasets",
            "criterion": "All 10 configured datasets must have a complete labeled test event; at least 7 show higher mean test F1; Wilcoxon p < 0.05 across the 10 paired datasets",
            "observed_count": h1_improved,
            "evaluable_datasets": n_f1_evaluable,
            "configured_datasets": total_datasets,
            "p_value": p_h1,
            "outcome": "Supported" if h1_supported else "Not supported / inconclusive",
        },
        {
            "hypothesis": "H2: Median + MAD reduces high-variance FPR by at least 15%",
            "criterion": "Mean relative reduction >= 15% and paired t-test p < 0.05",
            "observed_count": h2_count,
            "configured_datasets": int(len(pivot_fpr)),
            "p_value": p_h2,
            "outcome": "Supported" if h2_supported else "Not supported / inconclusive",
        },
        {
            "hypothesis": "H3: AWS CloudWatch has the greatest improvement",
            "criterion": "AWS CloudWatch ranks first by mean dataset-level F1 improvement",
            "observed_count": int((domain_summary["domain"] == "AWS CloudWatch").sum()),
            "configured_datasets": int(len(domain_summary)),
            "p_value": math.nan,
            "outcome": "Supported descriptively" if h3_supported else "Not supported descriptively",
        },
    ])
    return statistical_tests, hypothesis_outcomes, domain_summary
