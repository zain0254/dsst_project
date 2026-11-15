from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .nab_scoring import nab_style_test_score
from .thresholds import (
    METHOD_FIXED_PERCENTILE,
    METHOD_FIXED_MEAN_SIGMA,
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_ROLLING_MEAN_3SIGMA,
)


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
    *,
    training_labels: np.ndarray,
) -> np.ndarray:
    """Flag test rows whose trailing volatility exceeds a clean-normal train cutoff.

    The reference distribution contains only full trailing windows whose historical
    points are all labeled normal. This prevents labeled training anomalies from
    inflating the cutoff used to define changing *normal* variability. The current
    point is excluded from each volatility window, preserving causal semantics.
    """
    values = np.asarray(all_values, dtype=float)
    labels = np.asarray(training_labels, dtype=np.int8)
    indices = np.asarray(test_indices, dtype=int)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if len(labels) < train_end:
        raise ValueError("training_labels must cover the complete training partition")
    if train_end > len(values) or np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("train_end or test_indices are outside all_values")

    train_volatility = []
    for end_index in range(lookback, train_end):
        start_index = end_index - lookback
        if np.any(labels[start_index:end_index] != 0):
            continue
        train_volatility.append(float(np.std(values[start_index:end_index], ddof=0)))
    if not train_volatility:
        raise ValueError("No fully normal training history windows are available for volatility calibration")

    cutoff = float(np.percentile(train_volatility, percentile))
    test_volatility = rolling_past_std(values, indices, lookback)
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
    delay_seconds: list[float] = []
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
            elapsed_seconds = max(0.0, float((first_detection_time - start).total_seconds()))
            delay = float(np.clip(elapsed_seconds / duration, 0.0, 1.0))
            delay_seconds.append(elapsed_seconds)
            delay_fractions.append(delay)
            early_credits.append(1.0 - delay)

    considered_events = len(windows)
    return {
        "test_anomaly_events": float(considered_events),
        "detected_anomaly_events": float(detected_events),
        "event_recall": float(detected_events / considered_events) if considered_events else math.nan,
        "mean_detection_delay_fraction": float(np.mean(delay_fractions)) if delay_fractions else math.nan,
        "mean_detection_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else math.nan,
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
    has_both_classes = len(np.unique(y_true)) == 2
    # AP (average precision) is used as the project PR-AUC convention.
    # ROC-AUC is computed from continuous reconstruction errors, not hard labels.
    pr_auc = float(average_precision_score(y_true, anomaly_scores)) if has_both_classes else math.nan
    roc_auc = float(roc_auc_score(y_true, anomaly_scores)) if has_both_classes else math.nan
    accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else math.nan

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
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
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


def _holm_adjust_pvalues(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment, preserving missing p-values."""
    adjusted = [math.nan] * len(p_values)
    valid = [(i, float(p)) for i, p in enumerate(p_values) if np.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    count = len(valid)
    for rank, (original_index, p_value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[original_index] = running
    return adjusted


def _dataset_method_means(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Average each dataset-method result over sequence lengths and random seeds."""
    fields = ["f1_score", "accuracy", "false_positive_rate", "roc_auc", "high_variance_fpr", "pr_auc", "nab_style_score", "mean_detection_delay_seconds"]
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
    """Evaluate the three hypotheses exactly as stated in the submitted proposal.

    RQ4/domain exploration is returned separately as ``domain_summary``; it is not
    relabelled as H3. Repeated seeds/windows are averaged within each dataset before
    inferential tests so they are not treated as independent observations.
    """
    required = {"dataset", "domain", "method", "method_name", "f1_score", "high_variance_fpr"}
    missing = required.difference(metrics_frame.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")

    dataset_means = _dataset_method_means(metrics_frame)
    proposed = METHOD_MOVING_MEDIAN_MAD
    fixed_baselines = [METHOD_FIXED_PERCENTILE, METHOD_FIXED_MEAN_SIGMA]
    all_baselines = fixed_baselines + [METHOD_ROLLING_MEAN_3SIGMA]
    index_cols = ["dataset", "domain"]

    # Event-level comparisons (H3) require at least one complete ground-truth event.
    # Point-level F1 comparisons (H1) use datasets with positive test points.
    event_counts = (
        dataset_means.groupby(index_cols)["test_anomaly_events"].max()
        if "test_anomaly_events" in dataset_means.columns else None
    )
    eligible_event_keys = (
        event_counts[event_counts > 0].index if event_counts is not None else None
    )

    statistical_rows: list[dict] = []
    hypothesis_rows: list[dict] = []

    def _eligible_pair(metric: str, baseline: str, event_only: bool = False) -> pd.DataFrame:
        pivot = dataset_means.pivot(index=index_cols, columns="method", values=metric)
        if proposed not in pivot.columns or baseline not in pivot.columns:
            return pd.DataFrame(columns=[proposed, baseline])
        paired = pivot[[proposed, baseline]].dropna()
        if event_only and eligible_event_keys is not None:
            paired = paired.loc[paired.index.intersection(eligible_event_keys)]
        return paired

    def _append_comparison(
        *, hypothesis: str, metric: str, baseline: str, test: str,
        event_only: bool = False, lower_is_better: bool = False,
    ) -> tuple[pd.DataFrame, float, float, float]:
        paired = _eligible_pair(metric, baseline, event_only=event_only)
        pvals = paired[proposed].to_numpy(dtype=float)
        bvals = paired[baseline].to_numpy(dtype=float)
        difference = pvals - bvals
        if test == "Wilcoxon signed-rank (two-sided)":
            statistic, p_value = _safe_wilcoxon(pvals, bvals)
        else:
            # For FPR, use the same paired t-test specified by the V2 protocol.
            statistic, p_value = _safe_ttest(pvals, bvals)
        better = (difference < 0) if lower_is_better else (difference > 0)
        mean_difference = float(np.mean(difference)) if len(difference) else math.nan
        statistical_rows.append({
            "hypothesis": hypothesis,
            "metric": metric,
            "baseline": baseline,
            "test": test,
            "n_datasets_tested": int(len(difference)),
            "datasets_improved_proposed": int(np.sum(better)),
            "datasets_baseline_better": int(np.sum(~better & ~np.isclose(difference, 0.0))),
            "mean_difference_proposed_minus_baseline": mean_difference,
            "statistic": statistic,
            "p_value": p_value,
            "alpha": 0.05,
        })
        return paired, mean_difference, statistic, p_value

    # H1: proposal wording is "higher average F1 than at least one fixed threshold
    # baseline". The rolling baseline is supplementary, not the H1 comparator.
    h1_results = []
    for baseline in fixed_baselines:
        result = _append_comparison(
            hypothesis="H1: Proposed moving Median + MAD vs fixed-threshold F1",
            metric="f1_score", baseline=baseline,
            test="Wilcoxon signed-rank (two-sided)", event_only=False,
        )
        h1_results.append((baseline, result))
    h1_adjusted = _holm_adjust_pvalues([row[1][3] for row in h1_results])
    for (baseline, _), adjusted in zip(h1_results, h1_adjusted):
        for row in reversed(statistical_rows):
            if row["hypothesis"].startswith("H1") and row["baseline"] == baseline:
                row["p_value_holm_within_hypothesis"] = adjusted
                row["significant_holm_0_05"] = bool(np.isfinite(adjusted) and adjusted < 0.05)
                break
    h1_positive = [item for item in h1_results if np.isfinite(item[1][1]) and item[1][1] > 0]
    # The positive direction and adjusted significance must belong to the same
    # baseline comparison; evidence against one baseline cannot support a win
    # against a different baseline.
    h1_supported = any(
        np.isfinite(result[1]) and result[1] > 0
        and np.isfinite(adjusted) and adjusted < 0.05
        for (_, result), adjusted in zip(h1_results, h1_adjusted)
    )
    n_h1 = max((len(item[1][0]) for item in h1_results), default=0)
    h1_outcome = (
        "Supported on evaluated datasets (Holm-adjusted p < 0.05)" if h1_supported
        else "Directional only; not statistically established" if h1_positive
        else "Not supported on evaluated datasets"
    )
    hypothesis_rows.append({
        "hypothesis": "H1: Proposed moving Median + MAD achieves higher average F1 than at least one fixed-threshold baseline",
        "proposal_wording": "Higher average F1-score than at least one fixed-threshold baseline",
        "evaluable_datasets": n_h1,
        "fixed_baselines_with_positive_mean_difference": ";".join(x[0] for x in h1_positive),
        "outcome": h1_outcome,
        "interpretation": "The inferential sample is point-evaluable datasets only (at least one positive held-out test point); repeated seeds and windows are averaged within dataset.",
    })

    # H2: compare high-variance false-positive rate against both fixed methods and
    # the extra rolling comparator. The rolling comparison is supplementary.
    h2_results = []
    for baseline in all_baselines:
        result = _append_comparison(
            hypothesis="H2: High-variance false-positive rate",
            metric="high_variance_fpr", baseline=baseline,
            test="Paired t-test (two-sided)", lower_is_better=True,
        )
        h2_results.append((baseline, result))
    h2_adjusted = _holm_adjust_pvalues([row[1][3] for row in h2_results])
    for (baseline, _), adjusted in zip(h2_results, h2_adjusted):
        for row in reversed(statistical_rows):
            if row["hypothesis"].startswith("H2") and row["baseline"] == baseline:
                row["p_value_holm_within_hypothesis"] = adjusted
                row["significant_holm_0_05"] = bool(np.isfinite(adjusted) and adjusted < 0.05)
                break
    per_baseline_h2 = []
    for baseline, (paired, mean_diff, _, p_value) in h2_results:
        valid = paired.dropna()
        better_count = int((valid[proposed] < valid[baseline]).sum()) if len(valid) else 0
        worse_count = int((valid[proposed] > valid[baseline]).sum()) if len(valid) else 0
        per_baseline_h2.append((baseline, better_count, worse_count, mean_diff, p_value))
    any_isolated_win = any(x[1] > 0 for x in per_baseline_h2 if x[0] in fixed_baselines)
    overall_worse_significant = any(
        (x[0] == METHOD_ROLLING_MEAN_3SIGMA and np.isfinite(adj) and adj < 0.05 and x[3] > 0)
        for x, adj in zip(per_baseline_h2, h2_adjusted)
    )
    if any_isolated_win and overall_worse_significant:
        h2_outcome = "Mixed: isolated per-series reductions, but significantly higher mean FPR than rolling baseline"
    elif any_isolated_win:
        h2_outcome = "Partially observed on some series; aggregate evidence is inconclusive"
    elif overall_worse_significant:
        h2_outcome = "Not supported; significantly higher high-variance FPR than rolling baseline"
    else:
        h2_outcome = "Not statistically established on evaluated series"
    hypothesis_rows.append({
        "hypothesis": "H2: Proposed threshold reduces false-positive alerts on some series with changing normal variability",
        "proposal_wording": "Reduce false-positive rates on some time series with changing normal variability",
        "evaluable_datasets": max((len(x[1][0]) for x in h2_results), default=0),
        "per_baseline_results": "; ".join(
            f"{b}: better={better}, worse={worse}, mean_delta={delta:.6g}"
            for b, better, worse, delta, _ in per_baseline_h2
        ),
        "outcome": h2_outcome,
        "interpretation": "A per-series win is not presented as a general benefit when the aggregate false-alarm rate is worse.",
    })

    # H3: proposal requires maintaining/improving NAB score with fewer false alerts
    # or earlier detection. NAB-style event metrics are evaluated only where the
    # held-out test segment contains complete anomaly events.
    h3_results = []
    if "nab_style_score" in dataset_means.columns:
        for baseline in all_baselines:
            result = _append_comparison(
                hypothesis="H3: NAB-style event score", metric="nab_style_score",
                baseline=baseline, test="Wilcoxon signed-rank (two-sided)",
                event_only=True,
            )
            h3_results.append((baseline, result))
    h3_positive = [x for x in h3_results if x[0] in fixed_baselines and np.isfinite(x[1][1]) and x[1][1] >= 0]
    h3_adjusted = _holm_adjust_pvalues([x[1][3] for x in h3_results])
    for (baseline, _), adjusted in zip(h3_results, h3_adjusted):
        for row in reversed(statistical_rows):
            if row["hypothesis"].startswith("H3") and row["baseline"] == baseline:
                row["p_value_holm_within_hypothesis"] = adjusted
                row["significant_holm_0_05"] = bool(np.isfinite(adjusted) and adjusted < 0.05)
                break

    # Descriptive support components: compare mean FPR and mean delay to each
    # fixed baseline on the same event-evaluable dataset set.
    fpr_improved_all = True
    delay_improved_any = False
    h3_component_notes: list[str] = []
    for baseline in fixed_baselines:
        fpr_pair = _eligible_pair("false_positive_rate", baseline, event_only=True) if "false_positive_rate" in dataset_means.columns else pd.DataFrame()
        if len(fpr_pair):
            fpr_delta = fpr_pair[proposed] - fpr_pair[baseline]
            fpr_improved = float(fpr_delta.mean()) <= 0
            fpr_improved_all = fpr_improved_all and fpr_improved
            h3_component_notes.append(f"{baseline} mean FPR delta={float(fpr_delta.mean()):.6g}")
        delay_col = "mean_detection_delay_fraction"
        if delay_col in metrics_frame.columns:
            delay_means = metrics_frame.groupby(["dataset", "domain", "method"], as_index=False)[delay_col].mean()
            delay_pivot = delay_means.pivot(index=index_cols, columns="method", values=delay_col)
            if proposed in delay_pivot.columns and baseline in delay_pivot.columns:
                delay_pair = delay_pivot[[proposed, baseline]].dropna()
                if eligible_event_keys is not None:
                    delay_pair = delay_pair.loc[delay_pair.index.intersection(eligible_event_keys)]
                if len(delay_pair):
                    delay_delta = delay_pair[proposed] - delay_pair[baseline]
                    delay_improved_any = delay_improved_any or float(delay_delta.mean()) < 0
                    h3_component_notes.append(f"{baseline} mean delay-fraction delta={float(delay_delta.mean()):.6g}")
    h3_nab_noninferior = bool(fixed_baselines) and len(h3_positive) == len(fixed_baselines)
    if h3_nab_noninferior and (fpr_improved_all or delay_improved_any):
        h3_outcome = "Potentially supported descriptively; statistical power is limited"
    else:
        h3_outcome = "Not supported overall by the available NAB-style event-score evidence"
    hypothesis_rows.append({
        "hypothesis": "H3: Proposed method maintains or improves NAB score through fewer false alerts or earlier detection",
        "proposal_wording": "Maintain or improve NAB score by fewer false alerts or earlier detection",
        "evaluable_datasets": max((len(x[1][0]) for x in h3_results), default=0),
        "fixed_baselines_with_nonnegative_mean_NAB_difference": ";".join(x[0] for x in h3_positive),
        "support_components": "; ".join(h3_component_notes),
        "outcome": h3_outcome,
        "interpretation": "NAB-style scores are computed over complete events in the held-out test segment; they are not official full-corpus NAB leaderboard scores.",
    })

    # RQ4/domain exploration is kept separate from H3. It is descriptive because
    # event-evaluable counts per domain are small and uneven.
    domain_source = dataset_means.copy()
    if "test_anomaly_events" in domain_source.columns:
        domain_source = domain_source[domain_source["test_anomaly_events"] > 0]
    domain_pivot = domain_source.pivot_table(
        index=index_cols, columns="method", values="f1_score", aggfunc="mean"
    ).reset_index()
    if proposed in domain_pivot.columns and METHOD_ROLLING_MEAN_3SIGMA in domain_pivot.columns:
        domain_pivot["proposed_minus_rolling_mean_f1"] = (
            domain_pivot[proposed] - domain_pivot[METHOD_ROLLING_MEAN_3SIGMA]
        )
        domain_summary = (
            domain_pivot.groupby("domain", as_index=False)
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
    else:
        domain_summary = pd.DataFrame(columns=[
            "domain", "datasets", "mean_f1_improvement", "std_f1_improvement",
            "positive_dataset_count", "improvement_rank",
        ])

    return pd.DataFrame(statistical_rows), pd.DataFrame(hypothesis_rows), domain_summary
