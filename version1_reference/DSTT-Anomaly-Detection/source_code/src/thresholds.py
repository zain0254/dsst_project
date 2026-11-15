from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


METHOD_FIXED_PERCENTILE = "fixed_percentile"
METHOD_FIXED_MEAN_SIGMA = "fixed_mean_k_sigma"
METHOD_ROLLING_MEAN_3SIGMA = "rolling_mean_3sigma"
METHOD_MOVING_MEDIAN_MAD = "moving_median_lambda_mad"

METHOD_NAMES = {
    METHOD_FIXED_PERCENTILE: "Fixed percentile",
    METHOD_FIXED_MEAN_SIGMA: "Fixed mean + k sigma",
    METHOD_ROLLING_MEAN_3SIGMA: "Rolling mean + 3 sigma (proposal baseline)",
    METHOD_MOVING_MEDIAN_MAD: "Moving median + lambda MAD (proposed)",
}


@dataclass
class ThresholdResult:
    predictions: np.ndarray
    thresholds: np.ndarray
    parameter: float | str
    validation_f1: float
    validation_fpr: float


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    negatives = np.asarray(y_true) == 0
    fpr = float(np.sum((np.asarray(y_pred) == 1) & negatives) / max(int(negatives.sum()), 1))
    return f1, fpr, precision


def fixed_threshold(train_errors: np.ndarray, method: str, parameter: float) -> float:
    train_errors = np.asarray(train_errors, dtype=float)
    if len(train_errors) == 0:
        raise ValueError("Cannot calculate a fixed threshold from an empty training error array.")
    if method == METHOD_FIXED_PERCENTILE:
        return float(np.percentile(train_errors, parameter))
    if method == METHOD_FIXED_MEAN_SIGMA:
        return float(np.mean(train_errors) + parameter * np.std(train_errors, ddof=0))
    raise ValueError(f"Unsupported fixed-threshold method: {method}")


def rolling_thresholds(
    errors: np.ndarray,
    seed_history: np.ndarray,
    *,
    lookback: int,
    method: str,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling thresholds. The current error is tested before it enters history."""
    errors = np.asarray(errors, dtype=float)
    seed_history = np.asarray(seed_history, dtype=float)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if method not in {METHOD_ROLLING_MEAN_3SIGMA, METHOD_MOVING_MEDIAN_MAD}:
        raise ValueError(f"Unsupported rolling method: {method}")

    history = deque(seed_history[-lookback:].tolist(), maxlen=lookback)
    thresholds = np.zeros(len(errors), dtype=float)
    predictions = np.zeros(len(errors), dtype=np.int8)

    for index, current_error in enumerate(errors):
        hist = np.asarray(history, dtype=float)
        if len(hist) == 0:
            threshold = float(current_error)
        elif method == METHOD_ROLLING_MEAN_3SIGMA:
            threshold = float(np.mean(hist) + 3.0 * np.std(hist, ddof=0))
        else:
            median = float(np.median(hist))
            mad = float(np.median(np.abs(hist - median)))
            threshold = float(median + multiplier * mad)
        thresholds[index] = threshold
        predictions[index] = int(current_error > threshold)
        history.append(float(current_error))
    return predictions, thresholds


def _candidate_quality(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    f1, fpr, precision = _point_metrics(y_true, y_pred)
    # Higher F1 first, then lower FPR, then higher precision. Stable deterministic tie-break.
    return (f1, -fpr, precision)


def tune_threshold_method(
    method: str,
    train_errors: np.ndarray,
    validation_errors: np.ndarray,
    validation_labels: np.ndarray,
    *,
    lookback: int,
    percentiles: list[float],
    k_values: list[float],
    mad_lambdas: list[float],
) -> ThresholdResult:
    """Select a threshold parameter using validation labels only."""
    validation_errors = np.asarray(validation_errors, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=np.int8)

    if method == METHOD_FIXED_PERCENTILE:
        candidates = percentiles
        evaluator: Callable[[float], tuple[np.ndarray, np.ndarray]] = lambda q: (
            (validation_errors > fixed_threshold(train_errors, method, q)).astype(np.int8),
            np.full(len(validation_errors), fixed_threshold(train_errors, method, q), dtype=float),
        )
    elif method == METHOD_FIXED_MEAN_SIGMA:
        candidates = k_values
        evaluator = lambda k: (
            (validation_errors > fixed_threshold(train_errors, method, k)).astype(np.int8),
            np.full(
                len(validation_errors),
                fixed_threshold(train_errors, method, k),
                dtype=float,
            ),
        )
    elif method == METHOD_ROLLING_MEAN_3SIGMA:
        candidates = [3.0]
        evaluator = lambda _unused: rolling_thresholds(
            validation_errors,
            train_errors,
            lookback=lookback,
            method=method,
            multiplier=3.0,
        )
    elif method == METHOD_MOVING_MEDIAN_MAD:
        candidates = mad_lambdas
        evaluator = lambda lam: rolling_thresholds(
            validation_errors,
            train_errors,
            lookback=lookback,
            method=method,
            multiplier=lam,
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    best = None
    for candidate in candidates:
        predictions, thresholds = evaluator(float(candidate))
        quality = _candidate_quality(validation_labels, predictions)
        if best is None or quality > best[0]:
            f1, fpr, _ = _point_metrics(validation_labels, predictions)
            best = (quality, float(candidate), predictions, thresholds, f1, fpr)

    assert best is not None
    _, parameter, predictions, thresholds, val_f1, val_fpr = best
    return ThresholdResult(
        predictions=predictions,
        thresholds=thresholds,
        parameter=parameter,
        validation_f1=val_f1,
        validation_fpr=val_fpr,
    )


def apply_threshold_method(
    method: str,
    parameter: float,
    train_errors: np.ndarray,
    validation_errors: np.ndarray,
    test_errors: np.ndarray,
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the validation-selected setting to the untouched test partition."""
    test_errors = np.asarray(test_errors, dtype=float)
    if method in {METHOD_FIXED_PERCENTILE, METHOD_FIXED_MEAN_SIGMA}:
        threshold = fixed_threshold(train_errors, method, parameter)
        thresholds = np.full(len(test_errors), threshold, dtype=float)
        return (test_errors > threshold).astype(np.int8), thresholds

    if method == METHOD_ROLLING_MEAN_3SIGMA:
        seed = np.concatenate([train_errors, validation_errors])
        return rolling_thresholds(
            test_errors,
            seed,
            lookback=lookback,
            method=method,
            multiplier=3.0,
        )

    if method == METHOD_MOVING_MEDIAN_MAD:
        seed = np.concatenate([train_errors, validation_errors])
        return rolling_thresholds(
            test_errors,
            seed,
            lookback=lookback,
            method=method,
            multiplier=parameter,
        )

    raise ValueError(f"Unknown threshold method: {method}")
