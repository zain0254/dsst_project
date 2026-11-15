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
METHOD_MOVING_MEDIAN_SCALED_MAD = "moving_median_scaled_mad"
METHOD_MOVING_MEDIAN_IQR = "moving_median_iqr_scale"

MAD_NORMAL_CONSISTENCY = 1.4826
NORMAL_IQR_CONSISTENCY = 1.3489795
ROBUST_MAD_METHODS = {
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_MOVING_MEDIAN_SCALED_MAD,
    METHOD_MOVING_MEDIAN_IQR,
}

METHOD_NAMES = {
    METHOD_FIXED_PERCENTILE: "Fixed percentile",
    METHOD_FIXED_MEAN_SIGMA: "Fixed mean + k sigma",
    METHOD_ROLLING_MEAN_3SIGMA: "Rolling mean + k sigma (validation-calibrated)",
    METHOD_MOVING_MEDIAN_MAD: "Moving median + lambda raw MAD (original)",
    METHOD_MOVING_MEDIAN_SCALED_MAD: "Moving median + k scaled MAD (1.4826 x MAD)",
    METHOD_MOVING_MEDIAN_IQR: "Moving median + k IQR scale (IQR / 1.349)",
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


def _robust_scale(history: np.ndarray, calibration_reference: np.ndarray, method: str, floor_fraction: float) -> float:
    median = float(np.median(history))
    raw_mad = float(np.median(np.abs(history - median)))
    if method == METHOD_MOVING_MEDIAN_MAD:
        scale = raw_mad
    elif method == METHOD_MOVING_MEDIAN_SCALED_MAD:
        scale = MAD_NORMAL_CONSISTENCY * raw_mad
    elif method == METHOD_MOVING_MEDIAN_IQR:
        q25, q75 = np.percentile(history, [25, 75])
        scale = float(q75 - q25) / NORMAL_IQR_CONSISTENCY
    else:
        raise ValueError(f"Unsupported robust scale method: {method}")

    # A fixed, normal-only calibration floor avoids zero-scale collapse without
    # allowing a single recent outlier to inflate the local threshold.
    if len(calibration_reference):
        q25_seed, q75_seed = np.percentile(calibration_reference, [25, 75])
        seed_scale = float(q75_seed - q25_seed) / NORMAL_IQR_CONSISTENCY
        if seed_scale <= 1e-12:
            reference_median = float(np.median(calibration_reference))
            reference_mad = float(np.median(np.abs(calibration_reference - reference_median)))
            seed_scale = MAD_NORMAL_CONSISTENCY * reference_mad
    else:
        seed_scale = 0.0
    floor = max(seed_scale * float(floor_fraction), 1e-12)
    return max(float(scale), floor)


def rolling_thresholds(
    errors: np.ndarray,
    seed_history: np.ndarray,
    *,
    lookback: int,
    method: str,
    multiplier: float,
    mad_relative_floor: float = 0.001,
    scale_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling thresholds. Current error is tested before entering history."""
    errors = np.asarray(errors, dtype=float)
    seed_history = np.asarray(seed_history, dtype=float)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if method != METHOD_ROLLING_MEAN_3SIGMA and method not in ROBUST_MAD_METHODS:
        raise ValueError(f"Unsupported rolling method: {method}")
    reference = seed_history if scale_reference is None else np.asarray(scale_reference, dtype=float)
    if not np.isfinite(errors).all() or not np.isfinite(seed_history).all() or not np.isfinite(reference).all():
        raise ValueError("Rolling threshold inputs must be finite")
    if len(seed_history) == 0 or len(reference) == 0:
        raise ValueError("seed_history and scale_reference must not be empty")

    history = deque(seed_history[-lookback:].tolist(), maxlen=lookback)
    thresholds = np.zeros(len(errors), dtype=float)
    predictions = np.zeros(len(errors), dtype=np.int8)

    for index, current_error in enumerate(errors):
        hist = np.asarray(history, dtype=float)
        if method == METHOD_ROLLING_MEAN_3SIGMA:
            threshold = float(np.mean(hist) + multiplier * np.std(hist, ddof=0))
        else:
            median = float(np.median(hist))
            scale = _robust_scale(hist, reference, method, mad_relative_floor)
            threshold = float(median + multiplier * scale)
        thresholds[index] = threshold
        predictions[index] = int(current_error > threshold)
        history.append(float(current_error))
    return predictions, thresholds


def _candidate_quality(
    y_true: np.ndarray, y_pred: np.ndarray, max_validation_fpr: float
) -> tuple[float, float, float, float]:
    f1, fpr, precision = _point_metrics(y_true, y_pred)
    positives = int(np.sum(np.asarray(y_true) == 1))

    # If validation contains no positive labels, F1 cannot distinguish useful
    # thresholds. Choose the least-conservative candidate that stays within
    # the predeclared FPR budget (closest to the budget from below), rather than
    # minimizing FPR and systematically selecting an unnecessarily high cutoff.
    if positives == 0:
        if fpr <= max_validation_fpr:
            return (1.0, -abs(max_validation_fpr - fpr), -fpr, precision)
        return (0.0, -fpr, -abs(max_validation_fpr - fpr), precision)

    # With positive validation labels, maximize F1 subject to the common FPR
    # budget. If no candidate satisfies the budget, minimize FPR first.
    if fpr <= max_validation_fpr:
        return (1.0, f1, -fpr, precision)
    return (0.0, -fpr, f1, precision)


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
    max_validation_fpr: float = 0.05,
    mad_relative_floor: float = 0.001,
    robust_scale_multipliers: list[float] | None = None,
) -> ThresholdResult:
    """Select settings on validation only; never reads test labels."""
    train_errors = np.asarray(train_errors, dtype=float)
    validation_errors = np.asarray(validation_errors, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=np.int8)
    if len(validation_errors) != len(validation_labels):
        raise ValueError("validation_errors and validation_labels lengths do not match")
    if len(train_errors) == 0 or len(validation_errors) == 0:
        raise ValueError("Training and validation errors must be non-empty")

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
            np.full(len(validation_errors), fixed_threshold(train_errors, method, k), dtype=float),
        )
    elif method == METHOD_ROLLING_MEAN_3SIGMA:
        candidates = k_values
        evaluator = lambda k: rolling_thresholds(
            validation_errors, train_errors, lookback=lookback,
            method=method, multiplier=float(k),
        )
    elif method == METHOD_MOVING_MEDIAN_MAD:
        candidates = mad_lambdas
        evaluator = lambda lam: rolling_thresholds(
            validation_errors, train_errors, lookback=lookback,
            method=method, multiplier=float(lam), mad_relative_floor=mad_relative_floor,
            scale_reference=train_errors,
        )
    elif method in {METHOD_MOVING_MEDIAN_SCALED_MAD, METHOD_MOVING_MEDIAN_IQR}:
        candidates = robust_scale_multipliers or k_values
        evaluator = lambda multiplier: rolling_thresholds(
            validation_errors, train_errors, lookback=lookback,
            method=method, multiplier=float(multiplier), mad_relative_floor=mad_relative_floor,
            scale_reference=train_errors,
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    best = None
    for candidate in candidates:
        predictions, thresholds = evaluator(float(candidate))
        quality = _candidate_quality(validation_labels, predictions, max_validation_fpr)
        if best is None or quality > best[0]:
            f1, fpr, _ = _point_metrics(validation_labels, predictions)
            best = (quality, float(candidate), predictions, thresholds, f1, fpr)

    if best is None:
        raise ValueError(f"No threshold candidates configured for method {method}")
    _, parameter, predictions, thresholds, val_f1, val_fpr = best
    return ThresholdResult(predictions, thresholds, parameter, val_f1, val_fpr)


def apply_threshold_method(
    method: str,
    parameter: float,
    train_errors: np.ndarray,
    validation_errors: np.ndarray,
    test_errors: np.ndarray,
    *,
    lookback: int,
    mad_relative_floor: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply validation-selected settings to the untouched test partition."""
    test_errors = np.asarray(test_errors, dtype=float)
    if method in {METHOD_FIXED_PERCENTILE, METHOD_FIXED_MEAN_SIGMA}:
        threshold = fixed_threshold(train_errors, method, parameter)
        thresholds = np.full(len(test_errors), threshold, dtype=float)
        return (test_errors > threshold).astype(np.int8), thresholds

    if method in {METHOD_ROLLING_MEAN_3SIGMA, *ROBUST_MAD_METHODS}:
        seed = np.concatenate([np.asarray(train_errors, dtype=float), np.asarray(validation_errors, dtype=float)])
        return rolling_thresholds(
            test_errors, seed, lookback=lookback, method=method,
            multiplier=parameter, mad_relative_floor=mad_relative_floor,
            scale_reference=np.asarray(train_errors, dtype=float),
        )

    raise ValueError(f"Unknown threshold method: {method}")
