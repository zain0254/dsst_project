from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

"""NAB-compatible timing score for the held-out test portion of a time series.

The timing-reward and false-positive weighting below follow the scoring
mechanics in Numenta/NAB's MIT-licensed ``nab/sweeper.py`` (Copyright
2014-2015 Numenta Inc.). This project applies those mechanics to complete
anomaly windows contained in its held-out test partition, then normalizes
against null/perfect scores for that test subset. It is therefore a
*NAB-style test score*, not the official corpus-wide NAB leaderboard score.

Official source: https://github.com/numenta/NAB/blob/master/nab/sweeper.py
"""


# Numenta NAB Standard application-profile cost matrix.
STANDARD_TP_WEIGHT = 1.0
STANDARD_FN_WEIGHT = 1.0
STANDARD_FP_WEIGHT = 0.11


def _scaled_sigmoid(relative_position: float) -> float:
    """NAB's scaled sigmoid timing function for TP/FP rewards."""
    if relative_position > 3.0:
        return -1.0
    z = -5.0 * relative_position
    if z >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        sigmoid = exp_z / (1.0 + exp_z)
    return 2.0 * sigmoid - 1.0


def complete_test_events(
    timestamps: Iterable,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return only complete labeled events fully represented in test timestamps."""
    times = pd.DatetimeIndex(timestamps)
    if len(times) == 0:
        return []
    start_time, end_time = times[0], times[-1]
    available = set(times)
    complete = []
    for left, right in windows:
        left, right = pd.Timestamp(left), pd.Timestamp(right)
        if (
            start_time <= left <= right <= end_time
            and left in available
            and right in available
        ):
            complete.append((left, right))
    return sorted(complete, key=lambda item: item[0])


def nab_style_test_score(
    timestamps: Iterable,
    predictions: np.ndarray,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    probationary_percent: float = 0.0,
) -> dict[str, float]:
    """Score predictions with NAB's timing rewards on a held-out test subset.

    The official NAB runner uses a 15% probationary period at the beginning
    of each complete data file. Here the model has already consumed the
    chronological train and validation portions before test-time evaluation,
    so the default additional test probation is 0%. Set it above zero only
    when the scored timestamps begin at the start of a stream without prior
    warm-up history.

    The test threshold itself is not optimized here; ``predictions`` must be
    produced by the parameter selected on the validation partition.
    """
    times = pd.DatetimeIndex(timestamps)
    pred = np.asarray(predictions, dtype=np.int8)
    if len(times) != len(pred):
        raise ValueError("timestamps and predictions must have equal length")
    if not np.isin(pred, [0, 1]).all():
        raise ValueError("predictions must contain only 0/1 values")
    if not 0.0 <= probationary_percent < 1.0:
        raise ValueError("probationary_percent must be in [0, 1)")
    if len(times) == 0:
        return {
            "nab_style_score": math.nan,
            "nab_raw_score": math.nan,
            "nab_events_scored": 0,
            "nab_false_positive_penalty": math.nan,
        }

    events = complete_test_events(times, windows)
    if not events:
        return {
            "nab_style_score": math.nan,
            "nab_raw_score": math.nan,
            "nab_events_scored": 0,
            "nab_false_positive_penalty": math.nan,
        }

    # Map each timestamp to one event. NAB benchmark windows do not overlap;
    # fail explicitly rather than silently mis-scoring overlapping labels.
    event_at_index: dict[int, int] = {}
    event_bounds: list[tuple[int, int]] = []
    timestamp_to_index = {timestamp: i for i, timestamp in enumerate(times)}
    for event_id, (left, right) in enumerate(events):
        start_i = timestamp_to_index[left]
        end_i = timestamp_to_index[right]
        event_bounds.append((start_i, end_i))
        for index in range(start_i, end_i + 1):
            if index in event_at_index:
                raise ValueError("Overlapping NAB anomaly windows are not supported.")
            event_at_index[index] = event_id

    probationary_length = min(
        int(math.floor(probationary_percent * len(times))),
        int(probationary_percent * 5000),
    )
    scored_indices = range(probationary_length, len(times))

    # NAB initializes every scorable event as a false negative. A detected
    # point inside an event replaces that event's penalty with its best timing
    # reward; multiple detections in one event do not earn multiple TPs.
    event_rewards = {
        event_id: -STANDARD_FN_WEIGHT
        for event_id, (left, right) in enumerate(event_bounds)
        if any(index >= probationary_length for index in range(left, right + 1))
    }

    max_tp_reward = _scaled_sigmoid(-1.0)
    event_sweep_reward: dict[int, dict[int, float]] = {}
    false_positive_penalty = 0.0
    previous_event_right: int | None = None
    previous_event_width: int | None = None
    event_cursor = 0

    for index in scored_indices:
        # Advance the event cursor once the prior event has ended.
        while event_cursor < len(event_bounds) and index > event_bounds[event_cursor][1]:
            previous_event_right = event_bounds[event_cursor][1]
            previous_event_width = event_bounds[event_cursor][1] - event_bounds[event_cursor][0] + 1
            event_cursor += 1

        current_event_id = event_at_index.get(index)
        if current_event_id is not None:
            left, right = event_bounds[current_event_id]
            width = float(right - left + 1)
            position_in_window = -(right - index + 1) / width
            sweep_reward = _scaled_sigmoid(position_in_window) * (
                STANDARD_TP_WEIGHT / max_tp_reward
            )
            event_sweep_reward.setdefault(current_event_id, {})[index] = sweep_reward
            if pred[index] == 1 and current_event_id in event_rewards:
                event_rewards[current_event_id] = max(
                    event_rewards[current_event_id], sweep_reward
                )
        elif pred[index] == 1:
            if previous_event_right is None or previous_event_width is None:
                # No preceding anomaly window: NAB treats this as far away.
                relative_position = 4.0
            else:
                denominator = float(previous_event_width - 1)
                if denominator <= 0.0:
                    relative_position = 4.0
                else:
                    relative_position = abs(previous_event_right - index) / denominator
            false_positive_penalty += (
                _scaled_sigmoid(relative_position) * STANDARD_FP_WEIGHT
            )

    scored_event_count = len(event_rewards)
    if scored_event_count == 0:
        return {
            "nab_style_score": math.nan,
            "nab_raw_score": math.nan,
            "nab_events_scored": 0,
            "nab_false_positive_penalty": float(false_positive_penalty),
        }

    raw_score = float(sum(event_rewards.values()) + false_positive_penalty)
    null_score = -scored_event_count * STANDARD_FN_WEIGHT
    perfect_score = scored_event_count * STANDARD_TP_WEIGHT
    denominator = perfect_score - null_score
    normalized = 100.0 * (raw_score - null_score) / denominator

    return {
        "nab_style_score": float(normalized),
        "nab_raw_score": raw_score,
        "nab_events_scored": int(scored_event_count),
        "nab_false_positive_penalty": float(false_positive_penalty),
    }
