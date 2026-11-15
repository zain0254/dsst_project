from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler


def download_nab_assets(
    data_root: Path,
    selected_files: dict[str, str],
    raw_base_url: str,
    labels_file: str,
    force: bool = False,
) -> Path:
    """Download selected NAB CSV files and the combined label-window file."""
    data_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for relative_path in selected_files:
        destination = data_root / "data" / relative_path
        if destination.exists() and destination.stat().st_size > 0 and not force:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{raw_base_url.rstrip('/')}/data/{relative_path}"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        print(f"Downloaded: data/{relative_path}")

    labels_destination = data_root / labels_file
    if not labels_destination.exists() or labels_destination.stat().st_size == 0 or force:
        labels_destination.parent.mkdir(parents=True, exist_ok=True)
        labels_url = f"{raw_base_url.rstrip('/')}/{labels_file}"
        response = session.get(labels_url, timeout=60)
        response.raise_for_status()
        labels_destination.write_bytes(response.content)
        print(f"Downloaded: {labels_file}")
    return labels_destination


def load_label_windows(labels_path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    with labels_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for dataset_path, windows in raw.items():
        parsed = []
        for start, end in windows:
            parsed.append((pd.Timestamp(start), pd.Timestamp(end)))
        result[dataset_path] = sorted(parsed, key=lambda pair: pair[0])
    return result


def load_nab_series(csv_path: Path) -> pd.DataFrame:
    """Load and clean one univariate NAB series using causal forward filling.

    Leading missing values are dropped instead of backward-filled so that a
    future observation is never copied into an earlier timestamp.
    """
    frame = pd.read_csv(csv_path)
    required = {"timestamp", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    frame = frame[["timestamp", "value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan).ffill()
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError(f"Not enough valid observations in {csv_path}: {len(frame)}")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values remain after cleaning: {csv_path}")
    return frame


def make_point_labels(
    timestamps: pd.Series | pd.DatetimeIndex,
    windows: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    """Convert NAB's timestamp intervals to a point-wise binary label mask."""
    times = pd.DatetimeIndex(timestamps)
    labels = np.zeros(len(times), dtype=np.int8)
    for start, end in windows:
        labels |= ((times >= start) & (times <= end)).astype(np.int8)
    return labels


def chronological_split(
    n_rows: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[int, int]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Train and validation fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than 1.")
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    if train_end < 3 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError(f"Series is too short for 60/20/20 split: n={n_rows}")
    return train_end, validation_end


def make_sequences(values: np.ndarray, end_indices: np.ndarray, window_size: int) -> np.ndarray:
    """Create trailing windows ending at the requested rows (no future values)."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    valid_ends = np.asarray(end_indices, dtype=np.int64)
    valid_ends = valid_ends[valid_ends >= window_size - 1]
    if len(valid_ends) == 0:
        return np.empty((0, window_size, 1), dtype=np.float32)
    return np.stack(
        [values[end - window_size + 1 : end + 1] for end in valid_ends],
        axis=0,
    ).astype(np.float32)


def prepare_splits(
    frame: pd.DataFrame,
    labels: np.ndarray,
    window_size: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    normal_only_training: bool = False,
) -> dict:
    """Chronological split; optionally retain only fully normal training windows."""
    n_rows = len(frame)
    if len(labels) != n_rows:
        raise ValueError("labels length must equal the number of input rows")
    train_end, validation_end = chronological_split(
        n_rows, train_fraction=train_fraction, validation_fraction=validation_fraction
    )

    scaler = MinMaxScaler()
    if normal_only_training:
        normal_train_rows = np.flatnonzero(np.asarray(labels[:train_end]) == 0)
        if len(normal_train_rows) < 2:
            raise ValueError("Normal-only training requires at least two normal training rows.")
        scaler.fit(frame.loc[normal_train_rows, ["value"]])
    else:
        scaler.fit(frame.loc[: train_end - 1, ["value"]])
    scaled = scaler.transform(frame[["value"]]).astype(np.float32).reshape(-1)

    train_ends_all = np.arange(window_size - 1, train_end, dtype=np.int64)
    train_ends = train_ends_all.copy()
    if normal_only_training:
        # A training window is clean only if every point in its receptive field is normal.
        prefix = np.concatenate([[0], np.cumsum(np.asarray(labels[:train_end], dtype=np.int64))])
        starts = train_ends_all - window_size + 1
        anomaly_counts = prefix[train_ends_all + 1] - prefix[starts]
        train_ends = train_ends_all[anomaly_counts == 0]
    validation_ends = np.arange(train_end, validation_end, dtype=np.int64)
    test_ends = np.arange(validation_end, n_rows, dtype=np.int64)

    x_train = make_sequences(scaled[:train_end], train_ends, window_size)
    # Validation/test windows use earlier rows only as causal historical context.
    x_validation = make_sequences(scaled, validation_ends, window_size)
    validation_normal_ends = validation_ends.copy()
    if normal_only_training:
        prefix_all = np.concatenate([[0], np.cumsum(np.asarray(labels, dtype=np.int64))])
        validation_starts = validation_ends - window_size + 1
        validation_anomaly_counts = prefix_all[validation_ends + 1] - prefix_all[validation_starts]
        validation_normal_ends = validation_ends[validation_anomaly_counts == 0]
        if len(validation_normal_ends) < 2:
            raise ValueError("Normal-only early stopping requires at least two fully normal validation windows.")
    x_validation_normal = make_sequences(scaled, validation_normal_ends, window_size)
    x_test = make_sequences(scaled, test_ends, window_size)

    return {
        "scaler": scaler,
        "scaled_values": scaled,
        "train_end": train_end,
        "validation_end": validation_end,
        "train_ends": train_ends,
        "train_windows_before_filter": int(len(train_ends_all)),
        "train_windows_excluded_anomalous": int(len(train_ends_all) - len(train_ends)),
        "normal_only_training": bool(normal_only_training),
        "validation_ends": validation_ends,
        "validation_normal_ends": validation_normal_ends,
        "validation_windows_normal": int(len(validation_normal_ends)),
        "test_ends": test_ends,
        "x_train": x_train,
        "x_validation": x_validation,
        "x_validation_normal": x_validation_normal,
        "x_test": x_test,
        "y_train": labels[train_ends],
        "y_validation": labels[validation_ends],
        "y_test": labels[test_ends],
        "timestamps_train": frame.loc[train_ends, "timestamp"].reset_index(drop=True),
        "timestamps_validation": frame.loc[validation_ends, "timestamp"].reset_index(drop=True),
        "timestamps_test": frame.loc[test_ends, "timestamp"].reset_index(drop=True),
    }
