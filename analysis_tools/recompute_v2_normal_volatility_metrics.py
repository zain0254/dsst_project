#!/usr/bin/env python3
"""Recompute V2 metrics from saved predictions using clean-normal volatility calibration.

This is a post-hoc metric reanalysis. It does not retrain models or change model
scores/predictions; it updates only the high-variance mask and its dependent metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_code"
sys.path.insert(0, str(SOURCE))

from run_experiment import METHOD_ORDER, _complete_test_windows, _plot_aggregates, _write_summaries
from src.data_utils import chronological_split, load_label_windows, load_nab_series, make_point_labels
from src.metrics import calculate_metrics, high_variance_mask
from src.benchmark_reporting import write_benchmark_outputs


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=SOURCE / "outputs" / "version2_normal_only_fpr05")
    parser.add_argument(
        "--output-dir", type=Path,
        default=SOURCE / "outputs" / "version2_normal_only_fpr05_normal_volatility_reaudited_20261115",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (input_dir / "config_used.json").exists():
        raise SystemExit(f"Missing config_used.json under {input_dir}")
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    # Keep this reanalysis compact: original prediction traces remain in the
    # source run directory and are read-only inputs, not duplicated outputs.
    for filename in [
        "config_used.json", "environment.json", "dataset_split_audit.csv",
        "validation_tuning.csv", "run_summary.json",
    ]:
        shutil.copy2(input_dir / filename, output_dir / filename)
    config = json.loads((input_dir / "config_used.json").read_text(encoding="utf-8"))
    label_windows = load_label_windows(SOURCE / "data" / "nab" / config["labels_file"])
    metrics = pd.read_csv(input_dir / "test_metrics.csv")
    original_metrics = metrics.copy(deep=True)
    prediction_dir = input_dir / "predictions"
    audit_rows = []

    for relative_path, domain in config["selected_files"].items():
        frame = load_nab_series(SOURCE / "data" / "nab" / "data" / relative_path)
        labels = make_point_labels(frame["timestamp"], label_windows[relative_path])
        train_end, validation_end = chronological_split(
            len(frame),
            train_fraction=config["split"]["train_fraction"],
            validation_fraction=config["split"]["validation_fraction"],
        )
        test_indices = np.arange(validation_end, len(frame), dtype=np.int64)
        test_timestamps = frame.loc[test_indices, "timestamp"].reset_index(drop=True)
        y_test = labels[test_indices]
        test_windows, partial_excluded = _complete_test_windows(
            label_windows[relative_path], test_timestamps
        )
        values = frame["value"].to_numpy(dtype=float)

        for window_size in config["window_sizes"]:
            high_variance = high_variance_mask(
                values,
                train_end=train_end,
                test_indices=test_indices,
                lookback=int(window_size),
                percentile=float(config.get("high_variance_percentile", 75.0)),
                training_labels=labels,
            )
            for seed in config["run_seeds"]:
                prediction_path = prediction_dir / (
                    f"{Path(relative_path).stem}_W{int(window_size)}_seed{int(seed)}.csv"
                )
                if not prediction_path.exists():
                    raise FileNotFoundError(prediction_path)
                prediction_frame = pd.read_csv(prediction_path)
                if len(prediction_frame) != len(y_test):
                    raise ValueError(f"Prediction row mismatch: {prediction_path}")
                errors = prediction_frame["reconstruction_error"].to_numpy(dtype=float)

                for method in METHOD_ORDER:
                    slug = method.replace(" ", "_")
                    prediction = prediction_frame[f"{slug}_prediction"].to_numpy(dtype=np.int8)
                    recalculated = calculate_metrics(
                        y_test,
                        prediction,
                        test_timestamps=test_timestamps,
                        test_windows=test_windows,
                        high_variance_points=high_variance,
                        anomaly_scores=errors,
                    )
                    row_mask = (
                        (metrics["dataset"] == relative_path)
                        & (metrics["window_size"] == int(window_size))
                        & (metrics["seed"] == int(seed))
                        & (metrics["method"] == method)
                    )
                    row_indices = metrics.index[row_mask]
                    if len(row_indices) != 1:
                        raise ValueError(
                            f"Expected exactly one metrics row for {relative_path}, W={window_size}, seed={seed}, {method}; got {len(row_indices)}"
                        )
                    row_index = row_indices[0]
                    for column, value in recalculated.items():
                        metrics.loc[row_index, column] = value
                    audit_rows.append({
                        "dataset": relative_path,
                        "domain": domain,
                        "window_size": int(window_size),
                        "seed": int(seed),
                        "method": method,
                        "high_variance_normal_points": recalculated["high_variance_normal_points"],
                        "high_variance_fpr": recalculated["high_variance_fpr"],
                        "complete_test_events": len(test_windows),
                        "partial_test_events_excluded": partial_excluded,
                    })

    metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "normal_volatility_reanalysis.csv", index=False)

    # Every metric other than the high-variance subset metrics must reproduce from
    # the same saved predictions and labels to numerical precision.
    excluded = {"high_variance_fpr", "high_variance_normal_points"}
    compare_columns = [
        c for c in metrics.columns
        if c in original_metrics.columns and c not in excluded
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]
    differences = {}
    for column in compare_columns:
        a = pd.to_numeric(original_metrics[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if np.any(np.isfinite(a) ^ np.isfinite(b)):
            raise AssertionError(f"NaN coverage changed unexpectedly for {column}")
        differences[column] = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        if differences[column] > 1e-10:
            raise AssertionError(f"Unexpected change in {column}: {differences[column]}")

    tests, outcomes, _ = _write_summaries(metrics, output_dir)
    write_benchmark_outputs(metrics, output_dir, outcomes)
    _plot_aggregates(metrics, output_dir)

    run_summary_path = output_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary["high_variance_reference"] = (
        "75th percentile of full trailing-window standard deviations from fully normal training histories only"
    )
    prediction_source_relative = os.path.relpath(input_dir / "predictions", output_dir)
    run_summary["posthoc_metric_reanalysis"] = {
        "performed_without_model_retraining": True,
        "changed_metrics": ["high_variance_fpr", "high_variance_normal_points"],
        "model_fits_preserved": run_summary["models_trained"],
        "evaluation_rows_preserved": run_summary["metric_rows"],
        "prediction_traces_reused_from": prediction_source_relative,
        "prediction_traces_duplicated": False,
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")

    try:
        input_reference = input_dir.relative_to(SOURCE).as_posix()
    except ValueError:
        input_reference = str(input_dir)
    try:
        output_reference = output_dir.relative_to(SOURCE).as_posix()
    except ValueError:
        output_reference = str(output_dir)
    report = {
        "input_directory": input_reference,
        "output_directory": output_reference,
        "model_retraining_performed": False,
        "model_fits": int(metrics[["dataset", "window_size", "seed"]].drop_duplicates().shape[0]),
        "metric_rows": int(len(metrics)),
        "prediction_files_read_from_source": len(list(prediction_dir.glob("*.csv"))),
        "prediction_files_duplicated": False,
        "prediction_source_directory": input_reference.rstrip("/") + "/predictions/",
        "recalculated_rows": int(len(audit)),
        "changed_metrics": ["high_variance_fpr", "high_variance_normal_points"],
        "max_abs_difference_unchanged_metrics": differences,
        "high_variance_reference": run_summary["high_variance_reference"],
        "high_variance_fpr_mean_by_method_macro_dataset": (
            metrics.groupby(["dataset", "domain", "method"], as_index=False)["high_variance_fpr"]
            .mean().groupby("method")["high_variance_fpr"].mean().to_dict()
        ),
        "hypothesis_outcomes": outcomes.to_dict(orient="records"),
        "statistical_test_rows": int(len(tests)),
    }
    report = _json_safe(report)
    (output_dir / "normal_volatility_reanalysis_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "REANALYSIS_README.md").write_text(
        "# Clean-Normal Volatility Reanalysis\n\n"
        "This directory contains updated metric tables and reports for the H2 high-variance FPR analysis. "
        f"The original model prediction traces were not copied or modified; they are read from `{prediction_source_relative}/`. "
        "No model was retrained. Only `high_variance_fpr` and `high_variance_normal_points` changed; all other metrics were independently reproduced from the saved prediction traces.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
