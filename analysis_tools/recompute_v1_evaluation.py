#!/usr/bin/env python3
"""Recompute V1 evaluation metrics from the archived raw prediction CSVs.

Run from the extracted submission package root:
    python analysis_tools/recompute_v1_evaluation.py

The script never edits raw benchmark output files. It writes derived evaluation
artifacts into evaluation/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "source_code"
RUN = ROOT / "verified_results" / "V1_full_gpu_benchmark_restart"
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT))
from src.nab_scoring import complete_test_events  # noqa: E402

METHOD_SLUG = {
    "fixed_percentile": "fixed_percentile",
    "fixed_mean_k_sigma": "fixed_mean_k_sigma",
    "rolling_mean_3sigma": "rolling_mean_3sigma",
    "moving_median_lambda_mad": "moving_median_lambda_mad",
}


def main() -> None:
    metrics_path = RUN / "test_metrics.csv"
    predictions_dir = RUN / "predictions"
    labels_path = PROJECT / "data" / "nab" / "labels" / "combined_windows.json"
    for path in (metrics_path, predictions_dir, labels_path):
        if not path.exists():
            raise FileNotFoundError(path)

    original = pd.read_csv(metrics_path)
    with labels_path.open("r", encoding="utf-8") as stream:
        label_windows_raw = json.load(stream)

    derived_rows: list[dict] = []
    missing_predictions: list[str] = []
    for (dataset, window_size, seed), group in original.groupby(["dataset", "window_size", "seed"]):
        prediction_file = predictions_dir / f"{Path(dataset).stem}_W{window_size}_seed{seed}.csv"
        if not prediction_file.exists():
            missing_predictions.append(str(prediction_file.relative_to(ROOT)))
            continue
        pred = pd.read_csv(prediction_file)
        y_true = pred["actual_label"].to_numpy(dtype=np.int8)
        scores = pred["reconstruction_error"].to_numpy(dtype=float)
        timestamps = pd.DatetimeIndex(pd.to_datetime(pred["timestamp"]))
        events = [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in label_windows_raw.get(dataset, [])
        ]
        complete_events = complete_test_events(timestamps, events)
        has_both_classes = len(np.unique(y_true)) == 2

        for _, row in group.iterrows():
            slug = METHOD_SLUG[row["method"]]
            y_pred = pred[f"{slug}_prediction"].to_numpy(dtype=np.int8)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            delays_seconds: list[float] = []
            for start, end in complete_events:
                event_indices = np.flatnonzero((timestamps >= start) & (timestamps <= end))
                detections = event_indices[y_pred[event_indices] == 1]
                if len(detections):
                    first_time = timestamps[int(detections[0])]
                    delays_seconds.append(max(0.0, (first_time - start).total_seconds()))

            record = row.to_dict()
            record.update({
                "accuracy_recomputed": float(accuracy_score(y_true, y_pred)),
                "roc_auc_recomputed": float(roc_auc_score(y_true, scores)) if has_both_classes else np.nan,
                "pr_auc_recomputed": float(average_precision_score(y_true, scores)) if has_both_classes else np.nan,
                "f1_recomputed": float(f1_score(y_true, y_pred, zero_division=0)),
                "fpr_recomputed": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                "mean_detection_delay_seconds_recomputed": float(np.mean(delays_seconds)) if delays_seconds else np.nan,
                "median_detection_delay_seconds_recomputed": float(np.median(delays_seconds)) if delays_seconds else np.nan,
                "detected_event_count_recomputed": int(len(delays_seconds)),
                "complete_event_count_recomputed": int(len(complete_events)),
            })
            derived_rows.append(record)

    augmented = pd.DataFrame(derived_rows)
    augmented.to_csv(OUT / "test_metrics_augmented.csv", index=False)

    # Integrity audit: compare recomputed values with the original raw result table.
    checks: list[dict] = []
    checks.append({"check": "Original metric rows", "observed": len(original), "expected": 360, "status": "PASS" if len(original) == 360 else "FAIL"})
    run_count = original[["dataset", "window_size", "seed"]].drop_duplicates().shape[0]
    checks.append({"check": "Unique model runs", "observed": run_count, "expected": 90, "status": "PASS" if run_count == 90 else "FAIL"})
    checks.append({"check": "Prediction files matched", "observed": len(augmented) == len(original), "expected": True, "status": "PASS" if len(augmented) == len(original) else "FAIL"})
    checks.append({"check": "Missing prediction files", "observed": len(missing_predictions), "expected": 0, "status": "PASS" if not missing_predictions else "FAIL"})
    checks.append({"check": "All V1 metric rows recorded GPU device", "observed": sorted(original["device"].dropna().unique().tolist()), "expected": ["GPU:0"], "status": "PASS" if set(original["device"].dropna().unique()) == {"GPU:0"} else "FAIL"})
    for original_col, derived_col, check_name in [
        ("f1_score", "f1_recomputed", "F1 reproduces from prediction CSVs"),
        ("false_positive_rate", "fpr_recomputed", "FPR reproduces from prediction CSVs"),
        ("pr_auc", "pr_auc_recomputed", "AP/PR-AUC reproduces from raw error scores"),
    ]:
        differences = (augmented[original_col] - augmented[derived_col]).abs()
        maximum = float(differences.max(skipna=True))
        checks.append({"check": check_name, "observed_max_abs_difference": maximum, "tolerance": 1e-9, "status": "PASS" if maximum <= 1e-9 else "FAIL"})

    dataset_context = (
        original.groupby(["dataset", "domain"], as_index=False)
        .agg(
            rows_in_series=("test_points", "first"),
            test_points=("test_points", "first"),
            test_positive_points=("test_positive_points", "first"),
            complete_test_events=("test_anomaly_events", "first"),
            partial_test_events_excluded=("partial_events_excluded_from_event_scoring", "first"),
            train_anomaly_points=("train_anomaly_points", "first") if "train_anomaly_points" in original.columns else ("test_points", "first"),
            validation_anomaly_points=("validation_anomaly_points", "first") if "validation_anomaly_points" in original.columns else ("test_points", "first"),
            mean_epochs_run=("epochs_run", "mean"),
        )
    )
    # Use the split audit (rather than the metric table) for train/validation label counts.
    split_audit = pd.read_csv(RUN / "dataset_split_audit.csv")
    split_context = split_audit.groupby(["dataset", "domain"], as_index=False).agg(
        train_labeled_anomaly_points=("train_anomaly_points", "max"),
        validation_labeled_anomaly_points=("validation_anomaly_points", "max"),
        test_labeled_anomaly_points=("test_anomaly_points", "max"),
        complete_test_events=("test_complete_anomaly_events", "max"),
        partial_test_events_excluded=("test_partial_anomaly_events_excluded", "max"),
        series_rows=("rows", "first"),
    )
    coverage = split_context.copy()
    coverage["has_train_anomaly_labels"] = coverage["train_labeled_anomaly_points"] > 0
    coverage["has_validation_anomaly_labels"] = coverage["validation_labeled_anomaly_points"] > 0
    coverage["has_test_anomaly_labels"] = coverage["test_labeled_anomaly_points"] > 0
    coverage.to_csv(OUT / "dataset_coverage.csv", index=False)

    domain_coverage = coverage.groupby("domain", as_index=False).agg(
        datasets=("dataset", "nunique"),
        files_with_train_anomaly_labels=("has_train_anomaly_labels", "sum"),
        files_with_validation_anomaly_labels=("has_validation_anomaly_labels", "sum"),
        files_with_test_labels=("has_test_anomaly_labels", "sum"),
        files_with_complete_test_events=("complete_test_events", lambda x: int((x > 0).sum())),
        total_test_positive_points=("test_labeled_anomaly_points", "sum"),
        total_complete_test_events=("complete_test_events", "sum"),
    )
    domain_coverage.to_csv(OUT / "domain_coverage.csv", index=False)

    numeric_metrics = [
        "precision", "recall", "f1_score", "accuracy_recomputed", "false_positive_rate",
        "roc_auc_recomputed", "pr_auc", "high_variance_fpr", "event_recall",
        "mean_detection_delay_fraction", "mean_detection_delay_seconds_recomputed", "nab_style_score",
    ]
    long_dataset_means = augmented.groupby(["dataset", "domain", "method", "method_name"], as_index=False)[numeric_metrics].mean(numeric_only=True)
    long_dataset_means = long_dataset_means.merge(coverage, on=["dataset", "domain"], how="left")
    long_dataset_means.to_csv(OUT / "dataset_method_means.csv", index=False)

    wide = long_dataset_means.pivot(index=["dataset", "domain"], columns="method", values=[
        "f1_score", "accuracy_recomputed", "false_positive_rate", "roc_auc_recomputed", "pr_auc",
        "nab_style_score", "event_recall", "mean_detection_delay_fraction",
        "mean_detection_delay_seconds_recomputed", "high_variance_fpr",
    ])
    wide.columns = ["__".join(str(part) for part in column) for column in wide.columns]
    wide = wide.reset_index().merge(coverage, on=["dataset", "domain"], how="left")
    wide.to_csv(OUT / "evaluation_matrix_per_dataset.csv", index=False)

    overall_rows: list[dict] = []
    for method, frame in long_dataset_means.groupby("method"):
        item: dict = {"method": method, "method_name": frame["method_name"].iloc[0]}
        for metric in numeric_metrics:
            values = frame[metric].dropna()
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            item[f"{metric}_sd_between_datasets"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            item[f"{metric}_n_datasets"] = int(len(values))
        eligible = frame[frame["test_labeled_anomaly_points"] > 0]["f1_score"].dropna()
        item["f1_mean_positive_test_subset"] = float(eligible.mean()) if len(eligible) else np.nan
        item["f1_n_positive_test_datasets"] = int(len(eligible))
        overall_rows.append(item)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "evaluation_matrix_overall.csv", index=False)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT / "audit_checks.csv", index=False)
    audit_json = {
        "benchmark": "V1 full_gpu_benchmark_restart",
        "model_runs": int(run_count),
        "threshold_result_rows": int(len(original)),
        "prediction_rows_verified": int(len(augmented)),
        "prediction_files_missing": missing_predictions,
        "all_rows_gpu": bool(set(original["device"].dropna().unique()) == {"GPU:0"}),
        "max_abs_differences": {
            "f1": float((augmented["f1_score"] - augmented["f1_recomputed"]).abs().max()),
            "fpr": float((augmented["false_positive_rate"] - augmented["fpr_recomputed"]).abs().max()),
            "pr_auc": float((augmented["pr_auc"] - augmented["pr_auc_recomputed"]).abs().max()),
        },
        "roc_auc_added_from_raw_scores": True,
        "accuracy_added_from_predictions": True,
        "train_labeled_anomaly_files": int(coverage["has_train_anomaly_labels"].sum()),
        "validation_labeled_anomaly_files": int(coverage["has_validation_anomaly_labels"].sum()),
        "test_labeled_anomaly_files": int(coverage["has_test_anomaly_labels"].sum()),
        "complete_test_events": int(coverage["complete_test_events"].sum()),
        "checks": checks,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit_json, indent=2), encoding="utf-8")

    print("Recomputed outputs written to:", OUT)
    print("Runs:", run_count, "method rows:", len(original), "matched:", len(augmented), "missing:", len(missing_predictions))
    print(overall[["method_name", "f1_score_mean", "f1_score_n_datasets", "f1_mean_positive_test_subset", "f1_n_positive_test_datasets", "false_positive_rate_mean", "false_positive_rate_n_datasets", "roc_auc_recomputed_mean", "roc_auc_recomputed_n_datasets", "pr_auc_mean", "pr_auc_n_datasets", "nab_style_score_mean", "nab_style_score_n_datasets"]].to_string(index=False))
    print("\nDataset coverage:")
    print(coverage[["dataset", "domain", "train_labeled_anomaly_points", "validation_labeled_anomaly_points", "test_labeled_anomaly_points", "complete_test_events"]].to_string(index=False))
    if any(row["status"] != "PASS" for row in checks):
        raise SystemExit("One or more integrity checks failed; inspect evaluation/audit_checks.csv")


if __name__ == "__main__":
    main()
