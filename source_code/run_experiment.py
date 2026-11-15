from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import scipy

from src.data_utils import (
    download_nab_assets,
    load_label_windows,
    load_nab_series,
    make_point_labels,
    prepare_splits,
)
from src.eda import write_rolling_eda
from src.benchmark_reporting import metrics_for_reporting, write_benchmark_outputs
from src.metrics import build_statistical_outputs, calculate_metrics, high_variance_mask
from src.model import get_runtime_info, reconstruction_errors, save_checkpoint, train_autoencoder
from src.thresholds import (
    METHOD_FIXED_MEAN_SIGMA,
    METHOD_FIXED_PERCENTILE,
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_MOVING_MEDIAN_SCALED_MAD,
    METHOD_MOVING_MEDIAN_IQR,
    METHOD_NAMES,
    METHOD_ROLLING_MEAN_3SIGMA,
    apply_threshold_method,
    tune_threshold_method,
)

ROOT = Path(__file__).resolve().parent
METHOD_ORDER = [
    METHOD_FIXED_PERCENTILE,
    METHOD_FIXED_MEAN_SIGMA,
    METHOD_ROLLING_MEAN_3SIGMA,
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_MOVING_MEDIAN_SCALED_MAD,
    METHOD_MOVING_MEDIAN_IQR,
]


def read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fixed and adaptive thresholds on LSTM-autoencoder reconstruction errors."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nab")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="e.g. --windows 30 50 100")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Independent model-training seeds; defaults to config.json")
    parser.add_argument("--files", nargs="*", default=None, help="Exact NAB relative paths from config.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-save-calibration-traces", action="store_true", help="Do not save train/validation/test reconstruction-error arrays for threshold replay")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--allow-contaminated-training", action="store_true", help="Override config and train on all windows")
    return parser.parse_args()


def _complete_test_windows(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    test_timestamps: pd.Series,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], int]:
    """Keep only full labeled events wholly inside the held-out test partition."""
    if len(test_timestamps) == 0:
        return [], len(windows)
    test_index = pd.DatetimeIndex(pd.to_datetime(test_timestamps))
    start_time = test_index[0]
    end_time = test_index[-1]
    available_timestamps = set(test_index)
    complete = [
        (start, end)
        for start, end in windows
        if start_time <= start <= end <= end_time
        and pd.Timestamp(start) in available_timestamps
        and pd.Timestamp(end) in available_timestamps
    ]
    intersecting = [
        (start, end)
        for start, end in windows
        if end >= start_time and start <= end_time
    ]
    return complete, max(0, len(intersecting) - len(complete))


def _plot_dataset_result(
    timestamps: pd.Series,
    errors: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = pd.to_datetime(timestamps)
    ax.plot(x, errors, linewidth=1.0, label="Reconstruction error")
    ax.fill_between(
        x,
        0,
        max(float(np.nanmax(errors)), 1e-9),
        where=np.asarray(labels) == 1,
        alpha=0.15,
        label="Labeled anomaly window",
    )
    colors = {
        METHOD_FIXED_PERCENTILE: "#6b7280",
        METHOD_FIXED_MEAN_SIGMA: "#0f766e",
        METHOD_ROLLING_MEAN_3SIGMA: "#dc2626",
        METHOD_MOVING_MEDIAN_MAD: "#2563eb",
        METHOD_MOVING_MEDIAN_SCALED_MAD: "#7c3aed",
        METHOD_MOVING_MEDIAN_IQR: "#c026d3",
    }
    for method, threshold in thresholds.items():
        ax.plot(
            x,
            threshold,
            linewidth=1.0,
            alpha=0.9,
            color=colors.get(method),
            label=f"{METHOD_NAMES[method]} threshold",
        )
        flagged = np.asarray(predictions[method]) == 1
        if flagged.any():
            ax.scatter(
                x[flagged], errors[flagged], s=12, alpha=0.65,
                color=colors.get(method), marker="x",
            )
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Reconstruction error / threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_aggregates(metrics: pd.DataFrame, output_dir: Path):
    if metrics.empty:
        return
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [METHOD_NAMES[method] for method in METHOD_ORDER]

    for metric, ylabel, filename in [
        ("f1_score", "Mean test F1-score", "mean_test_f1.png"),
        ("false_positive_rate", "Mean test false-positive rate", "mean_test_fpr.png"),
        ("pr_auc", "Mean test PR-AUC / average precision", "mean_test_pr_auc.png"),
        ("nab_style_score", "Mean test NAB-style normalized score", "mean_test_nab_style_score.png"),
    ]:
        grouped = metrics.groupby("method")[metric].agg(["mean", "std"]).reindex(METHOD_ORDER)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(labels, grouped["mean"].values, yerr=grouped["std"].fillna(0).values, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by threshold method (mean +/- SD across runs)")
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)


def _write_summaries(metrics_frame: pd.DataFrame, output_dir: Path):
    reporting_frame = metrics_for_reporting(metrics_frame)
    summary = (
        reporting_frame.groupby(["method", "method_name"], as_index=False)
        .agg(
            mean_f1=("f1_score", "mean"),
            std_f1=("f1_score", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            std_fpr=("false_positive_rate", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            std_roc_auc=("roc_auc", "std"),
            mean_nab_style_score=("nab_style_score", "mean"),
            std_nab_style_score=("nab_style_score", "std"),
            mean_event_recall=("event_recall", "mean"),
            mean_high_variance_fpr=("high_variance_fpr", "mean"),
            mean_detection_delay_seconds=("mean_detection_delay_seconds", "mean"),
            mean_detection_delay_fraction=("mean_detection_delay_fraction", "mean"),
            f1_evaluable_runs=("f1_score", "count"),
            event_evaluable_runs=("event_recall", "count"),
            nab_evaluable_runs=("nab_style_score", "count"),
            fpr_evaluable_runs=("false_positive_rate", "count"),
            runs=("dataset", "count"),
        )
    )
    summary.to_csv(output_dir / "method_summary.csv", index=False)

    dataset_summary = (
        reporting_frame.groupby(["dataset", "domain", "method", "method_name"], as_index=False)
        .agg(
            mean_f1=("f1_score", "mean"),
            std_f1=("f1_score", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_event_recall=("event_recall", "mean"),
            mean_detection_delay_fraction=("mean_detection_delay_fraction", "mean"),
            mean_detection_delay_seconds=("mean_detection_delay_seconds", "mean"),
            mean_nab_style_score=("nab_style_score", "mean"),
            mean_high_variance_fpr=("high_variance_fpr", "mean"),
            runs=("seed", "count"),
        )
    )
    dataset_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)

    # Exploratory threshold-variant summary: average within each dataset first,
    # then summarize across datasets so long series / repeated seeds do not
    # silently dominate. These comparisons are descriptive, not confirmatory.
    exploratory_metrics = [
        "mean_f1", "mean_precision", "mean_recall", "mean_fpr",
        "mean_event_recall", "mean_detection_delay_fraction", "mean_nab_style_score",
        "mean_high_variance_fpr", "mean_pr_auc", "mean_roc_auc",
    ]
    available_metrics = [metric for metric in exploratory_metrics if metric in dataset_summary.columns]
    variant_summary = (
        dataset_summary.groupby(["method", "method_name"], as_index=False)
        .agg(**{f"macro_mean_{metric.removeprefix('mean_')}": (metric, "mean") for metric in available_metrics},
             **{f"macro_sd_{metric.removeprefix('mean_')}": (metric, "std") for metric in available_metrics},
             datasets=("dataset", "nunique"),
             positive_test_datasets=("mean_f1", "count"),
             complete_event_datasets=("mean_event_recall", "count"),
             false_alarm_datasets=("mean_fpr", "count"))
    )
    variant_summary["interpretation"] = "Exploratory macro-average across dataset-level means; not a confirmatory significance test"
    variant_summary.to_csv(output_dir / "threshold_ablation_summary.csv", index=False)

    statistical_tests, hypothesis_outcomes, domain_summary = build_statistical_outputs(reporting_frame)
    statistical_tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    hypothesis_outcomes.to_csv(output_dir / "hypothesis_outcomes.csv", index=False)
    domain_summary.to_csv(output_dir / "domain_improvement_summary.csv", index=False)
    return statistical_tests, hypothesis_outcomes, domain_summary


def run():
    args = parse_args()
    config = read_config(args.config)
    selected_files = config["selected_files"]
    if args.files:
        missing = set(args.files).difference(selected_files)
        if missing:
            raise SystemExit(f"These files are not in config.json: {sorted(missing)}")
        selected_files = {key: selected_files[key] for key in args.files}

    windows_to_run = args.windows or config["window_sizes"]
    run_seeds = args.seeds or config.get("run_seeds", [15, 42, 123])
    if not run_seeds:
        raise SystemExit("At least one run seed is required.")
    if any(window < 2 for window in windows_to_run):
        raise SystemExit("All sequence lengths must be >= 2.")

    model_config = dict(config["model"])
    if args.epochs is not None:
        model_config["epochs"] = args.epochs
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.patience is not None:
        model_config["patience"] = args.patience

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    plots_dir = args.output_dir / "plots"
    models_dir = args.output_dir / "models"
    traces_dir = args.output_dir / "calibration_traces"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_calibration_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.data_dir / config["labels_file"]
    files_missing = [rel for rel in selected_files if not (args.data_dir / "data" / rel).exists()]
    if not labels_path.exists() or files_missing or args.force_download:
        download_nab_assets(
            args.data_dir,
            selected_files,
            config["nab_repo_raw_base"],
            config["labels_file"],
            force=args.force_download,
        )

    label_windows = load_label_windows(labels_path)
    threshold_config = config["threshold_grids"]
    normal_only_training = bool(config.get("normal_only_training", False)) and not args.allow_contaminated_training
    validation_fpr_budget = float(config.get("validation_fpr_budget", 0.05))
    mad_relative_floor = float(config.get("mad_relative_floor", 0.001))
    all_metrics: list[dict] = []
    all_tuning: list[dict] = []
    all_split_audit: list[dict] = []
    all_eda: list[dict] = []
    numpy_seed = int(config.get("numpy_seed", 42))
    rolling_lookback_note = "same as sequence length W"
    high_variance_percentile = float(config.get("high_variance_percentile", 75.0))
    eda_window = int(config.get("eda_rolling_window", 50))

    for file_number, (relative_path, domain) in enumerate(selected_files.items(), start=1):
        csv_path = args.data_dir / "data" / relative_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find {csv_path}. Data should be under {args.data_dir / 'data'}."
            )
        if relative_path not in label_windows:
            raise KeyError(f"No NAB label-window entry found for {relative_path}")

        frame = load_nab_series(csv_path)
        windows = label_windows[relative_path]
        labels = make_point_labels(frame["timestamp"], windows)
        train_fraction = config["split"]["train_fraction"]
        validation_fraction = config["split"]["validation_fraction"]
        print(
            f"\n[{file_number}/{len(selected_files)}] {relative_path} | rows={len(frame)} "
            f"| labeled points={int(labels.sum())}"
        )

        if not args.skip_eda:
            all_eda.append(
                write_rolling_eda(
                    frame,
                    relative_path,
                    domain,
                    args.output_dir / "eda",
                    rolling_window=eda_window,
                )
            )

        for window_size in windows_to_run:
            split = prepare_splits(
                frame,
                labels,
                window_size=window_size,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                normal_only_training=normal_only_training,
            )
            if min(len(split["x_train"]), len(split["x_validation"]), len(split["x_test"])) == 0:
                print(f"  Skipped W={window_size}: not enough sequences in one of the partitions.")
                continue

            test_windows, partial_events_excluded = _complete_test_windows(
                windows, split["timestamps_test"]
            )
            high_var = high_variance_mask(
                frame["value"].to_numpy(dtype=float),
                split["train_end"],
                split["test_ends"],
                lookback=window_size,
                percentile=high_variance_percentile,
                training_labels=labels,
            )

            for seed in run_seeds:
                print(f"  W={window_size}, model seed={seed}: fitting LSTM autoencoder...")
                training = train_autoencoder(
                    split["x_train"],
                    split["x_validation_normal"] if normal_only_training else split["x_validation"],
                    hidden_size=model_config["hidden_size"],
                    latent_size=model_config["latent_size"],
                    epochs=model_config["epochs"],
                    batch_size=model_config["batch_size"],
                    learning_rate=model_config["learning_rate"],
                    weight_decay=model_config["weight_decay"],
                    patience=model_config["patience"],
                    seed=int(seed),
                    numpy_seed=numpy_seed,
                    device=args.device,
                )
                train_errors = reconstruction_errors(training.model, split["x_train"], training.device)
                validation_errors = reconstruction_errors(training.model, split["x_validation"], training.device)
                test_errors = reconstruction_errors(training.model, split["x_test"], training.device)

                tuned = {}
                for method in METHOD_ORDER:
                    tuning_result = tune_threshold_method(
                        method,
                        train_errors,
                        validation_errors,
                        split["y_validation"],
                        lookback=window_size,
                        percentiles=threshold_config["fixed_percentiles"],
                        k_values=threshold_config["fixed_k_values"],
                        mad_lambdas=threshold_config["mad_lambda_values"],
                        max_validation_fpr=validation_fpr_budget,
                        mad_relative_floor=mad_relative_floor,
                        robust_scale_multipliers=threshold_config.get("robust_scale_multiplier_values", threshold_config["fixed_k_values"]),
                    )
                    tuned[method] = tuning_result
                    all_tuning.append({
                        "dataset": relative_path,
                        "domain": domain,
                        "window_size": window_size,
                        "seed": int(seed),
                        "method": method,
                        "method_name": METHOD_NAMES[method],
                        "selected_parameter": tuning_result.parameter,
                        "validation_f1": tuning_result.validation_f1,
                        "validation_fpr": tuning_result.validation_fpr,
                        "validation_fpr_budget": validation_fpr_budget,
                        "validation_fpr_budget_met": bool(tuning_result.validation_fpr <= validation_fpr_budget),
                        "validation_positive_points": int(np.sum(split["y_validation"])),
                        "validation_points": int(len(split["y_validation"])),
                    })

                test_predictions: dict[str, np.ndarray] = {}
                test_thresholds: dict[str, np.ndarray] = {}
                metric_by_method: dict[str, dict] = {}
                for method in METHOD_ORDER:
                    chosen = tuned[method]
                    prediction, threshold = apply_threshold_method(
                        method,
                        float(chosen.parameter),
                        train_errors,
                        validation_errors,
                        test_errors,
                        lookback=window_size,
                        mad_relative_floor=mad_relative_floor,
                    )
                    test_predictions[method] = prediction
                    test_thresholds[method] = threshold
                    metrics = calculate_metrics(
                        split["y_test"],
                        prediction,
                        test_timestamps=split["timestamps_test"],
                        test_windows=test_windows,
                        high_variance_points=high_var,
                        anomaly_scores=test_errors,
                    )
                    metric_by_method[method] = metrics
                    all_metrics.append({
                        "dataset": relative_path,
                        "domain": domain,
                        "window_size": window_size,
                        "seed": int(seed),
                        "method": method,
                        "method_name": METHOD_NAMES[method],
                        "selected_parameter": chosen.parameter,
                        "best_validation_loss": training.best_validation_loss,
                        "epochs_run": training.epochs_run,
                        "device": training.device,
                        "test_points": int(len(split["y_test"])),
                        "test_positive_points": int(np.sum(split["y_test"])),
                        "partial_events_excluded_from_event_scoring": int(partial_events_excluded),
                        **metrics,
                    })

                output_table = pd.DataFrame({
                    "timestamp": split["timestamps_test"].to_numpy(),
                    "actual_label": split["y_test"],
                    "reconstruction_error": test_errors,
                    "high_variance": high_var.astype(int),
                })
                for method in METHOD_ORDER:
                    slug = method.replace(" ", "_")
                    output_table[f"{slug}_threshold"] = test_thresholds[method]
                    output_table[f"{slug}_prediction"] = test_predictions[method]
                stem = Path(relative_path).stem
                output_table.to_csv(
                    predictions_dir / f"{stem}_W{window_size}_seed{seed}.csv",
                    index=False,
                )

                # Preserve calibration traces so threshold variants can be replayed
                # later without retraining the LSTM-AE or leaking test labels.
                if not args.no_save_calibration_traces:
                    np.savez_compressed(
                        traces_dir / f"{stem}_W{window_size}_seed{seed}.npz",
                        train_errors=np.asarray(train_errors, dtype=np.float32),
                        validation_errors=np.asarray(validation_errors, dtype=np.float32),
                        validation_labels=np.asarray(split["y_validation"], dtype=np.int8),
                        test_errors=np.asarray(test_errors, dtype=np.float32),
                    )

                # Retain one readable threshold plot per file at W=50, seed 15 (or first configured seed).
                representative_seed = int(run_seeds[0])
                if seed == representative_seed and (
                    window_size == 50 or (50 not in windows_to_run and window_size == windows_to_run[0])
                ):
                    _plot_dataset_result(
                        split["timestamps_test"],
                        test_errors,
                        split["y_test"],
                        test_thresholds,
                        test_predictions,
                        plots_dir / f"{stem}_W{window_size}.png",
                        f"{stem} | Test partition | LSTM-AE reconstruction errors (W={window_size})",
                    )

                if args.save_models:
                    checkpoint_base = models_dir / f"{stem}_W{window_size}_seed{seed}"
                    checkpoint_metadata = {
                        "window_size": int(window_size),
                        "input_size": int(split["x_train"].shape[-1]),
                        "hidden_size": int(model_config["hidden_size"]),
                        "latent_size": int(model_config["latent_size"]),
                        "scaler_min": split["scaler"].min_.tolist(),
                        "scaler_scale": split["scaler"].scale_.tolist(),
                        "model_seed": int(seed),
                        "numpy_seed": numpy_seed,
                        "backend": get_runtime_info()["framework"],
                    }
                    save_checkpoint(training.model, checkpoint_base, checkpoint_metadata)

                all_split_audit.append({
                    "dataset": relative_path,
                    "domain": domain,
                    "window_size": window_size,
                    "seed": int(seed),
                    "rows": len(frame),
                    "start_timestamp": frame["timestamp"].iloc[0],
                    "end_timestamp": frame["timestamp"].iloc[-1],
                    "train_rows": split["train_end"],
                    "validation_rows": split["validation_end"] - split["train_end"],
                    "test_rows": len(frame) - split["validation_end"],
                    "train_anomaly_points": int(np.sum(labels[: split["train_end"]])),
                    "normal_only_training": normal_only_training,
                    "train_windows_before_filter": split["train_windows_before_filter"],
                    "train_windows_excluded_anomalous": split["train_windows_excluded_anomalous"],
                    "train_windows_used": int(len(split["x_train"])),
                    "normal_validation_windows_for_early_stopping": int(split["validation_windows_normal"]),
                    "validation_anomaly_points": int(np.sum(split["y_validation"])),
                    "test_anomaly_points": int(np.sum(split["y_test"])),
                    "test_complete_anomaly_events": len(test_windows),
                    "test_partial_anomaly_events_excluded": int(partial_events_excluded),
                    "best_validation_loss": training.best_validation_loss,
                    "epochs_run": training.epochs_run,
                    "device": training.device,
                })

    metrics_frame = pd.DataFrame(all_metrics)
    tuning_frame = pd.DataFrame(all_tuning)
    audit_frame = pd.DataFrame(all_split_audit)
    if metrics_frame.empty:
        raise RuntimeError("No experiment completed. Check selected files and window sizes.")

    metrics_frame.to_csv(args.output_dir / "test_metrics.csv", index=False)
    tuning_frame.to_csv(args.output_dir / "validation_tuning.csv", index=False)
    audit_frame.to_csv(args.output_dir / "dataset_split_audit.csv", index=False)
    if all_eda:
        pd.DataFrame(all_eda).to_csv(args.output_dir / "eda_summary.csv", index=False)

    statistical_tests, hypothesis_outcomes, domain_summary = _write_summaries(metrics_frame, args.output_dir)
    write_benchmark_outputs(metrics_frame, args.output_dir, hypothesis_outcomes)
    _plot_aggregates(metrics_for_reporting(metrics_frame), args.output_dir)

    dataset_audit_summary = audit_frame.groupby("dataset").agg(
        positive_test_points=("test_anomaly_points", "max"),
        complete_test_events=("test_complete_anomaly_events", "max"),
    )

    run_summary = {
        "selected_datasets": list(selected_files),
        "dataset_count": len(selected_files),
        "positive_test_dataset_count": int(
            (dataset_audit_summary["positive_test_points"] > 0).sum()
        ),
        "event_evaluable_dataset_count": int(
            (dataset_audit_summary["complete_test_events"] > 0).sum()
        ),
        "negative_only_test_dataset_count": int(
            (dataset_audit_summary["positive_test_points"] == 0).sum()
        ),
        "window_sizes": list(windows_to_run),
        "model_run_seeds": [int(seed) for seed in run_seeds],
        "numpy_seed": numpy_seed,
        "train_validation_test_split": config["split"],
        "rolling_threshold_lookback": rolling_lookback_note,
        "high_variance_percentile_from_clean_normal_training_histories": high_variance_percentile,
        "normal_only_training": normal_only_training,
        "validation_fpr_budget": validation_fpr_budget,
        "mad_relative_floor": mad_relative_floor,
        "calibration_traces_saved": not args.no_save_calibration_traces,
        "threshold_tuning_note": "When validation has no positive labels, select the candidate with validation FPR closest to the prespecified budget from below; if no candidate meets the budget, the fallback selects the lowest-FPR candidate.",
        "model_hyperparameters": model_config,
        "threshold_grids": threshold_config,
        "threshold_methods": METHOD_ORDER,
        "requested_device": args.device,
        "models_trained": int(metrics_frame[["dataset", "window_size", "seed"]].drop_duplicates().shape[0]),
        "metric_rows": int(len(metrics_frame)),
        "research_questions_report": "research_question_results.md",
        "benchmark_comparison_outputs": [
            "benchmark_method_summary.csv", "per_dataset_comparison.csv",
            "pairwise_proposed_vs_baselines.csv", "comparison_by_window_size.csv",
            "domain_comparison.csv", "high_variance_fpr_by_dataset.csv",
            "seed_variability.csv"
        ],
        "nab_score_note": "NAB-style test-partition score; not the official NAB corpus leaderboard score.",
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)

    config_snapshot = dict(config)
    config_snapshot["selected_files"] = selected_files
    config_snapshot["window_sizes"] = list(windows_to_run)
    config_snapshot["run_seeds"] = [int(seed) for seed in run_seeds]
    config_snapshot["model"] = model_config
    with (args.output_dir / "config_used.json").open("w", encoding="utf-8") as file:
        json.dump(config_snapshot, file, indent=2)

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "framework_runtime": get_runtime_info(),
    }
    with (args.output_dir / "environment.json").open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2)

    print("\nExperiment complete.")
    print(f"Outputs: {args.output_dir.resolve()}")
    print("Main outputs: test_metrics.csv, method_summary.csv, statistical_tests.csv, hypothesis_outcomes.csv")
    print("NAB score note: outputs contain a test-partition NAB-style score, not the official NAB leaderboard score.")


if __name__ == "__main__":
    run()
