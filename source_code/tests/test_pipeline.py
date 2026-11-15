import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import load_nab_series, make_point_labels, prepare_splits
from src.benchmark_reporting import _pairwise_table, write_benchmark_outputs
from src.metrics import build_statistical_outputs, calculate_metrics, high_variance_mask
from src.model import reconstruction_errors, train_autoencoder
from src.nab_scoring import nab_style_test_score
from src.thresholds import (
    METHOD_FIXED_PERCENTILE,
    METHOD_MOVING_MEDIAN_MAD,
    METHOD_MOVING_MEDIAN_SCALED_MAD,
    METHOD_MOVING_MEDIAN_IQR,
    METHOD_ROLLING_MEAN_3SIGMA,
    apply_threshold_method,
    rolling_thresholds,
    tune_threshold_method,
)


class DataPipelineTests(unittest.TestCase):
    def test_nab_window_labels(self):
        timestamps = pd.Series(pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            "2020-01-01 00:02:00",
            "2020-01-01 00:03:00",
        ]))
        windows = [(pd.Timestamp("2020-01-01 00:01:00"), pd.Timestamp("2020-01-01 00:02:00"))]
        labels = make_point_labels(timestamps, windows)
        np.testing.assert_array_equal(labels, np.array([0, 1, 1, 0], dtype=np.int8))

    def test_scaler_is_fitted_on_training_partition_only(self):
        values = np.concatenate([np.linspace(0, 1, 60), np.linspace(1, 2, 20), np.array([100.0] * 20)])
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="min"),
            "value": values,
        })
        labels = np.zeros(100, dtype=np.int8)
        split = prepare_splits(frame, labels, window_size=10)
        self.assertAlmostEqual(float(split["scaler"].data_max_[0]), 1.0)
        self.assertGreater(float(split["scaled_values"][-1]), 1.0)
        self.assertEqual(len(split["y_test"]), 20)

    def test_normal_only_training_excludes_any_window_touching_anomaly(self):
        values = np.linspace(0, 1, 100)
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="min"),
            "value": values,
        })
        labels = np.zeros(100, dtype=np.int8)
        labels[20:23] = 1
        split = prepare_splits(frame, labels, window_size=5, normal_only_training=True)
        self.assertGreater(split["train_windows_excluded_anomalous"], 0)
        self.assertTrue(np.all(split["y_train"] == 0))
        self.assertLessEqual(float(split["scaler"].data_max_[0]), float(values[59]))

    def test_high_variance_reference_excludes_anomaly_contaminated_training_histories(self):
        values = np.array([0, 1, 0, 1, 0, 100, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0], dtype=float)
        labels = np.zeros(len(values), dtype=np.int8)
        labels[5] = 1
        mask = high_variance_mask(
            values, train_end=10, test_indices=np.arange(10, 16),
            lookback=2, percentile=75.0, training_labels=labels,
        )
        np.testing.assert_array_equal(mask, np.array([True, True, True, True, False, True]))

    def test_loader_forward_fills_but_drops_leading_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            pd.DataFrame({
                "timestamp": pd.date_range("2020-01-01", periods=12, freq="min"),
                "value": [np.nan, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            }).to_csv(path, index=False)
            frame = load_nab_series(path)
            self.assertEqual(len(frame), 11)
            self.assertEqual(frame["value"].tolist()[:3], [2.0, 2.0, 4.0])


class ModelSmokeTests(unittest.TestCase):
    def test_lstm_autoencoder_trains_and_returns_one_error_per_window(self):
        rng = np.random.default_rng(11)
        train = rng.random((48, 12, 1), dtype=np.float32)
        validation = rng.random((16, 12, 1), dtype=np.float32)
        fitted = train_autoencoder(
            train, validation, hidden_size=8, latent_size=4, epochs=1,
            batch_size=16, patience=1, seed=15, numpy_seed=42, device="cpu"
        )
        errors = reconstruction_errors(fitted.model, validation, fitted.device)
        self.assertEqual(len(errors), len(validation))
        self.assertTrue(np.all(np.isfinite(errors)))


class ThresholdTests(unittest.TestCase):
    def test_rolling_threshold_does_not_use_current_error(self):
        predictions, thresholds = rolling_thresholds(
            np.array([10.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            lookback=3,
            method=METHOD_ROLLING_MEAN_3SIGMA,
            multiplier=3.0,
        )
        self.assertEqual(int(predictions[0]), 1)
        self.assertAlmostEqual(float(thresholds[0]), 1.0)

    def test_proposed_method_is_tuned_from_validation(self):
        train = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.10])
        validation = np.array([0.11, 0.12, 0.50, 0.10, 0.60, 0.11])
        labels = np.array([0, 0, 1, 0, 1, 0], dtype=np.int8)
        result = tune_threshold_method(
            METHOD_MOVING_MEDIAN_MAD,
            train,
            validation,
            labels,
            lookback=4,
            percentiles=[95],
            k_values=[2],
            mad_lambdas=[1.0, 2.0, 3.0],
        )
        self.assertIn(result.parameter, [1.0, 2.0, 3.0])
        self.assertEqual(len(result.predictions), len(labels))

    def test_threshold_tuning_respects_common_validation_fpr_budget_when_feasible(self):
        train = np.linspace(0.01, 0.10, 100)
        validation = np.array([0.02, 0.03, 0.04, 0.05, 0.20, 0.25, 0.30, 0.35])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = tune_threshold_method(
            METHOD_MOVING_MEDIAN_MAD, train, validation, labels,
            lookback=4, percentiles=[95], k_values=[1, 2, 3],
            mad_lambdas=[1, 2, 3, 4, 5, 6, 8, 10], max_validation_fpr=0.05,
        )
        self.assertLessEqual(result.validation_fpr, 0.05)


    def test_scaled_mad_and_iqr_thresholds_are_finite_and_causal(self):
        seed = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.10], dtype=float)
        test = np.array([0.50, 0.10], dtype=float)
        for method in (METHOD_MOVING_MEDIAN_SCALED_MAD, METHOD_MOVING_MEDIAN_IQR):
            predictions, thresholds = rolling_thresholds(
                test, seed, lookback=6, method=method, multiplier=2.0
            )
            self.assertTrue(np.isfinite(thresholds).all())
            self.assertEqual(int(predictions[0]), 1)
            self.assertEqual(int(predictions[1]), 0)

    def test_no_positive_validation_uses_fpr_budget_without_minimizing_fpr_blindly(self):
        train = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
        validation = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)
        labels = np.zeros(len(validation), dtype=np.int8)
        result = tune_threshold_method(
            METHOD_MOVING_MEDIAN_MAD, train, validation, labels,
            lookback=4, percentiles=[95], k_values=[1, 2, 3],
            mad_lambdas=[0.5, 1.0, 2.0, 3.0, 4.0],
            max_validation_fpr=0.25,
        )
        self.assertLessEqual(result.validation_fpr, 0.25)
        self.assertGreater(result.validation_fpr, 0.0)
        self.assertNotEqual(result.parameter, 4.0)

    def test_test_threshold_uses_validation_history_causally(self):
        train = np.array([0.1, 0.1, 0.1, 0.1])
        validation = np.array([0.1, 0.1, 0.1, 0.1])
        test = np.array([2.0, 0.1])
        preds, thresholds = apply_threshold_method(
            METHOD_MOVING_MEDIAN_MAD,
            3.0,
            train,
            validation,
            test,
            lookback=4,
        )
        self.assertEqual(int(preds[0]), 1)
        self.assertEqual(int(preds[1]), 0)  # equality is not an anomaly (strict >)
        self.assertAlmostEqual(float(thresholds[0]), 0.1)

    def test_fixed_percentile_returns_constant_test_threshold(self):
        train = np.array([0.0, 1.0, 2.0, 3.0])
        validation = np.array([0.0, 1.0, 2.0])
        result = tune_threshold_method(
            METHOD_FIXED_PERCENTILE, train, validation,
            np.array([0, 0, 1]), lookback=3,
            percentiles=[50], k_values=[2], mad_lambdas=[2],
        )
        self.assertEqual(result.parameter, 50.0)
        self.assertEqual(len(result.predictions), 3)

    def test_metric_function_accepts_window_labels(self):
        ts = pd.date_range("2020-01-01", periods=10, freq="min")
        y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
        pred = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        metrics = calculate_metrics(
            y,
            pred,
            test_timestamps=ts,
            test_windows=[(ts[2], ts[4])],
            high_variance_points=np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            anomaly_scores=np.array([0.1, 0.1, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1]),
        )
        self.assertEqual(metrics["test_anomaly_events"], 1.0)
        self.assertEqual(metrics["detected_anomaly_events"], 1.0)
        self.assertEqual(metrics["nab_events_scored"], 1)
        self.assertTrue(np.isfinite(metrics["nab_style_score"]))
        self.assertGreater(metrics["f1_score"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertTrue(np.isfinite(metrics["roc_auc"]))
        self.assertIn("mean_detection_delay_seconds", metrics)
        self.assertAlmostEqual(metrics["mean_detection_delay_seconds"], 0.0)

    def test_nab_style_score_early_detection_beats_late_detection(self):
        ts = pd.date_range("2020-01-01", periods=20, freq="min")
        event = [(ts[5], ts[9])]
        early = np.zeros(20, dtype=np.int8)
        late = np.zeros(20, dtype=np.int8)
        early[5] = 1
        late[9] = 1
        early_score = nab_style_test_score(ts, early, event)["nab_style_score"]
        late_score = nab_style_test_score(ts, late, event)["nab_style_score"]
        self.assertGreater(early_score, late_score)
        self.assertAlmostEqual(early_score, 100.0)

    def test_nab_style_score_null_and_false_alarm_behavior(self):
        ts = pd.date_range("2020-01-01", periods=20, freq="min")
        event = [(ts[5], ts[9])]
        no_detections = np.zeros(20, dtype=np.int8)
        perfect = no_detections.copy()
        perfect[5] = 1
        with_false_alarm = perfect.copy()
        with_false_alarm[1] = 1
        null_score = nab_style_test_score(ts, no_detections, event)["nab_style_score"]
        perfect_score = nab_style_test_score(ts, perfect, event)["nab_style_score"]
        false_alarm_score = nab_style_test_score(ts, with_false_alarm, event)["nab_style_score"]
        self.assertAlmostEqual(null_score, 0.0)
        self.assertAlmostEqual(perfect_score, 100.0)
        self.assertLess(false_alarm_score, perfect_score)

    def test_statistical_outputs_follow_original_proposal_hypotheses(self):
        methods = [
            "fixed_percentile", "fixed_mean_k_sigma", "rolling_mean_3sigma",
            "moving_median_lambda_mad",
        ]
        rows = []
        for d in range(4):
            for method in methods:
                rows.append({
                    "dataset": f"data_{d}",
                    "domain": "AWS CloudWatch" if d < 2 else "Traffic",
                    "method": method, "method_name": method,
                    "f1_score": 0.70 if method == "moving_median_lambda_mad" else 0.60,
                    "false_positive_rate": 0.10,
                    "high_variance_fpr": 0.05 if method == "moving_median_lambda_mad" else 0.10,
                    "pr_auc": 0.60, "nab_style_score": 50.0,
                    "mean_detection_delay_fraction": 0.2,
                    "test_anomaly_events": 1,
                })
        statistical, hypotheses, domains = build_statistical_outputs(pd.DataFrame(rows))
        self.assertEqual(len(statistical), 8)  # H1:2, H2:3, H3:3 comparisons
        self.assertEqual(len(hypotheses), 3)
        self.assertIn("AWS CloudWatch", domains["domain"].tolist())
        h1 = hypotheses[hypotheses["hypothesis"].str.startswith("H1")].iloc[0]
        self.assertIn("fixed_percentile", h1["fixed_baselines_with_positive_mean_difference"])
        self.assertIn("Directional only", h1["outcome"])
        h3 = hypotheses[hypotheses["hypothesis"].str.startswith("H3")].iloc[0]
        self.assertIn("NAB score", h3["hypothesis"])
        self.assertNotIn("AWS", h3["hypothesis"])

    def test_h1_requires_positive_and_significant_result_against_same_baseline(self):
        methods = [
            "fixed_percentile", "fixed_mean_k_sigma", "rolling_mean_3sigma",
            "moving_median_lambda_mad",
        ]
        rows = []
        for d in range(8):
            for method in methods:
                if method == "moving_median_lambda_mad":
                    f1 = 0.5
                elif method == "fixed_percentile":
                    f1 = 0.6  # Proposed loses consistently and significantly.
                elif method == "fixed_mean_k_sigma":
                    f1 = 0.0 if d == 0 else 0.51  # Positive mean delta, but not significant.
                else:
                    f1 = 0.5
                rows.append({
                    "dataset": f"data_{d}", "domain": "Test",
                    "method": method, "method_name": method,
                    "f1_score": f1, "false_positive_rate": 0.1,
                    "high_variance_fpr": 0.1, "pr_auc": 0.5,
                    "nab_style_score": 0.0, "mean_detection_delay_fraction": 0.5,
                    "test_anomaly_events": 1,
                })
        _, hypotheses, _ = build_statistical_outputs(pd.DataFrame(rows))
        h1 = hypotheses[hypotheses["hypothesis"].str.startswith("H1")].iloc[0]
        self.assertIn("fixed_mean_k_sigma", h1["fixed_baselines_with_positive_mean_difference"])
        self.assertNotIn("Supported on evaluated datasets", h1["outcome"])

    def test_h2_is_not_reported_as_a_general_gain_when_fpr_increases(self):
        methods = [
            "fixed_percentile", "fixed_mean_k_sigma", "rolling_mean_3sigma",
            "moving_median_lambda_mad",
        ]
        rows = []
        for d in range(8):
            for method in methods:
                rows.append({
                    "dataset": f"data_{d}", "domain": "AWS CloudWatch" if d < 4 else "Traffic",
                    "method": method, "method_name": method,
                    "f1_score": 0.2, "false_positive_rate": 0.1,
                    "high_variance_fpr": 0.30 if method == "moving_median_lambda_mad" else 0.05,
                    "pr_auc": 0.3, "nab_style_score": -10.0,
                    "mean_detection_delay_fraction": 0.2,
                    "test_anomaly_events": 1,
                })
        _, hypotheses, _ = build_statistical_outputs(pd.DataFrame(rows))
        h2 = hypotheses[hypotheses["hypothesis"].str.startswith("H2")].iloc[0]
        self.assertIn("Not supported", h2["outcome"])

    def test_h3_uses_nab_score_not_domain_ranking(self):
        methods = [
            "fixed_percentile", "fixed_mean_k_sigma", "rolling_mean_3sigma",
            "moving_median_lambda_mad",
        ]
        rows = []
        for d in range(3):
            for method in methods:
                rows.append({
                    "dataset": f"aws_{d}", "domain": "AWS CloudWatch",
                    "method": method, "method_name": method,
                    "f1_score": 0.70 if method == "moving_median_lambda_mad" else 0.60,
                    "false_positive_rate": 0.20 if method == "moving_median_lambda_mad" else 0.10,
                    "high_variance_fpr": 0.05, "pr_auc": 0.60,
                    "nab_style_score": 40.0 if method == "moving_median_lambda_mad" else 50.0,
                    "mean_detection_delay_fraction": 0.2,
                    "test_anomaly_events": 1,
                    "test_positive_points": 10,
                    "high_variance_normal_points": 20,
                })
        _, hypotheses, domains = build_statistical_outputs(pd.DataFrame(rows))
        h3 = hypotheses[hypotheses["hypothesis"].str.startswith("H3")].iloc[0]
        self.assertIn("Not supported overall", h3["outcome"])
        self.assertEqual(domains["domain"].tolist(), ["AWS CloudWatch"])


class BenchmarkReportingTests(unittest.TestCase):
    def test_pairwise_counts_roundoff_difference_as_single_tie(self):
        def make_pairwise_table(use_roundoff):
            methods = [
                "fixed_percentile",
                "fixed_mean_k_sigma",
                "rolling_mean_3sigma",
                "moving_median_lambda_mad",
            ]

            tiny_residue = 4.45607092891542e-17
            roundoff_baseline = (
                0.1 - tiny_residue if use_roundoff else 0.1
            )

            comparisons = [
                ("proposed_win", 0.1, 0.2),
                ("baseline_win", 0.2, 0.1),
                ("roundoff_tie", 0.1, roundoff_baseline),
                ("exact_tie", 0.3, 0.3),
            ]

            rows = []

            for dataset, proposed_fpr, rolling_fpr in comparisons:
                values = {
                    "fixed_percentile": 0.15,
                    "fixed_mean_k_sigma": 0.15,
                    "rolling_mean_3sigma": rolling_fpr,
                    "moving_median_lambda_mad": proposed_fpr,
                }

                for method in methods:
                    rows.append({
                        "dataset": dataset,
                        "domain": "Regression test",
                        "method": method,
                        "false_positive_rate": values[method],
                        "test_anomaly_events": 1,
                    })

            return _pairwise_table(pd.DataFrame(rows))

        noisy_result = make_pairwise_table(use_roundoff=True)
        exact_result = make_pairwise_table(use_roundoff=False)

        def get_rolling_fpr_row(result):
            selected = result[
                (result["metric"] == "false_positive_rate") &
                (result["baseline_method"] == "rolling_mean_3sigma")
            ]
            self.assertEqual(len(selected), 1)
            return selected.iloc[0]

        noisy = get_rolling_fpr_row(noisy_result)
        exact = get_rolling_fpr_row(exact_result)

        self.assertEqual(int(noisy["n_datasets"]), 4)
        self.assertEqual(int(noisy["proposed_better_datasets"]), 1)
        self.assertEqual(int(noisy["baseline_better_datasets"]), 1)
        self.assertEqual(int(noisy["ties"]), 2)

        self.assertEqual(
            int(noisy["proposed_better_datasets"]) +
            int(noisy["baseline_better_datasets"]) +
            int(noisy["ties"]),
            int(noisy["n_datasets"]),
        )

        for column in [
            "mean_improvement_directional",
            "test_statistic",
            "p_value",
            "ci95_low",
            "ci95_high",
        ]:
            self.assertAlmostEqual(
                float(noisy[column]),
                float(exact[column]),
                places=14,
                msg=f"Roundoff changed {column}",
            )

    def test_reporter_writes_pairwise_benchmark_and_rq_files(self):
        methods = [
            "fixed_percentile", "fixed_mean_k_sigma", "rolling_mean_3sigma",
            "moving_median_lambda_mad",
        ]
        rows = []
        domains = ["AWS CloudWatch", "Twitter", "Traffic", "Known Cause"]
        for dataset_index in range(8):
            domain = domains[dataset_index % len(domains)]
            for window_size in [30, 50]:
                for seed in [15, 42]:
                    for method_index, method in enumerate(methods):
                        rows.append({
                            "dataset": f"dataset_{dataset_index}.csv",
                            "domain": domain,
                            "window_size": window_size,
                            "seed": seed,
                            "method": method,
                            "method_name": method,
                            "f1_score": 0.50 + 0.04 * method_index,
                            "false_positive_rate": 0.20 - 0.02 * method_index,
                            "pr_auc": 0.45 + 0.03 * method_index,
                            "nab_style_score": 25.0 + 4.0 * method_index,
                            "event_recall": 0.40 + 0.10 * method_index,
                            "high_variance_fpr": 0.16 - 0.01 * method_index,
                            "mean_detection_delay_fraction": 0.60 - 0.05 * method_index,
                            "test_anomaly_events": 1,
                            "test_positive_points": 10,
                            "high_variance_normal_points": 30,
                        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = write_benchmark_outputs(pd.DataFrame(rows), output)
            self.assertEqual(len(result["pairwise"]), 7 * 3)
            self.assertTrue((output / "pairwise_proposed_vs_baselines.csv").exists())
            self.assertTrue((output / "domain_comparison.csv").exists())
            self.assertTrue((output / "research_question_results.md").exists())
            report = (output / "research_question_results.md").read_text(encoding="utf-8")
            self.assertIn("RQ1 - Baseline performance", report)
            self.assertIn("RQ2 - Does moving median", report)
            self.assertIn("RQ3 - False alarms", report)
            self.assertIn("RQ4 - Which NAB domain", report)


if __name__ == "__main__":
    unittest.main()
