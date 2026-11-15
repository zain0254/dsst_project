import unittest
import numpy as np
import pandas as pd

from src.benchmark_reporting import METHOD_ORDER, dataset_level_means


class ReportingPopulationTests(unittest.TestCase):
    def make_row(self, dataset, positive_points, complete_events):
        return {
            "dataset": dataset,
            "domain": "test",
            "method": "moving_median_lambda_mad",
            "method_name": "Moving median + raw MAD",
            "f1_score": 0.5,
            "accuracy": 0.9,
            "precision": 0.6,
            "recall": 0.7,
            "roc_auc": 0.8,
            "pr_auc": 0.7,
            "nab_style_score": 20.0,
            "event_recall": 0.5,
            "false_positive_rate": 0.1,
            "high_variance_fpr": 0.2,
            "mean_detection_delay_fraction": 0.3,
            "mean_detection_delay_seconds": 10.0,
            "mean_early_detection_credit": 0.7,
            "test_anomaly_events": complete_events,
            "test_positive_points": positive_points,
            "high_variance_normal_points": 100,
        }

    def test_negative_only_dataset_keeps_false_alarm_metrics(self):
        frame = pd.DataFrame([
            self.make_row("negative_only", 0, 0),
            self.make_row("event_dataset", 10, 1),
        ])
        result = dataset_level_means(frame).set_index("dataset")
        negative = result.loc["negative_only"]

        for column in (
            "f1_score", "precision", "recall", "roc_auc", "pr_auc",
            "event_recall", "nab_style_score",
        ):
            self.assertTrue(np.isnan(negative[column]), column)

        self.assertAlmostEqual(negative["false_positive_rate"], 0.1)
        self.assertAlmostEqual(negative["high_variance_fpr"], 0.2)

    def test_partial_event_keeps_point_metrics_but_not_event_metrics(self):
        frame = pd.DataFrame([self.make_row("partial", 5, 0)])
        result = dataset_level_means(frame).iloc[0]

        self.assertAlmostEqual(result["f1_score"], 0.5)
        self.assertFalse(np.isnan(result["pr_auc"]))
        self.assertTrue(np.isnan(result["event_recall"]))
        self.assertTrue(np.isnan(result["nab_style_score"]))
        self.assertTrue(np.isnan(result["mean_detection_delay_fraction"]))

    def test_complete_event_keeps_event_metrics(self):
        frame = pd.DataFrame([self.make_row("complete", 10, 1)])
        result = dataset_level_means(frame).iloc[0]

        self.assertAlmostEqual(result["f1_score"], 0.5)
        self.assertAlmostEqual(result["event_recall"], 0.5)
        self.assertAlmostEqual(result["nab_style_score"], 20.0)

    def test_reporting_order_contains_six_methods(self):
        self.assertEqual(len(METHOD_ORDER), 6)
        self.assertEqual(len(set(METHOD_ORDER)), 6)
        self.assertIn("moving_median_scaled_mad", METHOD_ORDER)
        self.assertIn("moving_median_iqr_scale", METHOD_ORDER)


if __name__ == "__main__":
    unittest.main()
