# Final46 — Metrics and Dataset Coverage

## Why the metric populations differ

The benchmark contains 46 selected time series, but a metric is only meaningful when its required labels or observations exist.

| Population | Count | Eligibility | Appropriate use |
|---|---:|---|---|
| Full selected cohort | 46 | All selected series | Point false-positive rate and accuracy summaries |
| Positive-test cohort | 18 | At least one positive point in test | F1, precision, recall, ROC-AUC, PR-AUC |
| Complete-event cohort | 17 | At least one complete anomaly event in test | Event recall, detection delay, custom NAB-style score |
| High-variance cohort | 42 | Eligible normal high-variance test points | High-variance FPR |

Twenty-eight datasets have no positive anomaly points in their chronological test segment. They can contribute to negative-only false-alarm analysis but cannot contribute to positive-class F1 or event-detection claims. One dataset has positive test points but only a partial event, so it can contribute to point metrics but not the complete-event evaluation.

The 42 high-variance population is a separate eligibility rule; four datasets do not have eligible normal high-variance test points.

## Metric meanings

- **Precision:** among predicted anomalies, the fraction that are true anomalies.
- **Recall:** among true anomaly points, the fraction detected.
- **F1:** harmonic mean of precision and recall.
- **Accuracy:** fraction of all points classified correctly; can be misleading when anomalies are rare.
- **FPR:** fraction of normal points incorrectly flagged as anomalies.
- **ROC-AUC:** ranking discrimination across score thresholds.
- **PR-AUC / Average Precision:** precision-recall ranking performance, especially informative for rare positives.
- **Event recall:** fraction of eligible anomaly events detected.
- **Detection delay:** elapsed time from event start to the first qualifying detection under the project's event definition.
- **Detection-delay fraction:** delay normalized by event duration.
- **High-variance FPR:** false-positive rate measured on eligible normal test points within high-variability regions defined using training-derived clean-normal histories.
- **NAB-style score:** custom normalized timing reward and false-positive penalty on complete events in the held-out test partition. It is not the official full-corpus NAB leaderboard score.

ROC-AUC and PR-AUC are ranking metrics based on the continuous score stream. If all threshold methods share the same model reconstruction-error scores, changing only the threshold does not change these ranking metrics; they should therefore be identical across threshold methods for a given fit.

## Aggregation and denominators

The published method summary reports means across eligible datasets; the count columns in `benchmark_method_summary.csv` are the denominator for each metric. This is distinct from averaging all individual points or all 414 model fits. Statistical tests use one paired observation per dataset after averaging seeds and window sizes within each dataset.

Always report the metric's valid dataset count alongside the mean. Do not substitute zero for an undefined metric when a dataset has no positive class or no eligible high-variance normal points.
