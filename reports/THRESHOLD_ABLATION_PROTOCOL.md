> **Historical protocol notice:** This document records the threshold-ablation candidate protocol that preceded the completed Final46 benchmark. The candidate was subsequently evaluated in Final46. Use `source_code/outputs/final46_six_methods/` and `FINAL_BENCHMARK/FINAL_RESULTS_AND_FINDINGS.md` for the completed results. The candidate-stage statement below that the full benchmark was still pending is historical and must not be read as current status.

# Threshold Ablation Candidate — Protocol and Interpretation

**Status:** Candidate implementation awaiting the full GPU benchmark. No new full-benchmark F1, NAB-style, or event metrics are claimed in this document.

## Purpose

The prior benchmark found a large F1/recall reduction for the moving-median + raw-MAD method relative to fixed thresholds. The validation-tuning audit also found that 117 of 207 model runs had no positive anomaly labels in validation. In those runs, maximizing F1 provides no useful signal; the old common selector instead preferred the candidate with the lowest validation FPR, even when a less conservative candidate could remain within the predeclared 5% FPR budget.

This candidate makes two controlled changes:

1. It adds two robust scale estimators so the effect of the raw MAD scale convention can be measured.
2. It changes the no-positive-validation fallback for **all methods**: among candidates within the existing validation FPR budget, select the candidate whose FPR is closest to the budget from below. If no candidate meets the budget, select the one with the lowest FPR. This avoids optimizing an uninformative zero-recall objective by minimizing FPR without limit.

The normal-positive validation case retains the prior objective: maximize validation F1 among candidates within the FPR budget; if none qualify, minimize FPR first.

## Methods in the candidate run

| Method | Threshold scale |
|---|---|
| Fixed percentile | Training reconstruction-error percentile |
| Fixed mean + k sigma | Training mean plus k standard deviations |
| Rolling mean + k sigma | Causal rolling mean plus k rolling standard deviations |
| Moving median + raw MAD | Median plus lambda times unscaled MAD |
| Moving median + scaled MAD | Median plus k times 1.4826 × MAD |
| Moving median + IQR scale | Median plus k × IQR / 1.3489795 |

All rolling methods test the current reconstruction error before appending it to the rolling history. The history is causal. Robust scale floors are derived from training reconstruction errors only, using an IQR-based scale (with MAD fallback if the IQR is zero); validation/test spikes cannot inflate this floor. The rolling median/scale history itself remains causal and is initialized from the preceding train-plus-validation stream.

## Calibration and leakage controls

- Training, validation, and test partitions remain chronological (60/20/20).
- The LSTM-AE is trained once per dataset/window/seed; all six threshold methods reuse that model's reconstruction errors.
- Training windows remain normal-only under the label-informed V2 protocol.
- Threshold parameters are selected using validation errors and validation labels only.
- Test labels are used only for final metric calculation.
- The 5% validation FPR budget remains fixed across methods.
- No test metric, including F1, event recall, FPR, or NAB-style score, is used to select a parameter.

## Expected run size

The full configuration has 23 datasets × 3 window sizes × 3 seeds = 207 LSTM-AE fits. Each fit is evaluated with six threshold methods, producing 1,242 method-level metric rows. This is still 207 model fits, not 1,242 model fits.

## Output files

The recommended output directory is `source_code/outputs/threshold_ablation_v3_20261115/`.

- `test_metrics.csv`: all method-level test metrics.
- `validation_tuning.csv`: selected parameters and validation metrics.
- `dataset_split_audit.csv`: split and label-coverage audit.
- `threshold_ablation_summary.csv`: macro-average summary after averaging within datasets first; descriptive, not a significance test.
- `method_summary.csv` and `dataset_method_summary.csv`: standard summaries.
- `predictions/`: timestamp-aligned test errors, labels, thresholds, and predictions.
- `calibration_traces/`: compressed train, validation, and test reconstruction-error arrays; enables threshold re-analysis without retraining.
- `statistical_tests.csv` and `hypothesis_outcomes.csv`: existing confirmatory hypothesis comparisons for the original raw-MAD proposal against the prespecified baselines. New scaled-MAD and IQR variants are exploratory and are not silently added to those confirmatory hypotheses.

## How to interpret the result

This is a diagnostic and ablation study, not a score-optimization exercise. A higher F1 or NAB-style score is not guaranteed. The corrected no-positive-validation selector may lower thresholds and increase recall, but it may also increase test false alarms. Scaled MAD and IQR scale may produce different thresholds because their scale units differ from raw MAD.

Compare at least: F1, precision, recall, point FPR, event recall, detection delay, NAB-style test score, high-variance FPR, PR-AUC, ROC-AUC, selected parameter distributions, and validation-FPR-budget compliance. Keep the 17 event-evaluable datasets separate from the six no-positive-test datasets for event-level and NAB-style analyses.

The project's NAB-style score remains a test-partition score on complete events, not the official full-corpus NAB leaderboard score. Do not compare its numeric values directly with official NAB leaderboard results.

## Reproducibility

Run the tests first, then a one-fit smoke run, then the full GPU run using the commands in `REPRODUCIBILITY_AND_RUN_STEPS.md`. Preserve the previous completed benchmark directories; do not overwrite them. Keep this candidate's output directory separate so old and new protocols can be compared transparently.
