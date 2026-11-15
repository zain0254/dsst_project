# Final46 Evaluation Package

## Authoritative sources

The complete benchmark output is stored once, in:

`source_code/outputs/final46_six_methods/`

This folder is the source of truth. This evaluation guide points to it; it does not duplicate raw predictions or traces.

## Main tables

| Purpose | Source |
|---|---|
| Run and cohort metadata | `run_summary.json` |
| Exact configuration snapshot | `config_used.json` |
| Per-fit/per-method test metrics | `test_metrics.csv` |
| Dataset-macro method summary and metric-specific counts | `benchmark_method_summary.csv` |
| Dataset and method aggregation | `dataset_method_summary.csv` |
| Per-dataset comparisons | `per_dataset_comparison.csv` |
| Proposed method versus baselines | `pairwise_proposed_vs_baselines.csv` |
| Statistical tests and Holm correction | `statistical_tests.csv` |
| H1/H2/H3 outcomes | `hypothesis_outcomes.csv` |
| Validation calibration | `validation_tuning.csv` |
| Dataset split and eligibility | `dataset_split_audit.csv` |
| High-variance FPR by dataset | `high_variance_fpr_by_dataset.csv` |
| Window-size comparisons | `comparison_by_window_size.csv` |
| Domain comparisons | `domain_comparison.csv`, `domain_improvement_summary.csv` |
| Seed variability | `seed_variability.csv` |

## Final46 workbook

A separate review workbook has now been generated from the canonical Final46 dataset-level CSV outputs and is supplied as **`FINAL46_EVALUATION_MATRIX.xlsx`** with this delivery. It is not yet committed to the repository.

It contains these worksheets:

1. Executive Summary
2. Overall Metrics
3. Dataset-wise F1 (18 positive-test datasets)
4. Dataset-wise FPR (all 46 datasets)
5. Event Evaluation (17 complete-event datasets)
6. High-Variance FPR (42 eligible datasets)
7. Dataset Coverage
8. Statistical Tests
9. Hypothesis Decisions
10. Method Definitions

To add the workbook to a local clone, place it in this folder as:

`FINAL_BENCHMARK/EVALUATION/FINAL46_EVALUATION_MATRIX.xlsx`

Then review and commit it normally. The canonical CSV outputs remain the source of truth; the workbook is a reader-friendly presentation layer and does not replace the per-fit result rows, predictions, or calibration traces.

The existing root workbook `evaluation/LSTM_AE_EVALUATION_MATRIX.xlsx` belongs to historical V1. Keep it separate and do not rename it as Final46.

## Statistical reporting note

H1 has two valid adjusted p-values in different analysis families: **0.0003815** for proposal-specific H1's two fixed-threshold comparisons, and **0.0005722** in the broader RQ2 table that additionally includes the rolling comparator. These use different Holm correction families; see `FINAL_RESULTS_AND_FINDINGS.md` for the explanation.
