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

## Workbook status

The existing root workbook `evaluation/LSTM_AE_EVALUATION_MATRIX.xlsx` belongs to the historical V1 evaluation package. It is not a Final46 workbook and must not be described as one.

A separate Final46 workbook should be generated from the canonical Final46 CSVs. It should include: Executive Summary, Overall Metrics, Dataset-wise F1, Dataset-wise FPR, Event Evaluation, NAB-style Scores, High-Variance FPR, Statistical Tests, Hypothesis Decisions, Dataset Coverage, and Method Definitions.

Do not copy V1 workbook values into a new workbook. Do not create a Final46 workbook until the source tables and statistical-summary discrepancy noted in `FINAL_RESULTS_AND_FINDINGS.md` have been reconciled.
