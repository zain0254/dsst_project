# Final Benchmark — Final46

<sub>Authoritative completed experiment · 46 datasets · six threshold methods</sub>

## Purpose

This folder is the teacher-facing reading path for the completed benchmark. It explains the research design and points to the canonical evidence without duplicating the complete raw outputs.

**Single source of truth for full experiment outputs:** `source_code/outputs/final46_six_methods/`.

Do not merge V1/V2 metrics into Final46.

## Experiment at a glance

- 46 NAB time series from five domains: AWS CloudWatch (16), Twitter (10), Traffic (7), Known Cause (7), Ad Exchange (6).
- Chronological 60/20/20 train/validation/test split.
- Three window lengths: 30, 50, 100.
- Three model seeds: 15, 42, 123.
- 414 LSTM-AE fits (46 × 3 × 3).
- Six threshold methods per fit; 2,484 method-level test rows.
- Model configuration: hidden size 64, latent size 32, max epochs 40, batch size 256, learning rate 0.001, weight decay 0.00001, patience 6.
- Normal-only training windows and a 5% validation FPR calibration budget.

## Read this folder in order

1. [Final results and findings](FINAL_RESULTS_AND_FINDINGS.md)
2. [Methodology and architecture](METHODOLOGY_AND_ARCHITECTURE.md)
3. [Metrics and dataset coverage](METRICS_AND_DATASET_COVERAGE.md)
4. [Evaluation guide](EVALUATION/README.md)
5. [Audit and verification](AUDIT_AND_VERIFICATION/README.md)
6. [Reproducibility guide](REPRODUCIBILITY.md)

## Canonical output files

| Question | Canonical evidence |
|---|---|
| What was run? | `run_summary.json`, `config_used.json`, `environment.json` |
| What were test results? | `test_metrics.csv` |
| What are the dataset-level summaries? | `benchmark_method_summary.csv`, `dataset_method_summary.csv` |
| What are per-dataset comparisons? | `per_dataset_comparison.csv`, `pairwise_proposed_vs_baselines.csv` |
| What are the hypothesis tests? | `statistical_tests.csv`, `hypothesis_outcomes.csv` |
| What was calibrated? | `validation_tuning.csv`, `calibration_traces/` |
| What are the split and cohort details? | `dataset_split_audit.csv` |
| What predictions were made? | `predictions/` |
| What figures were generated? | `plots/`, `eda/` |

## Important scope notes

- The 32-dimensional latent size is the configured architecture, not a proven optimum; no latent-dimension ablation is part of this benchmark.
- Training is label-informed normal-only training: labels are used to exclude anomaly-containing training windows.
- Validation labels are used to tune threshold parameters under the common 5% FPR target.
- The NAB-style score is a custom score on complete events in held-out test segments, not the official full-corpus NAB leaderboard score.
- The final output directory includes predictions and calibration traces. Do not claim all 414 model weight checkpoints are archived unless a checkpoint inventory verifies that.
