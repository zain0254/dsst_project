# Start Here — Final46 Research Repository

Welcome. This repository contains the implementation and experimental evidence for:

**Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series**

## Which experiment is final?

The authoritative completed experiment is:

**Final46 — 46 NAB datasets, 414 LSTM Autoencoder fits, six threshold methods, 2,484 method-level test result rows.**

Canonical result directory:

**source_code/outputs/final46_six_methods/**

V1 and V2 are historical experiments. Their metrics must not be combined with Final46.

## Recommended reading order for a supervisor or reviewer

1. [README.md](README.md) — short repository overview and principal results.
2. [FINAL_BENCHMARK/README.md](FINAL_BENCHMARK/README.md) — guided explanation of the completed experiment.
3. [FINAL_BENCHMARK/FINAL_RESULTS_AND_FINDINGS.md](FINAL_BENCHMARK/FINAL_RESULTS_AND_FINDINGS.md) — actual results and H1/H2/H3 conclusions.
4. [FINAL_BENCHMARK/METHODOLOGY_AND_ARCHITECTURE.md](FINAL_BENCHMARK/METHODOLOGY_AND_ARCHITECTURE.md) — model, data pipeline, six thresholds, and calibration.
5. [FINAL_BENCHMARK/METRICS_AND_DATASET_COVERAGE.md](FINAL_BENCHMARK/METRICS_AND_DATASET_COVERAGE.md) — metric definitions and eligible populations.
6. [FINAL_BENCHMARK/EVALUATION/README.md](FINAL_BENCHMARK/EVALUATION/README.md) — evaluation workbook and source tables.
7. [FINAL_BENCHMARK/AUDIT_AND_VERIFICATION/README.md](FINAL_BENCHMARK/AUDIT_AND_VERIFICATION/README.md) — evidence, checks, and known limitations.
8. [FINAL_BENCHMARK/REPRODUCIBILITY.md](FINAL_BENCHMARK/REPRODUCIBILITY.md) — how to inspect, test, and reproduce without overwriting evidence.
9. [HISTORICAL_EXPERIMENTS/README.md](HISTORICAL_EXPERIMENTS/README.md) — V1 and V2 development history.

## Important numbers

| Item | Final46 value |
|---|---:|
| NAB datasets | 46 |
| Domains | 5 |
| Sequence lengths | 30, 50, 100 |
| Seeds | 15, 42, 123 |
| Trained LSTM-AE models | 414 |
| Threshold methods | 6 |
| Method-level result rows | 2,484 |
| Positive-test datasets | 18 |
| Complete-event datasets | 17 |
| No-positive-test datasets | 28 |
| High-variance FPR-evaluable datasets | 42 |

## How the result should be interpreted

The original raw-MAD method lowers false-positive rates against fixed thresholds but sacrifices F1 and event recall. It is not the overall winner: the rolling adaptive comparator has lower high-variance FPR, while fixed methods have higher F1 and event recall.

The hypotheses are reported honestly:
- H1: not supported.
- H2: mixed.
- H3: descriptively favorable against fixed baselines, but not statistically established after Holm correction.

## Where is the implementation?

- Runner: source_code/run_experiment.py
- TensorFlow architecture: source_code/src/model_tf.py
- Data preparation: source_code/src/data_utils.py
- Threshold implementations: source_code/src/thresholds.py
- Metrics: source_code/src/metrics.py
- NAB-style scoring: source_code/src/nab_scoring.py
- Reporting and statistics: source_code/src/benchmark_reporting.py
- Tests: source_code/tests/

## What is historical?

The root evaluation/ folder is the earlier V1 evaluation package. The V2 result directories and old reports are preserved under their original paths. They are documented in [HISTORICAL_EXPERIMENTS/README.md](HISTORICAL_EXPERIMENTS/README.md).

**Do not use the V1 Excel workbook or old V2 tables as the Final46 evaluation workbook.**