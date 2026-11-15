# Start Here — Final46 Research Repository

**Research:** Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

## Which experiment is final?

The authoritative completed experiment is **Final46**: 46 NAB datasets, 414 LSTM Autoencoder fits, six threshold methods, and 2,484 method-level test result rows.

Canonical experiment evidence: `source_code/outputs/final46_six_methods/`.

V1 and V2 are historical experiments. Never combine their rows or aggregate metrics with Final46.

## Recommended reading order

1. [Root README](README.md) — short overview and principal results.
2. [Final Benchmark guide](FINAL_BENCHMARK/README.md) — complete guided tour.
3. [Final results and findings](FINAL_BENCHMARK/FINAL_RESULTS_AND_FINDINGS.md) — metric summaries and hypotheses.
4. [Methodology and architecture](FINAL_BENCHMARK/METHODOLOGY_AND_ARCHITECTURE.md) — model and data pipeline.
5. [Metrics and dataset coverage](FINAL_BENCHMARK/METRICS_AND_DATASET_COVERAGE.md) — eligibility and denominators.
6. [Evaluation guide](FINAL_BENCHMARK/EVALUATION/README.md) — source tables and workbook status.
7. [Audit and verification](FINAL_BENCHMARK/AUDIT_AND_VERIFICATION/README.md) — limitations and evidence checks.
8. [Reproducibility](FINAL_BENCHMARK/REPRODUCIBILITY.md) — safe inspection and fresh-run instructions.
9. [Historical experiments](HISTORICAL_EXPERIMENTS/README.md) — V1/V2 context.

## Final46 counts

| Item | Count |
|---|---:|
| Datasets | 46 |
| Domains | 5 |
| Window lengths | 30, 50, 100 |
| Seeds | 15, 42, 123 |
| Model fits | 414 |
| Threshold methods | 6 |
| Method-level metric rows | 2,484 |
| Positive-test datasets | 18 |
| Complete-event datasets | 17 |
| Negative-only test datasets | 28 |
| High-variance FPR-evaluable datasets | 42 |

## Important interpretation

The proposed raw-MAD method lowers high-variance FPR relative to the two fixed baselines, but has lower F1 and event recall. It has higher high-variance FPR than the rolling comparator. H1 is not supported, H2 is mixed, and H3 is not statistically established after Holm correction.

The root `evaluation/` folder contains the historical V1 evaluation workbook. Do not present it as the Final46 evaluation workbook.
