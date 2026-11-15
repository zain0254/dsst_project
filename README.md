# Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

Research repository · Final46 six-method benchmark · Updated 15 November 2026

## Start here

**The authoritative completed experiment is the Final46 benchmark** in [source_code/outputs/final46_six_methods/](source_code/outputs/final46_six_methods/). V1 and V2 are retained as historical development evidence; their results must not be combined with Final46.

For a guided explanation, start with [START_HERE.md](START_HERE.md), then read the [Final Benchmark guide](FINAL_BENCHMARK/README.md).

## Final46 at a glance

| Component | Verified experiment record |
|---|---:|
| Selected NAB time series | 46 |
| Domains | 5 |
| Window sizes | 30, 50, 100 |
| Model-training seeds | 15, 42, 123 |
| LSTM Autoencoder fits | 414 |
| Threshold methods evaluated per fit | 6 |
| Method-level test metric rows | 2,484 |
| Positive-test datasets | 18 |
| Complete-event datasets | 17 |
| Datasets without positive test points | 28 |
| Datasets with eligible normal high-variance test points | 42 |

The arithmetic is 46 datasets × 3 window sizes × 3 seeds = 414 model fits, followed by 414 fits × 6 threshold methods = 2,484 method-level result rows. These are 414 trained models, not 2,484 models.

The final run summary is [run_summary.json](source_code/outputs/final46_six_methods/run_summary.json), and the exact configuration snapshot is [config_used.json](source_code/outputs/final46_six_methods/config_used.json).

## Research question

The project investigates whether adaptive thresholding of LSTM Autoencoder reconstruction errors can reduce false alarms in streaming time series while retaining useful anomaly-detection performance.

The model is trained to reconstruct historical time-series windows. Reconstruction error becomes an anomaly score. Thresholding methods then convert that continuous score into a binary normal/anomaly decision.

## Six threshold methods

| Method identifier | Description |
|---|---|
| fixed_percentile | Fixed percentile of training reconstruction errors |
| fixed_mean_k_sigma | Training mean plus a tuned multiple of standard deviation |
| rolling_mean_3sigma | Causal rolling mean plus a validation-calibrated k × rolling standard deviation; the internal name is legacy |
| moving_median_lambda_mad | Proposed moving median plus raw MAD |
| moving_median_scaled_mad | Moving median plus 1.4826 × MAD |
| moving_median_iqr_scale | Moving median plus IQR / 1.3489795 |

The original proposal's method is moving median plus raw MAD. Scaled MAD and IQR are additional exploratory ablations; they are not silently substituted for the original proposed method.

## Final benchmark results

The table below uses dataset-level summaries from benchmark_method_summary.csv. Each metric has its own eligible dataset population.

| Method | Mean F1 (n=18) | Mean FPR (n=46) | Event recall (n=17) | NAB-style score (n=17) | High-variance FPR (n=42) |
|---|---:|---:|---:|---:|---:|
| Fixed mean + k sigma | 0.4346 | 0.1423 | 0.9706 | -417.18 | 0.2469 |
| Fixed percentile | 0.4280 | 0.1425 | 0.9684 | -425.26 | 0.2449 |
| Rolling mean + k sigma | 0.1333 | 0.0501 | 0.9466 | -109.72 | 0.0615 |
| Moving median + raw MAD | 0.2256 | 0.0615 | 0.8606 | -92.52 | 0.1164 |
| Moving median + scaled MAD | 0.2421 | 0.0729 | 0.9150 | -143.71 | 0.1400 |
| Moving median + IQR scale | 0.1695 | 0.0563 | 0.8780 | -109.49 | 0.0864 |

Higher F1 and event recall are better; lower FPR is better. A higher NAB-style score is better. The NAB-style value is a custom score calculated on complete anomaly events in each held-out test partition, not an official full-corpus NAB leaderboard score.

**Main finding:** the proposed raw-MAD method reduces false-positive rates relative to the two fixed-threshold baselines, but it has substantially lower F1 and event recall. It does not outperform the rolling comparator on high-variance FPR. The evidence supports a trade-off, not overall superiority.

## Hypothesis outcomes

- **H1 — F1 improvement:** Not supported. On 18 point-evaluable datasets, raw MAD has significantly lower F1 than both fixed baselines after Holm correction.
- **H2 — high-variance false-positive reduction:** Mixed. Raw MAD has significantly lower high-variance FPR than the two fixed baselines, but significantly higher high-variance FPR than the rolling comparator.
- **H3 — event-level performance:** Descriptive improvement against the fixed baselines, but not statistically significant after Holm correction. This does not establish superiority or formal non-inferiority.

See the actual tables: [statistical_tests.csv](source_code/outputs/final46_six_methods/statistical_tests.csv), [hypothesis_outcomes.csv](source_code/outputs/final46_six_methods/hypothesis_outcomes.csv), and [research_question_results.md](source_code/outputs/final46_six_methods/research_question_results.md).

## Methodology in brief

- NAB time series and anomaly labels.
- Chronological 60% training / 20% validation / 20% test partitioning.
- Label-informed normal-only training windows.
- Scaling based on normal training data.
- Univariate LSTM Autoencoder; hidden size 64, latent size 32.
- Three sequence lengths: 30, 50, 100.
- Three model seeds: 15, 42, 123.
- Threshold hyperparameters selected on validation data with a common 5% validation FPR target.
- Test labels reserved for final evaluation metrics.
- Causal rolling thresholds use preceding reconstruction errors and exclude the current score from its own threshold history.

This is not fully label-free unsupervised learning: NAB labels are used to filter anomaly-touching training windows and validation labels are used during threshold calibration.

## Repository map

| Location | Purpose |
|---|---|
| [FINAL_BENCHMARK/](FINAL_BENCHMARK/README.md) | Teacher-facing guide to the authoritative experiment |
| [source_code/](source_code/) | Executable implementation, configuration, tests, datasets, and outputs |
| [source_code/outputs/final46_six_methods/](source_code/outputs/final46_six_methods/) | Canonical Final46 raw and derived experiment outputs |
| [HISTORICAL_EXPERIMENTS/](HISTORICAL_EXPERIMENTS/README.md) | Navigation guide to V1/V2 evidence |
| [proposal/](proposal/) | Original research proposal |
| [reports/](reports/) | Historical audits, change log, cohort documentation, and development protocol |
| [evaluation/](evaluation/) | Historical V1 evaluation package; not the Final46 workbook |
| [version1_reference/](version1_reference/) | Historical V1 source reference |
| [verified_results/](verified_results/) | Archived V1 experimental evidence |

## Reproducibility

The final benchmark's exact configuration snapshot is stored with its results. To run a fresh reproduction, activate the recorded WSL environment and use a **new output directory**. This retrains 414 models and is not required merely to inspect existing results.

From the repository's source_code directory:

~~~bash
source ~/venvs/lstm_tf221_gpu/bin/activate
python run_tests.py

python run_experiment.py \
  --config outputs/final46_six_methods/config_used.json \
  --output-dir outputs/final46_reproduction_run \
  --device cuda --skip-eda
~~~

Do not use outputs/final46_six_methods as the output directory for a new run. Preserve the committed benchmark unchanged. See [Final Benchmark reproducibility instructions](FINAL_BENCHMARK/REPRODUCIBILITY.md).

## Important research limitations

1. Training is label-informed normal-only rather than fully label-free.
2. Validation labels participate in threshold calibration.
3. Cohort selection and test-label eligibility must be described transparently; 28 selected datasets have no positive test points in their chronological test partition.
4. H1 is not supported; H2 is comparator-dependent; H3 is not statistically established.
5. The NAB-style score is limited to complete events in held-out test partitions and is not an official NAB leaderboard score.
6. The 32-dimensional latent representation was a configured design choice; a latent-dimension ablation has not established it as optimal.
7. The Final46 output directory contains predictions and calibration traces. Do not claim that all 414 trained weight checkpoints are archived unless separately verified.

## Historical experiments

V1 and V2 are retained because they document the project's development. V1 used 10 datasets, 90 model fits, and four methods. V2 used 23 datasets, 207 fits, and four methods. Final46 is the authoritative completed six-method benchmark. See [HISTORICAL_EXPERIMENTS/README.md](HISTORICAL_EXPERIMENTS/README.md).

## Citation and use

The project draws its time-series data and anomaly labels from the Numenta Anomaly Benchmark. Consult [WEB_REFERENCES.md](WEB_REFERENCES.md) and the official NAB repository for dataset attribution and licensing details.