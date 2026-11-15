# Final benchmark status

**Authoritative experiment:** `source_code/outputs/final46_six_methods/`

- 46 unique NAB datasets
- 3 sequence lengths: 30, 50, and 100
- 3 random seeds: 15, 42, and 123
- 414 trained LSTM autoencoders
- 6 threshold methods
- 2,484 method-level test results

Evaluation populations: 18 datasets for point-level F1, 17 for complete-event NAB-style scoring, and 42 for high-variance FPR.

The earlier 23-dataset V2 experiments are retained as historical evidence. They are not the authoritative final benchmark.

# DSTT Project: Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection

## Research Project Overview

**Repository:** `dsst_project`

**Research Title:** Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

This repository contains the implementation, experimental evaluation, benchmark results, statistical analysis, and reproducibility materials for a research project investigating reconstruction-error thresholding for time-series anomaly detection using Long Short-Term Memory Autoencoders (LSTM-AEs).

The study investigates whether adaptive thresholding based on a moving median and Median Absolute Deviation (MAD) can reduce false-positive alarms while maintaining useful anomaly-detection performance.

The repository presents a **Primary Benchmark** on 23 selected real-world datasets from the Numenta Anomaly Benchmark (NAB). Of these, 17 have complete positive anomaly events in the held-out test partition; six have no positive test labels and are retained as negative-only test cases for false-alarm analysis. An earlier experiment, referred to as V1, is preserved separately.

The results are reported transparently, including findings that do not support the original research hypotheses.

## New threshold-ablation candidate (not yet full-benchmark validated)

This package also contains a new candidate implementation for investigating the low recall/F1 observed with raw MAD. It adds scaled MAD and IQR-based robust scale variants and corrects the validation fallback when a validation partition has no positive labels. The completed V2 results remain preserved as historical evidence; the new candidate has passed 22 automated tests and a one-fit CPU smoke run, but its full 207-fit GPU benchmark must still be run before any new research results can be claimed.

Use `reports/THRESHOLD_ABLATION_PROTOCOL.md` and `reports/REPRODUCIBILITY_AND_RUN_STEPS.md` for the protocol and exact commands. The full candidate run evaluates six threshold methods per model fit (1,242 method-level rows across 207 fits) and saves calibration traces for later re-analysis. New scaled-MAD and IQR results are exploratory; the existing confirmatory statistical tests remain focused on the original raw-MAD proposal and prespecified baselines.

---

## 1. Research Objectives

The main objectives of this project are:

1. Implement an LSTM Autoencoder for time-series reconstruction and anomaly scoring.
2. Compare fixed and adaptive reconstruction-error thresholding methods.
3. Evaluate false-positive rates, F1 scores, event recall, detection delay, and NAB-style scoring.
4. Investigate whether moving-median and MAD-based thresholding can reduce false alarms.
5. Evaluate the methods across multiple datasets, window sizes, and random seeds.
6. Preserve experimental outputs, statistical analyses, and verification evidence for reproducibility.

**Central research question:** Can adaptive thresholding reduce false-positive alarms while maintaining acceptable anomaly-detection performance?

---

## 2. Primary Benchmark Summary

The primary completed experiment contains:

| Experimental component | Result |
|---|---:|
| Selected NAB datasets | 23 |
| Event-evaluable test datasets | 17 |
| No-positive-test datasets | 6 |
| Window sizes | 3: 30, 50, 100 |
| Random seeds | 3: 15, 42, 123 |
| LSTM-AE model fits | 207 |
| Threshold methods per fit | 4 |
| Evaluation rows | 828 |
| Prediction files | 207 |
| Automated tests passed | 26 |

### Historical V2 benchmark (superseded by the final 46-dataset benchmark)\n\nThe total number of model fits is:

\[
23 \times 3 \times 3 = 207
\]

Each model fit is evaluated using four thresholding methods:

\[
207 \times 4 = 828
\]

The original benchmark outputs are preserved in:

`source_code/outputs/version2_normal_only_fpr05/`

The floating-point reporting correction is preserved in:

`source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/`

The clean-normal volatility reanalysis, which supersedes the original high-variance FPR analysis, is preserved in:

`source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`

The names of these folders and the configuration file retain their original development identifiers. They are intentionally unchanged so that the repository paths remain consistent with the implementation.

---

## 3. Experimental Pipeline

The implementation follows this general workflow:

1. Load NAB time-series datasets and anomaly labels.
2. Apply chronological training, validation, and test partitioning.
3. Select normal-only training windows using NAB labels.
4. Fit preprocessing and scaling using normal training data.
5. Construct time-series windows.
6. Train the LSTM Autoencoder.
7. Calculate reconstruction errors.
8. Calibrate threshold methods using validation data.
9. Apply the selected thresholds to test reconstruction errors.
10. Calculate point-level and event-level evaluation metrics.
11. Generate statistical comparisons, reports, plots, and prediction traces.

### Chronological data split

| Partition | Approximate proportion |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

The chronological test partition starts at `int(n * 0.80)`.

The rolling threshold implementation is causal: the historical observations used to calculate a threshold exclude the current reconstruction error.

---

## 4. LSTM Autoencoder Configuration

The primary experiment uses the following model settings:

| Parameter | Value |
|---|---:|
| Hidden size | 64 |
| Latent size | 32 |
| Maximum epochs | 40 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Early-stopping patience | 6 |
| Window sizes | 30, 50, 100 |
| Random seeds | 15, 42, 123 |

The experiment configuration is stored in:

`source_code/config_v2.json`

---

## 5. Thresholding Methods

Four reconstruction-error thresholding methods are compared.

### 5.1 Fixed Percentile

Implementation identifier: `fixed_percentile`

Uses a fixed percentile of reconstruction errors to determine the anomaly threshold.

### 5.2 Fixed Mean Plus Standard Deviations

Implementation identifier: `fixed_mean_k_sigma`

Uses a threshold based on the mean reconstruction error and a configurable multiple of its standard deviation.

### 5.3 Moving Median Plus MAD — Proposed Method

Implementation identifier: `moving_median_lambda_mad`

The proposed method uses a moving median and Median Absolute Deviation to adapt the reconstruction-error threshold to changing error distributions.

Its objective is to provide a more robust threshold under changing time-series conditions and unusually large reconstruction errors.

### 5.4 Rolling Mean Plus k Standard Deviations (Validation-Calibrated)

Implementation identifier: `rolling_mean_3sigma`

Uses a rolling mean and standard deviation to calculate a time-dependent threshold. The multiplier k is selected from the same candidate grid as the fixed mean-plus-k-sigma method using validation labels and the shared 5% validation-FPR target. It is not a fixed 3-sigma threshold.

The rolling calculation is causal and excludes the current reconstruction error from the historical threshold calculation.

---

## 6. Primary Benchmark Results

The completed benchmark produced the following aggregate results:

| Threshold method | Mean F1 | Mean FPR | Mean event recall | Mean NAB-style score |
|---|---:|---:|---:|---:|
| Fixed mean + kσ | 0.309673 | 0.134183 | 0.957516 | -330.948530 |
| Fixed percentile | 0.307111 | 0.138636 | 0.968410 | -368.674679 |
| Proposed moving median + MAD | 0.128007 | 0.029905 | 0.697168 | -27.367282 |
| Rolling mean + kσ (validation-calibrated) | 0.048590 | 0.019042 | 0.789760 | -23.640771 |

**Metric coverage:** Mean F1 and FPR are reported across the 23-dataset benchmark. Event recall, event delay, and NAB-style test-partition score use the 17 event-evaluable datasets. High-variance FPR uses 20 datasets with at least one normal high-variance test point. ROC-AUC and PR-AUC are defined only where both classes occur in the test partition. The saved output tables contain the underlying metric records and coverage information.

### 6.1 False-Positive Reduction

The proposed MAD method achieved a substantially lower mean false-positive rate than the two fixed baselines:

- Approximately 78.4% lower mean FPR than fixed percentile.
- Approximately 77.7% lower mean FPR than fixed mean plus kσ.

This is the principal favorable finding for the proposed method.

### 6.2 F1 and Event-Recall Trade-Off

The proposed method did not achieve the best overall anomaly-detection performance.

Its mean F1 was 0.128007, compared with:

- 0.307111 for fixed percentile.
- 0.309673 for fixed mean plus kσ.

Its mean event recall was 0.697168, compared with:

- 0.968410 for fixed percentile.
- 0.957516 for fixed mean plus kσ.

These findings indicate a trade-off between reducing false alarms and maintaining anomaly-detection coverage.

**The results do not demonstrate overall superiority of the proposed MAD method.**

### 6.3 Event-Evaluable Cohort

Seventeen datasets have positive test labels and evaluable test events.

| Method | Mean F1 | Mean FPR | Event recall | Delay fraction |
|---|---:|---:|---:|---:|
| Fixed mean + kσ | 0.418970 | 0.168924 | 0.957516 | 0.433546 |
| Fixed percentile | 0.415503 | 0.170556 | 0.968410 | 0.439675 |
| Proposed moving median + MAD | 0.173186 | 0.022602 | 0.697168 | 0.461520 |
| Rolling mean + kσ (validation-calibrated) | 0.065739 | 0.016207 | 0.789760 | 0.398610 |

The proposed method's delay fraction is higher than both fixed baselines. Therefore, these results do not support a claim that the proposed method detects events earlier.

The rolling mean method has a lower delay fraction in this cohort, but also substantially lower F1.

---

## 7. Statistical Hypothesis Evaluation

The statistical reporting evaluates three research hypotheses.

### H1 — Improvement in F1

**Outcome: Not supported.**

For the 17 event-evaluable datasets:

| Comparison | Mean F1 difference (proposed − baseline) | Wilcoxon p-value | Holm-adjusted p-value |
|---|---:|---:|---:|
| Proposed MAD vs fixed percentile | -0.242316 | 0.000015 | 0.000046 |
| Proposed MAD vs fixed mean + kσ | -0.245783 | 0.000015 | 0.000046 |

The fixed baselines performed better on all 17 paired datasets in these comparisons.

### H2 — Reduction in False-Positive Rate

**Outcome: Some dataset-level improvements observed; aggregate evidence inconclusive after Holm correction.**

| Comparison | Mean FPR difference (proposed − baseline) | Paired t-test p-value | Holm-adjusted p-value |
|---|---:|---:|---:|
| Proposed MAD vs fixed percentile | -0.147631 | 0.021253 | 0.063758 |
| Proposed MAD vs fixed mean + kσ | -0.138423 | 0.025440 | 0.063758 |
| Proposed MAD vs rolling mean + kσ | 0.041851 | 0.069577 | 0.069577 |

These re-audited comparisons use 20 paired datasets. The first two unadjusted p-values are below 0.05, but none of the Holm-adjusted results is below 0.05. The aggregate statistical evidence therefore does not establish H2 at the 0.05 significance level.

The macro dataset mean high-variance FPR is 0.223287 for fixed percentile, 0.214080 for fixed mean + kσ, 0.075657 for proposed MAD, and 0.033806 for rolling mean + kσ. MAD improves on the fixed baselines but does not outperform the rolling adaptive comparator.

### H3 — NAB-Style Score and Alert Timing

**Outcome: Descriptive only; not statistically established.**

On the 17 event-evaluable datasets, the proposed method's mean NAB-style test-partition score was -27.367, compared with -368.675 for fixed percentile, -330.949 for fixed mean + kσ, and -23.641 for rolling mean + kσ. The paired differences versus the two fixed baselines were positive, but were not statistically significant after Holm correction (adjusted p = 0.264740 and 0.191483). The proposed method did not outperform the rolling comparator (mean difference -3.727; adjusted p = 0.264740).

The proposed method also had lower event recall than the fixed baselines and a higher mean detection-delay fraction than all three comparators. Its more favorable score relative to the fixed methods is therefore a descriptive false-alarm trade-off, not evidence of overall superiority or earlier detection.

---

## 8. Dataset Selection and Methodological Limitations

The primary benchmark uses a selected cohort of 23 datasets.

The benchmark configuration retains the original 10 V1 datasets and adds 13 further datasets selected by `prepare_expanded_nab_manifest.py`. The screening rules (complete positive test event, at least 100 clean W=100 training windows, and at least 50 clean W=100 validation windows) were applied to the 13 additions, not retroactively to the original 10. Four of the original 10 have complete positive test events in the held-out test partition; six do not. Consequently, the final 23-dataset run contains **17 event-evaluable datasets and six no-positive-test datasets**. The six are retained for supplementary negative-only / false-alarm analysis, not counted as event-evaluable. The dataset-by-dataset cohort classification is available in `reports/DATASET_COHORT_AUDIT.csv` and `reports/DATASET_COHORT_AUDIT.md`.

### 8.1 Label-Informed Normal-Only Training

NAB labels are used to exclude anomaly-touching windows from training. The scaler is fitted using normal training rows only.

Consequently, this is not completely label-free training. The protocol should be described as **label-informed normal-only training**.

### 8.2 Validation-Based Threshold Calibration

Validation labels are used during threshold selection and validation-FPR-budget evaluation.

The configured 5% validation FPR is a target budget, not a guarantee. If no candidate meets the budget, the selection procedure minimizes FPR among the available candidates.

### 8.3 Test-Label-Informed Dataset Selection

Test-label properties are used to determine whether a dataset qualifies for inclusion in the benchmark cohort.

This introduces potential cohort-selection bias. Model fitting and threshold tuning do not use test labels, but dataset inclusion depends on test-label information.

Therefore, the results describe this selected cohort and should not be treated as an unbiased estimate of performance across all possible time series.

### 8.4 Datasets Without Positive Test Labels

Six selected datasets have no positive test labels:

- `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`
- `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv`
- `realTraffic/TravelTime_387.csv`
- `realTraffic/TravelTime_451.csv`
- `realTweets/Twitter_volume_AAPL.csv`
- `realTweets/Twitter_volume_AMZN.csv`

Their F1 is zero for every method because the test partition contains no positive labels. FPR can still be computed.

These datasets remain in the 23-dataset cohort and affect all-cohort F1 averages. Event-recall interpretation is based on the 17 event-evaluable datasets.

### 8.5 NAB-Style Score

The benchmark includes a custom NAB-style score calculated on the chronological test partition.

It is **not the official full-corpus NAB leaderboard score**. Its values must be interpreted alongside false-positive rate, event recall, and detection delay.

A better custom score does not independently establish that more anomalies were detected or that detection occurred earlier.

---

## 9. Verification and Reporting Integrity

The completed benchmark verification recorded:

- 207 expected prediction files; 207 present.
- 828 expected evaluation rows; 828 present.
- Zero missing prediction files.
- Zero prediction row-count mismatches.
- Zero label-count mismatches.
- Zero confusion-matrix mismatches.
- Zero ROC-AUC recomputation mismatches.
- Zero PR-AUC recomputation mismatches.
- Zero unexpected missing AUC values.
- Normal-only training checks passed for all 207 model fits.
- Zero training-window accounting errors.
- Zero training splits with zero usable windows.
- 24 complete test events and zero partial test events.

These checks support the internal numerical consistency of the saved predictions and evaluation records. They do not, by themselves, establish methodological validity or generalization to new datasets.

### Clean-normal high-variance reanalysis

The original H2 implementation used all training histories to set the high-volatility cutoff, including histories containing labeled anomalies. The reviewed code now estimates the 75th-percentile reference from full trailing windows whose historical points are all labeled normal. The current point is excluded from the volatility history.

The saved predictions were re-evaluated without retraining any models. All 828 metric rows were recalculated; only `high_variance_fpr` and `high_variance_normal_points` changed. Every other metric reproduced within an absolute difference of 4.6e-13. The reanalysis uses 20 paired datasets for H2.

### ROC-AUC and PR-AUC

ROC-AUC and PR-AUC are identical across thresholding methods for the same model fit because the methods use the same continuous reconstruction-error scores. The methods change the threshold and resulting binary predictions, not the underlying score ranking.

This is expected behavior.

### Floating-Point Reporting Correction

A numerical-tie issue was identified in pairwise directional reporting. Extremely small floating-point differences could previously be counted as strict improvements or regressions.

The reporting implementation was corrected using an absolute numerical-zero tolerance of `1e-12`. A regression test was added.

After the correction:

- 20 automated tests passed.
- 10 out of 10 report files passed the original-versus-corrected audit.
- All 8 statistical output rows were independently reproduced.
- The statistical output reproduction audit passed.
- Raw model predictions and raw benchmark metrics were not changed.

This was a reporting correction. Model training did not need to be repeated.

---

## 10. Earlier Benchmark (V1)

The package preserves V1 as a separate historical experiment.

| Component | Earlier benchmark |
|---|---:|
| NAB datasets | 10 |
| Window sizes | 3 |
| Random seeds | 3 |
| Model fits | 90 |
| Evaluation rows | 360 |
| Prediction files | 90 |

Independent recomputation of archived V1 F1, FPR, PR-AUC, and average precision matched the archived values within `1e-9`.

However, V1 training included all training windows without the primary benchmark's normal-only window filtering. Nine of the ten files had labeled anomalies in their training partitions, creating a potential training-contamination concern.

The earlier benchmark and the primary benchmark use different protocols. Their aggregate performance results must not be combined.

---

## 11. Repository Structure

| Path | Description |
|---|---|
| `README.md` | Main research project overview and findings |
| `README_FIRST.md` | Package navigation guide |
| `FINAL_PACKAGE_MANIFEST.md` | Package inventory and verification summary |
| `PROPOSAL_HYPOTHESIS_ALIGNMENT.md` | Relationship between proposal and hypotheses |
| `WEB_REFERENCES.md` | External references |
| `proposal/` | Original research proposal |
| `source_code/` | Main implementation, configuration, tests, data, and outputs |
| `source_code/config_v2.json` | Primary benchmark configuration |
| `source_code/data/nab/` | NAB datasets used in the experiment |
| `source_code/outputs/version2_normal_only_fpr05/` | Full benchmark metrics, prediction traces, tuning records, audits, and summaries |
| `source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/` | Corrected reporting tables and plots |
| `version1_reference/` | Earlier benchmark source reference |
| `verified_results/` | Archived earlier benchmark evidence |
| `reports/` | Final results audit and reproducibility instructions |
| `evaluation/` | Evaluation material |
| `analysis_tools/` | Supporting analysis utilities |
| `verification/` | Verification evidence |

For detailed scientific interpretation, see:

`reports/FINAL_RESULTS_AND_METHOD_AUDIT.md`

For reproduction instructions, see:

`reports/REPRODUCIBILITY_AND_RUN_STEPS.md`

---

## 12. Software Environment

The completed primary benchmark was run using:

| Component | Environment |
|---|---|
| Operating environment | Ubuntu under WSL2 |
| Python | 3.12.3 |
| TensorFlow | 2.21.0 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| GPU execution | CUDA-compatible TensorFlow execution; cuDNN loaded |

The working virtual environment was:

`~/venvs/lstm_tf221_gpu`

Activate it with:

```bash
source ~/venvs/lstm_tf221_gpu/bin/activate
```

Check Python and TensorFlow:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## 13. Running the Automated Tests

From the repository root:

```bash
cd source_code
python run_tests.py
```

The candidate test suite passes **22 tests**.

The tests cover data loading, NAB anomaly labels, normal-only training-window filtering, scaler fitting, LSTM Autoencoder smoke behavior, raw/scaled MAD and IQR thresholds, causal rolling threshold calculations, the no-positive-validation FPR-budget fallback, reporting, and the floating-point tie regression.

---

## 14. Reproducing the Primary Benchmark

The experiment can be reproduced using:

```bash
cd source_code

python run_experiment.py \
  --config config_v2.json \
  --output-dir outputs/threshold_ablation_v3_20261115 \
  --device cuda --skip-eda
```

A new output directory is deliberately used to avoid overwriting the archived completed benchmark. This candidate trains 207 models and evaluates six threshold methods per fit (1,242 metric rows). It may require substantial GPU resources. Run the one-fit smoke test first; see `reports/REPRODUCIBILITY_AND_RUN_STEPS.md`. The full candidate GPU run has not yet been completed, so no new candidate performance claims are made.

---

## 15. Research Conclusions

The completed primary benchmark demonstrates that moving-median plus MAD thresholding can substantially reduce mean false-positive rate relative to fixed-percentile and fixed mean-plus-kσ thresholds on the selected NAB cohort.

However, that reduction is accompanied by lower mean F1 and event recall. The proposed method also does not demonstrate an earlier-detection advantage over the fixed baselines.

The resulting hypothesis interpretation is:

- **H1:** Not supported.
- **H2:** Some dataset-level FPR improvements observed; aggregate evidence inconclusive after Holm correction.
- **H3:** Descriptive only; no inferential evidence of earlier detection.

The findings must be interpreted in light of label-informed training-window selection, validation-based threshold calibration, test-label-informed cohort selection, and the custom test-partition NAB-style score.

This repository preserves unfavorable as well as favorable results. It does not claim that the proposed method is universally superior, fully unsupervised, or production-ready.

---

## 16. Future Research Directions

Further research could investigate:

1. Improving the trade-off between false-positive reduction and event recall.
2. Evaluating threshold calibration on a development cohort separated from final test evaluation.
3. Selecting benchmark datasets without using test-label outcomes.
4. Testing generalization on additional time-series datasets.
5. Studying sensitivity to window size, random seed, and threshold hyperparameters.
6. Comparing adaptive thresholds under consistent false-positive constraints.

---

**Project:** `dsst_project`

**Research focus:** LSTM Autoencoders, adaptive thresholding, time-series anomaly detection, and statistical evaluation.
