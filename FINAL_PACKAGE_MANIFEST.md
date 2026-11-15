# Final Package Manifest — Final46

## Authoritative experiment

**Final46 — Six-method LSTM Autoencoder benchmark**

Canonical output directory: **source_code/outputs/final46_six_methods/**

| Component | Verified record |
|---|---:|
| Selected NAB datasets | 46 |
| NAB domains | 5 |
| Sequence lengths | 30, 50, 100 |
| Model-training seeds | 15, 42, 123 |
| LSTM Autoencoder model fits | 414 |
| Threshold methods per fit | 6 |
| Method-level test metric rows | 2,484 |
| Positive-test datasets | 18 |
| Complete-event datasets | 17 |
| Datasets without positive test points | 28 |
| High-variance FPR-evaluable datasets | 42 |

The run summary and saved configuration are the primary experiment metadata:
- source_code/outputs/final46_six_methods/run_summary.json
- source_code/outputs/final46_six_methods/config_used.json
- source_code/outputs/final46_six_methods/environment.json

## 1. Repository purpose

This package contains the original proposal, implementation, NAB data, Final46 experiment outputs, statistical reports, automated tests, verification material, and historical V1/V2 evidence for:

**Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series**

## 2. Final46 configuration

| Parameter | Value |
|---|---:|
| Train / validation / test | 60% / 20% / 20%, chronological |
| Hidden size | 64 |
| Latent size | 32 |
| Maximum epochs | 40 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Early-stopping patience | 6 |
| Sequence lengths | 30, 50, 100 |
| Model seeds | 15, 42, 123 |
| Normal-only training windows | Enabled |
| Validation FPR target | 0.05 |
| High-variance reference percentile | 75th percentile of clean-normal training histories |

The latent size of 32 is a configured architecture choice. The Final46 experiment does not include a separate latent-dimension ablation proving that 32 is optimal.

## 3. Final46 threshold methods

1. Fixed percentile.
2. Fixed mean plus k standard deviations.
3. Causal rolling mean plus validation-calibrated k standard deviations (legacy internal identifier: rolling_mean_3sigma).
4. Moving median plus raw MAD (original proposed method).
5. Moving median plus scaled MAD (1.4826 × MAD).
6. Moving median plus IQR scale (IQR / 1.3489795).

The raw-MAD method is the method tested against the original proposal's fixed baselines in H1. Scaled MAD and IQR are additional exploratory ablations.

## 4. Final results source files

| File | Role |
|---|---|
| run_summary.json | Dataset, fit, row, method, and evaluation-population metadata |
| config_used.json | Configuration snapshot stored with the benchmark |
| environment.json | Recorded runtime environment |
| test_metrics.csv | Method-level held-out test metrics |
| benchmark_method_summary.csv | Dataset-level summary with metric-specific counts |
| threshold_ablation_summary.csv | Dataset-macro descriptive six-method comparison |
| dataset_method_summary.csv | Dataset/method summaries |
| per_dataset_comparison.csv | Per-dataset method comparisons |
| pairwise_proposed_vs_baselines.csv | Pairwise proposed-method comparisons |
| statistical_tests.csv | Prespecified H1/H2/H3 comparisons and Holm correction |
| hypothesis_outcomes.csv | Final hypothesis decisions |
| research_question_results.md | Generated research-question summary |
| validation_tuning.csv | Selected threshold parameters and validation metrics |
| dataset_split_audit.csv | Split, label, and eligibility audit |
| high_variance_fpr_by_dataset.csv | High-variance false-positive comparisons |
| comparison_by_window_size.csv | Window-size analysis |
| domain_comparison.csv and domain_improvement_summary.csv | Exploratory domain analysis |
| seed_variability.csv | Seed-variability evidence |
| predictions/ | Timestamp-aligned predictions and thresholds |
| calibration_traces/ | Saved reconstruction-error traces for threshold re-analysis |
| plots/ and eda/ | Figures and exploratory summaries |

## 5. Evaluation populations

| Population | Number of datasets | Use |
|---|---:|---|
| Full selected cohort | 46 | Overall point false-alarm analysis |
| Positive-test cohort | 18 | Point-level F1, precision, recall, ROC-AUC, PR-AUC |
| Complete-event cohort | 17 | Event recall, delay, NAB-style score |
| High-variance normal-test cohort | 42 | High-variance FPR |

Twenty-eight datasets have no positive anomaly points in their held-out test partition. They remain useful as negative-only cases for false-alarm analysis; they cannot establish positive-event detection performance. One additional dataset has positive test points but no complete test event, so the positive-test and complete-event populations differ.

## 6. Main interpretation

- H1 is not supported: raw MAD has significantly lower F1 than both fixed baselines on the 18 positive-test datasets after Holm correction.
- H2 is mixed: raw MAD has significantly lower high-variance FPR than the fixed baselines, but significantly higher high-variance FPR than the rolling comparator.
- H3 is descriptive only: raw MAD has higher mean NAB-style test scores than the two fixed baselines, but the paired differences are not statistically significant after Holm correction. It does not establish superiority or formal non-inferiority.

The NAB-style score is a custom score on complete anomaly events in held-out test partitions, not the official full-corpus NAB leaderboard score.

## 7. Historical experiments

Historical evidence is preserved and must not be combined with Final46.

| Version | Cohort | Fits | Methods | Result rows | Status |
|---|---:|---:|---:|---:|---|
| V1 | 10 datasets | 90 | 4 | 360 | Historical original experiment |
| V2 | 23 datasets | 207 | 4 | 828 | Historical normal-only benchmark |
| Final46 | 46 datasets | 414 | 6 | 2,484 | Authoritative completed benchmark |

See HISTORICAL_EXPERIMENTS/README.md for the original paths and context.

## 8. Reproducibility and tests

Use the instructions in FINAL_BENCHMARK/REPRODUCIBILITY.md. A full reproduction trains 414 models and must use a new output directory. Merely inspecting saved results does not require retraining.

Run the automated tests from source_code/ with python run_tests.py. The latest development session reported 26/26 passing; rerun on the checkout being reviewed and record the actual output before academic submission.

## 9. Important limitations

- Normal-only training uses NAB labels to filter training windows; it is label-informed, not fully label-free.
- Validation labels are used to select threshold parameters under a 5% validation FPR target.
- Dataset selection and metric eligibility should be fully disclosed.
- The Final46 result directory includes predictions and calibration traces; the directory inventory does not show a saved models/ checkpoint folder. Do not claim all 414 weight checkpoints are archived without verifying another location.
- The original proposal's raw-MAD method did not outperform the fixed F1 baselines.
- Domain analyses are exploratory and have small sample sizes.
- The 32-dimensional latent setting has not been established as optimal by an ablation.

## 10. Repository navigation

- README.md — short overview.
- START_HERE.md — recommended reading order.
- FINAL_BENCHMARK/ — teacher-facing Final46 guide and report.
- source_code/ — implementation, tests, data, and canonical outputs.
- HISTORICAL_EXPERIMENTS/ — historical-version navigation.
- proposal/ — original proposal.
- reports/ — audit history and supporting documentation.
- evaluation/ — historical V1 evaluation package.