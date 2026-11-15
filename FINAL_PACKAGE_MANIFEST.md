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

# Final Submission Package Manifest

## 1. Project Identification

**Repository:** `dsst_project`

**Research Title:** Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

**Primary Dataset Source:** Numenta Anomaly Benchmark (NAB)

This package contains the research proposal, implementation, experiment configuration, datasets, completed benchmark outputs, statistical reports, automated tests, verification evidence, and reproducibility instructions.

The primary completed experiment is referred to as the **Primary Benchmark** throughout the project documentation. Historical V1 material is preserved separately.

---

## 2. Overall Package Status

The package contains a completed primary benchmark with the following experimental record:

| Component | Verified project record |
|---|---:|
| Selected primary benchmark datasets | 23 |
| Event-evaluable test datasets | 17 |
| No-positive-test datasets | 6 |
| Window sizes | 3: 30, 50, 100 |
| Random seeds | 3: 15, 42, 123 |
| LSTM Autoencoder model fits | 207 |
| Threshold methods per fit | 4 |
| Evaluation rows | 828 |
| Prediction files | 207 |
| Automated tests passed | 20 |
| Complete labeled test events | 24 |
| Partial test events | 0 |

The earlier, historical V2 benchmark's 207 model fits consist of:

\[
23 \times 3 \times 3 = 207
\]

The four thresholding methods produce:

\[
207 \times 4 = 828
\]

method-level evaluation records.

The original benchmark evidence is preserved in:

`source_code/outputs/version2_normal_only_fpr05/`

The clean-normal high-variance reanalysis is preserved in:

`source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`

The corrected reporting outputs are preserved in:

`source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/`

The authoritative H2 high-variance analysis is in `source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`.

---

## 3. Package Inventory

| Path | Description |
|---|---|
| `README.md` | Main research overview, results, methodology, and conclusions |
| `README_FIRST.md` | Package navigation and recommended reading order |
| `FINAL_PACKAGE_MANIFEST.md` | This package inventory and verification summary |
| `PROPOSAL_HYPOTHESIS_ALIGNMENT.md` | Alignment between the original proposal and implemented hypotheses |
| `WEB_REFERENCES.md` | External references used by the project |
| `proposal/` | Original research proposal |
| `source_code/` | Main implementation, configuration, tests, datasets, and experiment outputs |
| `source_code/config_v2.json` | Primary benchmark experiment configuration |
| `source_code/data/nab/` | NAB datasets used in the benchmark |
| `source_code/outputs/version2_normal_only_fpr05/` | Original benchmark metrics, prediction traces, tuning records, audits, summaries, and plots |
| `source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/` | Floating-point corrected reporting tables and plots |
| `source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/` | Clean-normal H2 reanalysis, updated metrics, statistical tables, plots, and audit; reuses original predictions without duplicating them |
| `analysis_tools/recompute_v2_normal_volatility_metrics.py` | Recomputes H2 high-variance metrics from saved predictions without model retraining |
| `version1_reference/` | Historical V1 source-code reference |
| `verified_results/` | Archived V1 benchmark evidence |
| `reports/DATASET_COHORT_AUDIT.md` and `.csv` | Dataset-by-dataset eligibility and test-label coverage audit |
| `reports/REVIEW_BUILD_CHANGELOG.md` | Review corrections and verification record |
| `reports/` | Detailed results audit and reproducibility documentation |
| `evaluation/` | Evaluation material |
| `analysis_tools/` | Supporting analysis utilities |
| `verification/` | Verification evidence |

---

## 4. Primary Benchmark Configuration

The primary benchmark uses the following experimental configuration:

| Parameter | Value |
|---|---:|
| Chronological training partition | 60% |
| Chronological validation partition | 20% |
| Chronological test partition | 20% |
| Window sizes | 30, 50, 100 |
| Random seeds | 15, 42, 123 |
| Hidden size | 64 |
| Latent size | 32 |
| Maximum epochs | 40 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Early-stopping patience | 6 |
| Normal-only training | Enabled |
| Validation FPR target budget | 0.05 |

The configuration is stored in:

`source_code/config_v2.json`

### Dataset Eligibility

The implemented eligibility conditions are:

1. The chronological test partition begins at `int(n * 0.80)`.
2. At least one complete labeled anomaly event lies wholly within the test partition.
3. The test partition contains positive anomaly labels.
4. At least 100 normal-only training windows are available at window size 100.
5. At least 50 clean validation windows are available.

The 13 additional datasets passed the screening conditions. The original 10 V1 datasets were retained without reapplying that screen: four have complete positive test events and six have no positive test labels. The final run therefore contains 17 event-evaluable datasets and six no-positive-test datasets. The six are retained as negative-only test cases for supplementary false-alarm analysis; do not describe all 23 as event-eligible.

---

## 5. Thresholding Methods

The experiment evaluates four methods:

| Method identifier | Description |
|---|---|
| `fixed_percentile` | Fixed reconstruction-error percentile threshold |
| `fixed_mean_k_sigma` | Fixed mean-plus-k-standard-deviations threshold |
| `moving_median_lambda_mad` | Proposed moving-median plus MAD threshold |
| `rolling_mean_3sigma` | Validation-calibrated causal rolling mean + k standard deviations (legacy internal identifier) |

The LSTM Autoencoder generates reconstruction errors. The thresholding methods convert those continuous anomaly scores into binary anomaly predictions.

---

## 6. Primary Benchmark Results

| Threshold method | Mean F1 | Mean FPR | Mean event recall | Mean NAB-style score |
|---|---:|---:|---:|---:|
| Fixed mean + kσ | 0.309673 | 0.134183 | 0.957516 | -330.948530 |
| Fixed percentile | 0.307111 | 0.138636 | 0.968410 | -368.674679 |
| Proposed moving median + MAD | 0.128007 | 0.029905 | 0.697168 | -27.367282 |
| Rolling mean + kσ (validation-calibrated) | 0.048590 | 0.019042 | 0.789760 | -23.640771 |

Mean F1 and FPR are reported across the 23-dataset benchmark. Event-recall interpretation uses the 17 event-evaluable datasets because six selected datasets have no positive test labels.

### Main Finding

The proposed MAD method reduced mean FPR by approximately:

- 78.4% relative to fixed percentile.
- 77.7% relative to fixed mean plus kσ.

However, its mean F1 and event recall were substantially lower than those of both fixed baselines.

The results indicate a trade-off between reducing false positives and maintaining anomaly-detection performance. They do not establish overall superiority of the proposed method.

---

## 7. Hypothesis Outcomes

### H1 — F1 Improvement

**Not supported.**

On the 17 event-evaluable datasets:

- Proposed MAD versus fixed percentile: mean difference -0.242316, Wilcoxon p = 0.000015, Holm-adjusted p = 0.000046.
- Proposed MAD versus fixed mean plus kσ: mean difference -0.245783, Wilcoxon p = 0.000015, Holm-adjusted p = 0.000046.

The fixed baselines performed better on all 17 paired datasets in both comparisons.

### H2 — False-Positive Reduction

**Some dataset-level improvements observed; aggregate evidence inconclusive after Holm correction.**

After the clean-normal volatility reanalysis, H2 uses 20 paired datasets. Holm-adjusted p-values are 0.063758 versus fixed percentile, 0.063758 versus fixed mean + kσ, and 0.069577 versus rolling mean + kσ. The unadjusted p-values versus the fixed baselines are below 0.05, but none of the adjusted comparisons reaches the 0.05 threshold.

### H3 — Earlier Detection

**Descriptive only.**

No inferential test establishes an earlier-detection advantage for the proposed method. Its mean delay fraction was higher than both fixed baselines in the event-evaluable cohort.

---

## 8. Verification Records

The independent prediction and metric integrity audit recorded:

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

The event audit recorded:

- 24 complete test events.
- Zero partial test events.
- Zero test points inside partial events.

These checks support the internal numerical consistency of saved predictions and evaluation records. They do not independently establish methodological validity or generalization to unseen datasets.

---

## 9. Statistical Reporting Correction

A floating-point tie-handling issue was identified in pairwise directional reporting. Extremely small numerical differences could previously be counted as strict improvements or regressions.

The reporting implementation was corrected using an absolute numerical-zero tolerance of `1e-12`.

A regression test was added to verify the behavior.

After the correction:

- 20 automated tests passed.
- 10 of 10 report files passed the original-versus-corrected audit.
- All 8 statistical output rows were independently reproduced.
- The statistical output reproduction audit passed.
- No changes were made to model training, saved predictions, or raw benchmark metrics.

Corrected reports and plots are stored in:

`source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/`

The authoritative H2 high-variance analysis is in `source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`.

The original benchmark outputs remain preserved in their separate directory.

---

## 10. Methodological Disclosures

The primary benchmark must not be described as fully label-free unsupervised learning.

The experimental protocol uses labels in the following ways:

1. NAB labels exclude anomaly-touching windows from training.
2. The scaler is fitted on normal training rows only.
3. Validation labels are used during threshold calibration.
4. Test-label properties are used in eligible-dataset selection.

The use of test-label properties for cohort selection introduces potential selection bias. Results should be interpreted as conditional on the selected eligible cohort.

The configured 5% validation FPR is a target budget, not a guaranteed upper bound. If no candidate satisfies it, the tuner minimizes FPR among the available candidates.

### Datasets Without Positive Test Labels

Six selected datasets have no positive test labels:

- `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`
- `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv`
- `realTraffic/TravelTime_387.csv`
- `realTraffic/TravelTime_451.csv`
- `realTweets/Twitter_volume_AAPL.csv`
- `realTweets/Twitter_volume_AMZN.csv`

Their F1 is zero for all methods because their test partitions have no positive labels. They remain in the 23-dataset cohort and affect all-cohort F1 averages. Event recall is interpreted using the remaining 17 event-evaluable datasets.

### NAB-Style Score

The reported NAB-style score is a custom score calculated on the chronological test partition.

It is not the official full-corpus NAB leaderboard score and must be described accordingly.

---

## 11. Earlier Benchmark (V1)

The earlier experiment is preserved separately:

| Component | V1 result |
|---|---:|
| NAB datasets | 10 |
| Window sizes | 3 |
| Random seeds | 3 |
| Model fits | 90 |
| Evaluation rows | 360 |
| Prediction files | 90 |

Independent recomputation of archived V1 F1, FPR, PR-AUC, and average precision matched the archived values within `1e-9`.

V1 training included all training windows without normal-only filtering. Nine of the ten files had labeled anomalies in their training partitions, creating a potential training-contamination concern.

V1 and the primary benchmark have different protocols. Their aggregate metrics must not be combined.

V1 evidence is preserved in:

`verified_results/V1_full_gpu_benchmark_restart/`

The historical source code is preserved in:

`version1_reference/`

---

## 12. Software Environment

The completed primary benchmark used:

| Component | Environment |
|---|---|
| Operating environment | Ubuntu under WSL2 |
| Python | 3.12.3 |
| TensorFlow | 2.21.0 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| GPU execution | CUDA-compatible TensorFlow execution; cuDNN loaded |

The working virtual environment was:

`~/venvs/lstm_tf221_gpu`

The automated test command is:

```bash
cd source_code
python run_tests.py
```

The current threshold-ablation candidate test suite passes 22 tests. The archived V2 benchmark report recorded 20 tests at the time of its original audit.

---

## 13. Reproduction Instructions

To reproduce the primary benchmark without overwriting the archived outputs:

```bash
cd source_code

python run_experiment.py \
  --config config_v2.json \
  --output-dir outputs/primary_benchmark_reproduction \
  --device cuda
```

This command trains the models again and may require substantial GPU resources.

It is not necessary simply to inspect the saved benchmark evidence.

Full instructions are available in:

`reports/REPRODUCIBILITY_AND_RUN_STEPS.md`

---

## 14. Interpretation and Research Integrity

The package preserves favorable and unfavorable findings.

The proposed MAD method reduced false-positive rates but achieved lower mean F1 and event recall than the two fixed baselines. H1 was not supported, H2 was inconclusive after multiple-comparison correction, and H3 remains descriptive.

The limitations involving label-informed training, validation calibration, test-label-informed cohort selection, and custom NAB-style scoring are disclosed throughout the documentation.

The verification results establish internal consistency of the archived experiment. They do not establish universal superiority, external validity, or production readiness.

---

## 15. Final Package Status

The submission package contains:

- Original research proposal.
- Main research overview.
- Package navigation guide.
- Primary benchmark source code and configuration.
- NAB datasets used in the experiment.
- Completed benchmark metrics and prediction traces.
- Corrected statistical reporting tables and plots.
- Automated tests and verification evidence.
- Historical V1 source reference and benchmark evidence.
- Detailed methodology audit.
- Reproducibility instructions.
- External references.

The package is intended to provide a transparent and reproducible record of the research implementation and its experimental findings.


## New candidate additions (2026-11-15)

- `reports/THRESHOLD_ABLATION_PROTOCOL.md` — candidate methodology and interpretation rules.
- `reports/REPRODUCIBILITY_AND_RUN_STEPS.md` — updated tests, smoke-run, and full GPU run commands.
- `source_code/src/thresholds.py` — raw MAD, scaled MAD, and IQR robust scale variants plus no-positive-validation FPR-budget fallback.
- `source_code/run_experiment.py` — six-method evaluation and compressed calibration-trace saving.
- `source_code/tests/test_pipeline.py` — tests for robust scale variants and the no-positive-validation fallback.

The candidate passed 22 tests and a one-fit CPU smoke run. Its full GPU benchmark has not yet been run; the archived V2 outputs remain the completed results until the candidate run is executed and independently audited.
