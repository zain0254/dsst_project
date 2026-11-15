# README FIRST — Project Navigation Guide

## 1. Welcome

This package contains the research implementation and experimental evidence for:

**Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series**

The GitHub repository is named `dsst_project`.

The archived completed experiment evaluates four reconstruction-error thresholding methods across 23 selected NAB datasets: 17 have complete positive test events and six have no positive test labels. The six no-positive-test datasets are retained for supplementary false-alarm analysis, not counted as event-evaluable.

Start with the root-level `README.md` for the research overview and principal results.

## 2. Recommended Reading Order

For someone reviewing this research project, the recommended order is:

1. `README.md` — research overview, experimental results, methods, and conclusions.
2. `PROPOSAL_HYPOTHESIS_ALIGNMENT.md` — relationship between the original proposal, research questions, and implemented hypotheses.
3. `reports/DATASET_COHORT_AUDIT.md` — exact 23-dataset composition and test-label coverage.
4. `reports/FINAL_RESULTS_AND_METHOD_AUDIT.md` — detailed experimental findings, statistical tests, verification, and methodological limitations.
5. `reports/REVIEW_BUILD_CHANGELOG.md` — corrections made during the review build.
6. `reports/REPRODUCIBILITY_AND_RUN_STEPS.md` — software environment, test instructions, and benchmark reproduction procedure.
7. `FINAL_PACKAGE_MANIFEST.md` — package inventory and verification summary.
8. `WEB_REFERENCES.md` — external references used by the project.

## 3. Primary Benchmark Summary

The completed primary benchmark contains:

| Component | Result |
|---|---:|
| Selected NAB datasets | 23 |
| Event-evaluable test datasets | 17 |
| No-positive-test datasets | 6 |
| Window sizes | 30, 50, 100 |
| Random seeds | 15, 42, 123 |
| LSTM-AE model fits | 207 |
| Threshold methods per fit (archived V2) | 4 |
| Evaluation rows (archived V2) | 828 |
| Prediction files | 207 |
| Current package automated tests passed | 22 |

The primary benchmark's original output directory is:

`source_code/outputs/version2_normal_only_fpr05/`

The floating-point corrected reporting directory is:

`source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/`

The clean-normal volatility reanalysis directory (authoritative for H2 high-variance FPR) is:

`source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`

These directory names retain their original development identifiers so that they remain consistent with the implementation.

## 4. Main Experimental Finding

The proposed moving-median plus MAD threshold achieved a substantially lower mean false-positive rate than the fixed-percentile and fixed mean-plus-standard-deviation baselines.

However, it also achieved lower mean F1 and event recall than both fixed baselines.

The findings therefore indicate a trade-off between false-alarm reduction and anomaly-detection performance. They do not establish overall superiority of the proposed method.

The statistical interpretation is:

- **H1 — F1 improvement:** Not supported.
- **H2 — False-positive reduction:** Some dataset-level improvements observed, but aggregate evidence was inconclusive after Holm correction.
- **H3 — NAB-style score and timing:** Descriptive trade-off only; no inferential evidence establishes a significant advantage.

## 5. Important Methodological Disclosures

The primary benchmark is not fully label-free unsupervised learning.

- NAB labels are used to exclude anomaly-touching training windows.
- The scaler is fitted on normal training rows only.
- Validation labels are used during threshold calibration.
- Test-label properties are used in eligible-dataset selection, creating potential cohort-selection bias.
- The 5% validation FPR is a target budget, not a guaranteed upper bound.
- Six selected datasets have no positive test labels. Event-recall interpretation uses the remaining 17 event-evaluable datasets.
- The NAB-style score is a custom chronological test-partition score, not the official full-corpus NAB leaderboard score.

These limitations must be considered when interpreting or citing the results.

## 6. Earlier Benchmark (V1)

The package preserves an earlier experiment separately:

- 10 NAB datasets.
- 90 model fits.
- 360 threshold-method evaluation rows.
- 90 prediction files.

V1 and the primary benchmark use different experimental protocols. Their aggregate results must not be combined.

Historical V1 evidence is retained in:

`verified_results/V1_full_gpu_benchmark_restart/`

The earlier source-code reference is retained in:

`version1_reference/`

## 7. Package Directory Guide

| Directory or file | Purpose |
|---|---|
| `README.md` | Main research overview and results |
| `README_FIRST.md` | This navigation guide |
| `FINAL_PACKAGE_MANIFEST.md` | Package inventory and verification summary |
| `PROPOSAL_HYPOTHESIS_ALIGNMENT.md` | Proposal and hypothesis alignment |
| `WEB_REFERENCES.md` | External references |
| `proposal/` | Original research proposal |
| `source_code/` | Main implementation, configuration, tests, data, and outputs |
| `source_code/config_v2.json` | Primary benchmark configuration |
| `source_code/data/nab/` | NAB datasets used in the experiment |
| `source_code/outputs/version2_normal_only_fpr05/` | Original primary benchmark outputs |
| `source_code/outputs/version2_normal_only_fpr05_reporting_corrected_20261115/` | Floating-point corrected reporting tables and plots |
| `source_code/outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/` | Re-audited H2 high-variance metrics and reporting |
| `reports/` | Results audit and reproducibility documentation |
| `evaluation/` | Evaluation material |
| `analysis_tools/` | Supporting analysis utilities |
| `verification/` | Verification evidence |
| `verified_results/` | Archived earlier benchmark evidence |
| `version1_reference/` | Earlier implementation reference |

## 8. Software Environment

The completed primary benchmark used:

- Ubuntu under WSL2.
- Python 3.12.3.
- TensorFlow 2.21.0.
- NVIDIA RTX 4050 Laptop GPU.
- CUDA-compatible TensorFlow GPU execution, with cuDNN loaded.

The working virtual environment was:

`~/venvs/lstm_tf221_gpu`

## 9. Running the Tests

Activate the environment:

```bash
source ~/venvs/lstm_tf221_gpu/bin/activate
```

Move to the source directory:

```bash
cd source_code
```

Run the test suite:

```bash
python run_tests.py
```

The current candidate test suite passes 22 tests. The archived V2 benchmark was originally verified with 20 tests; two new tests cover the robust-scale variants and no-positive-validation selection correction.

## 10. Reproducing the Primary Benchmark

To run a new experiment without overwriting the archived benchmark outputs:

```bash
python run_experiment.py \
  --config config_v2.json \
  --output-dir outputs/threshold_ablation_v3_20261115 \
  --device cuda --skip-eda
```

This command trains the models again and may require substantial GPU resources. It is not necessary simply to inspect the saved results.

See `reports/REPRODUCIBILITY_AND_RUN_STEPS.md` for detailed instructions.

## 11. Final Note

The package preserves the completed experiment, its raw outputs, corrected statistical reporting, historical evidence, and methodological limitations.

For scientific interpretation, use the detailed audit report rather than relying only on the summary tables in the main README.

**Recommended starting point: `README.md`**

## 11. New threshold-ablation candidate

This ZIP adds raw MAD, scaled MAD, and IQR-scale variants plus a validation-only fallback for runs with no positive validation labels. The full 207-fit GPU benchmark is not yet run. Read `reports/THRESHOLD_ABLATION_PROTOCOL.md` and follow `reports/REPRODUCIBILITY_AND_RUN_STEPS.md`; do not treat the one-fit smoke run as research evidence.
