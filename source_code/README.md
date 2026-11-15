> **Historical report notice:** This document records the earlier 23-dataset V2 experiment and its review. The authoritative final experiment uses 46 datasets and is documented in `source_code/outputs/final46_six_methods/`. Use the final benchmark outputs and root `README.md` for current results.

# LSTM Autoencoder Adaptive Thresholding — V1 Evidence + V2 Code

The root of the ZIP contains `README_FIRST.md`, the original proposal, a completed V1 GPU benchmark and its audited evaluation, plus this revised V2 implementation. This `source_code/` folder contains the V2 implementation and the completed expanded GPU benchmark: 23 selected datasets × 3 windows × 3 seeds = 207 model fits and 828 threshold-method rows. Seventeen datasets have complete positive test events; six legacy datasets have no positive test labels and are retained for supplementary false-alarm analysis.

## V1 vs V2 — keep them separate

- **V1 completed evidence:** 10 NAB series × 3 windows × 3 seeds = 90 GPU model fits and 360 threshold-method rows. The raw results and 90 timestamp-aligned prediction files are preserved under `../verified_results/V1_full_gpu_benchmark_restart/`. V1 trained on all train windows without using labels to filter them; 9/10 files have labelled anomalies inside their train partition, so training data are contaminated by anomalies even though training is label-blind.
- **V2 revised code:** normal-only training windows selected using NAB labels; scaler fitted on normal training rows; 5% validation FPR budget used consistently; rolling mean+kσ tuned on validation; broader threshold grids; 40-epoch cap. Because labels filter training data, V2 is a label-informed / normal-only controlled experiment, not fully label-free unsupervised training. It is not a continuation of V1 under an unchanged protocol.

## Files

- `run_experiment.py` — training/evaluation runner.
- `config.json` — original V1 configuration reference.
- `config_v2.json` — V2 experiment settings.
- `prepare_expanded_nab_manifest.py` — screens and adds up to 15 more real NAB files; inspect `expanded_dataset_selection.json` after running.
- `src/` — preprocessing, LSTM-AE, thresholds, metrics, NAB-style scoring and reporting.
- `tests/` + `run_tests.py` — automated tests.
- `data/nab/` — original selected 10 NAB files and anomaly-window labels. The package contains 46 NAB CSV files. The 23 selected benchmark files are listed in `config_v2.json`; other CSVs are candidate files retained from dataset screening.

## Environment

The archived V1 GPU run was executed using Python 3.12.3 + TensorFlow 2.21.0 and an NVIDIA GeForce RTX 4050 Laptop GPU under WSL2. Main `requirements.txt` matches that runtime. `requirements-reference-tf215.txt` is an optional older reference environment for Python 3.10/3.11 only. Check the currently active environment and GPU before installing; do not downgrade a working TensorFlow GPU setup.

## Run V2

Follow `../reports/REPRODUCIBILITY_AND_RUN_STEPS.md`. At minimum:

```bash
python run_tests.py
python run_experiment.py --config config_v2.json --output-dir outputs/threshold_ablation_v3_20261115 --device cuda --skip-eda
```

The archived benchmark has 207 fits. The new candidate adds scaled-MAD and IQR variants and changes no-positive-validation calibration for all methods; it should be run into a separate output directory as described in `../reports/REPRODUCIBILITY_AND_RUN_STEPS.md`. Do not rerun the screening script unless intentionally rebuilding the dataset cohort; it screens additions only and does not reclassify the original 10 files.

## Re-audited H2 high-variance metrics

Use `outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/` for the H2 high-variance FPR analysis. The original prediction traces are read from `outputs/version2_normal_only_fpr05/predictions/`; they are not duplicated or modified. The reanalysis recalculates metrics without retraining models and uses only fully normal training histories to define the high-volatility reference cutoff. See `../reports/REVIEW_BUILD_CHANGELOG.md` for the audit record.

## Evaluation conventions

- The point-level PR-AUC field uses scikit-learn average precision (AP).
- ROC-AUC uses continuous reconstruction errors, not binary predictions.
- The moving threshold uses only preceding errors and excludes the current error from the history.
- NAB results are called *NAB-style test-partition scores*, not official full-corpus leaderboard scores.
- The `--allow-contaminated-training` option is only for a separately named sensitivity ablation; it overrides normal-only training.

## New threshold-ablation candidate

The candidate includes raw MAD, normal-scaled MAD (1.4826 × MAD), and IQR/1.3489795 scale methods. For validation partitions without positive labels, parameter selection now targets the least-conservative candidate within the common validation FPR budget instead of minimizing FPR without any recall evidence. The run saves calibration traces for future threshold-only re-analysis. These are methodological changes; higher F1 or NAB-style scores are not guaranteed. See `../reports/THRESHOLD_ABLATION_PROTOCOL.md`.
