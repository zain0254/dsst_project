> **Historical report notice:** This document records the earlier 23-dataset V2 experiment and its review. The authoritative final experiment uses 46 datasets and is documented in `source_code/outputs/final46_six_methods/`. Use the final benchmark outputs and root `README.md` for current results.

# Reproducibility and Run Steps — Threshold Ablation Candidate

This ZIP preserves the completed V1/V2 evidence and adds a new threshold-ablation candidate. The new candidate has passed its local automated tests and a one-dataset CPU smoke run, but the full 207-fit GPU experiment has **not** yet been run. Do not treat candidate F1/NAB results as available until that run completes on the reference Ubuntu/WSL environment.

## 1. Extract the package in WSL

```bash
mkdir -p ~/projects/LSTM_AE_FINAL
unzip ~/LSTM_AE_RESEARCH_CANDIDATE.zip -d ~/projects/LSTM_AE_FINAL
cd ~/projects/LSTM_AE_FINAL/LSTM_AE_FINAL_SUBMISSION_PACKAGE/source_code
```

The extra `LSTM_AE_FINAL_SUBMISSION_PACKAGE/` directory is part of the ZIP structure.

## 2. Activate and verify the GPU environment

```bash
source ~/venvs/lstm_tf221_gpu/bin/activate
python --version
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
nvidia-smi
```

The previously recorded environment was Python 3.12.3, TensorFlow 2.21.0, and an NVIDIA RTX 4050 Laptop GPU under WSL2. If this environment is already working, do not reinstall or downgrade TensorFlow. Install only missing packages.

## 3. Run the automated tests

```bash
python run_tests.py
```

Expected: **22 tests pass**. These include tests for causal rolling thresholds, raw/scaled MAD and IQR scale estimators, the no-positive-validation FPR-budget fallback, metrics, NAB-style scoring, and reporting.

## 4. Run a one-fit smoke experiment first

This checks the full new six-method pipeline without launching all 207 model fits:

```bash
python run_experiment.py \
  --config config_v2.json \
  --files realKnownCause/ambient_temperature_system_failure.csv \
  --windows 30 --seeds 15 \
  --epochs 1 --batch-size 64 --patience 1 \
  --device cpu \
  --output-dir outputs/threshold_ablation_smoke \
  --skip-eda
```

Confirm that `test_metrics.csv` has six rows, `validation_tuning.csv` has six rows, and `calibration_traces/` contains one `.npz` file. This is only a software smoke test, not research evidence.

## 5. Run the full GPU experiment

Keep the archived completed benchmark folders untouched. Run the new candidate into its own output directory:

```bash
set -o pipefail
python run_experiment.py \
  --config config_v2.json \
  --output-dir outputs/threshold_ablation_v3_20261115 \
  --device cuda --skip-eda \
  2>&1 | tee outputs/threshold_ablation_v3_20261115_console.log
```

This run trains **207 models** (23 datasets × 3 window sizes × 3 seeds) and evaluates each model with six threshold methods, producing **1,242 metric rows**. It is not 1,242 model fits. Calibration traces are saved automatically so later threshold ablations can reuse reconstruction errors without retraining.

Do not use `--allow-contaminated-training` for the primary protocol. Do not change the dataset cohort, window sizes, seeds, split, or validation FPR budget mid-run.

## 6. What changed in this candidate

See `THRESHOLD_ABLATION_PROTOCOL.md` and `REVIEW_BUILD_CHANGELOG.md`.

- Added scaled MAD: `1.4826 × MAD`.
- Added IQR scale: `IQR / 1.3489795`.
- Retained the original raw MAD threshold formula as a separate method.
- Corrected the no-positive-validation fallback for all methods. In the previous saved benchmark, 117 of 207 model runs had no positive validation labels. In those runs, F1 cannot distinguish threshold candidates; the new fallback chooses the least-conservative candidate within the same predeclared 5% validation FPR budget.
- Saves compressed calibration traces for future re-analysis.
- Adds dataset-macro exploratory summaries. New scaled-MAD/IQR comparisons are descriptive and not confirmatory hypothesis tests.

The fallback changes threshold selection for every method in runs with no positive validation labels. Therefore, do not attribute any new F1/NAB change solely to scaled MAD or IQR. Compare the archived V2 benchmark with this candidate as a protocol sensitivity analysis, and compare the six methods within the new run as the controlled within-run ablation.

## 7. Cohort and reporting rules

The selected cohort contains 23 datasets. The prior audit found 17 datasets with complete positive test events and six datasets with no positive test labels. The six negative-only datasets can contribute to point-level false-alarm analysis but must not enter event recall or event-based NAB denominators.

The `threshold_ablation_summary.csv` macro-averages dataset-level means first. Treat it as descriptive. The existing `statistical_tests.csv` and `hypothesis_outcomes.csv` preserve the prespecified raw-MAD comparisons against the existing baselines; the new robust-scale variants are exploratory and must not be described as confirmatory wins.

The reported score is a NAB-style score on complete events in each held-out test partition, not the official NAB full-corpus leaderboard score.

## 8. Previous results remain available

- Original completed V2 benchmark: `outputs/version2_normal_only_fpr05/`
- Corrected reporting: `outputs/version2_normal_only_fpr05_reporting_corrected_20261115/`
- Clean-normal volatility reanalysis: `outputs/version2_normal_only_fpr05_normal_volatility_reaudited_20261115/`

Keep these folders and the new candidate output separate. Do not overwrite historical evidence.
