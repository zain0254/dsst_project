# Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection

**Implementation for the proposal defense presentation**  
Researcher: Zain Hassan | FAST-NUCES Islamabad | MS Data Science

This repository implements the experiment in the supplied proposal slides. It trains a univariate LSTM autoencoder on NAB time series, converts the reconstruction error into anomaly alerts using four thresholding strategies, tunes threshold parameters on the validation partition, and evaluates the selected settings on an untouched chronological test partition.

## 1. What the project implements

The four threshold methods are:

1. **Fixed percentile** - a percentile of the training reconstruction-error distribution. The percentile is selected from the validation set.
2. **Fixed mean + k sigma** - `mean(training_error) + k * std(training_error)`. The value of `k` is selected from validation.
3. **Rolling mean + 3 sigma** - the baseline shown as the 2026 paper method in the slides. It uses a causal rolling history and a fixed multiplier of 3.
4. **Moving median + lambda MAD** - the proposed method. At time `t`, it computes `median(past errors) + lambda * MAD(past errors)`. The lambda is selected from validation only.

The MAD used here is the **raw median absolute deviation**:

`MAD = median(abs(error_i - median(errors)))`

No 1.4826 normal-consistency multiplier is applied. **The arithmetic printed on proposal slide 11 needs correction:** for the displayed array `[0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 5.0]`, the median is 0.15 (not 0.20), the raw MAD is 0.05, and a multiplier of 3 gives a threshold of 0.30 (not 0.35). The implementation computes these quantities from the data rather than copying the inconsistent displayed median.


## 1A. Proposal objective-to-code mapping

| Proposal objective | Implemented files / outputs |
|---|---|
| RO1: EDA and rolling variability analysis | `src/eda.py`; outputs in `outputs/eda/` and `outputs/eda_summary.csv` |
| RO2: cleaning, chronological split, scaling, and sliding windows | `src/data_utils.py`; split audit in `outputs/dataset_split_audit.csv` |
| RO3: LSTM autoencoder and reconstruction errors | `src/model.py` |
| RO4: three baseline threshold methods | `src/thresholds.py`: fixed percentile, fixed mean + k sigma, rolling mean + 3 sigma |
| RO5: proposed moving median + lambda MAD tuned on validation only | `src/thresholds.py`; selected parameters in `outputs/validation_tuning.csv` |
| RO6: metrics, hypothesis checks, and domain comparison | `src/metrics.py`, `src/nab_scoring.py`, `src/benchmark_reporting.py`; benchmark tables and RQ report in `outputs/` |

### Project files

```text
lstm_ae_robust_thresholding/
├── .github/workflows/tests.yml
├── config.json
├── requirements.txt
├── run_experiment.py
├── run_tests.py
├── README.md
├── RESEARCH_QUESTIONS.md
├── data/nab/.gitkeep
├── src/
│   ├── data_utils.py
│   ├── eda.py
│   ├── benchmark_reporting.py
│   ├── metrics.py
│   ├── model.py
│   ├── model_tf.py
│   ├── model_torch.py
│   ├── nab_scoring.py
│   └── thresholds.py
├── tests/
│   └── test_pipeline.py
└── outputs/
    └── .gitkeep
```

The reference implementation uses **TensorFlow 2.15.1 / Keras**, matching the environment specified in the proposal slides. `src/model.py` selects the TensorFlow implementation when TensorFlow is installed. A PyTorch fallback (`src/model_torch.py`) is included only to keep the smoke tests usable in environments where TensorFlow is unavailable; it is not the reference framework for the research experiment. The three independent model runs use seeds `15`, `42`, and `123`; NumPy preprocessing is fixed at seed `42`. TensorFlow deterministic operations are enabled where supported, although exact repeatability can still vary across hardware and library builds.

## 2. Dataset selection

The code downloads selected CSV files and the official NAB `combined_windows.json` label file from the Numenta NAB GitHub repository. It does not download the entire benchmark.

The default configuration contains 10 files:

| NAB domain | Files |
|---|---|
| AWS CloudWatch | `ec2_cpu_utilization_24ae8d.csv`, `ec2_cpu_utilization_53ea38.csv`, `ec2_cpu_utilization_5f5533.csv` |
| Twitter | `Twitter_volume_AAPL.csv`, `Twitter_volume_AMZN.csv` |
| Traffic | `TravelTime_387.csv`, `TravelTime_451.csv` |
| Known Cause | `ambient_temperature_system_failure.csv`, `machine_temperature_system_failure.csv`, `nyc_taxi.csv` |

**Count note:** the presentation says 10-12 files, but its listed category counts (3 + 2 + 2 + 2) add up to 9. The configuration adds `nyc_taxi.csv` as a third Known Cause file so that the default experiment has 10 datasets, consistent with the hypothesis and statistical-testing slides. If your supervisor confirms that only two Known Cause files should be used, remove one from `config.json` and revise the hypothesis wording from 10 files to 9 files.

NAB's real-data categories are described by the official dataset README: CloudWatch contains AWS server metrics; Twitter contains mention-volume series; Traffic contains Twin Cities traffic metrics; and Known Cause contains series with documented anomaly causes. See the NAB repository links at the end of this document.

## 3. Experimental protocol

For every dataset and every requested sequence length `W`:

1. Sort timestamps chronologically, remove duplicate timestamps, convert values to numeric, forward-fill missing values, and drop leading rows that have no earlier value to carry forward. The code never backward-fills from future observations.
2. Split the complete series chronologically into 60% train, 20% validation, and 20% test.
3. Fit `MinMaxScaler` on the training partition only. Use that fitted scaler to transform the complete series.
4. Create univariate sliding windows for `W = 30, 50, 100`. Each window ends at the timestamp it is scored for. Validation/test windows may use only observations from the past, never future observations.
5. Train the LSTM autoencoder using the training windows. Early stopping monitors validation reconstruction loss. The code currently uses all training windows; it does not filter out labeled anomalies from training because the proposal slides do not specify such filtering.
6. Generate one reconstruction MSE per window, assigned to its final timestamp.
7. Select threshold settings using validation labels only. The test labels are not used for parameter selection.
8. Refit no model and retune no threshold on the test partition. Dynamic threshold methods update their rolling history causally as each test error arrives.

### Explicit implementation assumptions

The slides do not specify a separate lookback length for the rolling thresholds, so the code uses the same value as the sequence length `W`. The rolling threshold for time `t` is calculated from the preceding `W` reconstruction errors, excluding the current error before deciding whether it is anomalous.

The slides refer to “high variance” but do not define it. The code defines a high-variance test point as one whose standard deviation over the preceding `W` raw observations is at least the 75th percentile of rolling standard deviations calculated from the training partition. This definition is training-only and causal at test time. Change `high_variance_percentile` in `config.json` if the supervisor chooses another definition.

The model architecture (64-unit encoder/decoder LSTM, 32-dimensional latent representation) is a concrete implementation choice because the slides do not specify layer widths. You can adjust these in `config.json`.

## 4. Installation (Windows / VS Code)

Use **Python 3.10**, as specified in the proposal, in a virtual environment. Open PowerShell in this project folder.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file pins TensorFlow 2.15.1 and the principal numerical libraries to a compatible reproducibility environment. On Windows, TensorFlow may use the CPU unless you have a supported GPU environment (for example, a correctly configured WSL2 setup). The default `--device auto` uses a TensorFlow GPU when one is visible and otherwise falls back to CPU. You can always select `--device cpu`.

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Verify the code before running the full experiment

Read `RESEARCH_QUESTIONS.md` for the exact RQ1-RQ4 / H1-H3 mapping, metrics, high-variance definition, and interpretation limits.

Run the included automated tests after installing the requirements:

```powershell
python run_tests.py
```

These tests cover cleaning and label alignment, train-only scaler fitting, LSTM training and reconstruction errors, threshold tuning, causal rolling thresholds, event scoring, and statistical-output generation. A successful test run is a software check; it is not a substitute for running the full NAB experiment.

## 6. Download NAB data and run the full experiment

The first run downloads the selected NAB CSV files plus `labels/combined_windows.json`. An internet connection is required for this first download.

```powershell
python run_experiment.py
```

The full configuration runs 10 datasets x 3 sequence lengths x 3 independent seeds (90 model fits). The code also saves the exact configuration and library/hardware versions used in each run. For a quick end-to-end check, use fewer epochs and one window length:

```powershell
python run_experiment.py --windows 30 --epochs 2 --patience 1 --batch-size 128
```

To run only one or two datasets, pass exact paths from `config.json`:

```powershell
python run_experiment.py --windows 30 50 --files realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv realTweets/Twitter_volume_AAPL.csv
```

To force a fresh download:

```powershell
python run_experiment.py --force-download
```

To save trained model weights and the scaler parameters:

```powershell
python run_experiment.py --save-models
```

The script automatically selects a TensorFlow GPU when one is visible. Use `--device cpu` to force CPU execution or `--device cuda` to require a visible TensorFlow GPU.

## 7. Output files

All outputs are saved under `outputs/`.

| File/folder | Contents |
|---|---|
| `test_metrics.csv` | Test precision, recall, F1, FPR, PR-AUC, confusion counts, event recall, detection delay, and high-variance FPR for each dataset/window/method |
| `validation_tuning.csv` | The parameter selected on validation for every method and dataset/window |
| `dataset_split_audit.csv` | Dataset row counts, date ranges, anomaly points/events in train, validation, and test |
| `method_summary.csv` | Aggregate mean and standard deviation by threshold method |
| `dataset_method_summary.csv` | Results averaged by dataset and threshold method |
| `domain_improvement_summary.csv` | Mean F1 improvement of the proposed method over rolling mean + 3 sigma, grouped by AWS, Twitter, Traffic, and Known Cause domains |
| `statistical_tests.csv` | Wilcoxon signed-rank test for dataset-level F1 and paired t-test for high-variance FPR |
| `hypothesis_outcomes.csv` | H1-H3 criteria, observed counts, evaluable counts, p-values, and supported/inconclusive decisions |
| `benchmark_method_summary.csv` | Equal-weighted dataset-level comparison of all four threshold methods |
| `per_dataset_comparison.csv` | One averaged row per NAB file and method (averages seeds and sequence lengths) |
| `pairwise_proposed_vs_baselines.csv` | Proposed method versus each of the three baselines for F1, PR-AUC, NAB-style score, FPR, high-variance FPR, event recall, and delay; includes paired tests, bootstrap confidence intervals, and Holm-adjusted p-values |
| `comparison_by_window_size.csv` | Metric comparison grouped by sequence length W |
| `seed_variability.csv` | Mean and standard deviation across the three independent training seeds for each file/window/method |
| `domain_comparison.csv` | Proposed-minus-rolling-mean+3-sigma F1 effect by AWS, Twitter, Traffic, and Known Cause domains, with bootstrap intervals and evaluable file counts |
| `high_variance_fpr_by_dataset.csv` | Per-file absolute and relative high-variance false-positive-rate reduction for H2 |
| `research_question_results.md` | Automatically generated results report organized by RQ1-RQ4 and H1-H3; only generated after the experiment has actually run |
| `config_used.json` | Exact experiment configuration, including command-line overrides |
| `environment.json` | Python/library versions, platform, framework version, and accelerator information |
| `run_summary.json` | Dataset list, sequence lengths, seeds, model settings, split, and experiment counts |
| `predictions/` | Timestamp-aligned test labels, reconstruction errors, threshold traces, and method predictions |
| `plots/` and `eda/` | Aggregate comparison plots, example threshold plots, EDA plots, and rolling-statistics CSVs |
| `models/` | Saved Keras model weights (or fallback checkpoint) and a JSON sidecar containing architecture, seed, backend, and MinMax scaling parameters (only with `--save-models`) |

### Statistical tests

The code first averages each method's test result across the configured sequence lengths and random seeds within each dataset. It then treats each dataset as one paired unit. This avoids incorrectly treating multiple sequence lengths or repeated runs from the same file as independent samples.

- **RQ1:** `benchmark_method_summary.csv` and `per_dataset_comparison.csv` compare fixed percentile, fixed mean + k sigma, rolling mean + 3 sigma, and proposed moving median + lambda MAD.
- **RQ2 / H1:** Pairwise Wilcoxon signed-rank comparisons compare proposed versus each baseline for F1, PR-AUC, and NAB-style test score. H1 uses the proposal's specific ≥7-of-10 F1 criterion against rolling mean + 3 sigma and is only supported when all 10 configured datasets have complete labeled test events, at least 7 improve, and the paired Wilcoxon p-value is below 0.05.
- **RQ3 / H2:** Paired t-tests compare high-variance FPR; `high_variance_fpr_by_dataset.csv` records per-file relative reductions. H2 requires a mean relative FPR reduction of at least 15% and p < 0.05.
- **RQ4 / H3:** `domain_comparison.csv` ranks mean file-level F1 improvements by domain. Because only two or three selected files represent each domain, this is reported as descriptive/exploratory evidence, not as a generalizable significant domain effect.
- Pairwise p-values in the wider benchmark comparison are Holm-adjusted within each metric family to reduce false positives from multiple baseline comparisons.
- Significance level: `alpha = 0.05`.

The statistical tests are only meaningful when enough datasets have labeled anomaly events in the held-out test partition and enough normal high-variance points. Check `dataset_split_audit.csv`, `n_datasets_tested` in `statistical_tests.csv`, and the evaluable/configured counts in `hypothesis_outcomes.csv` before interpreting a p-value. A p-value below 0.05 does not prove practical importance; report the mean paired difference as well.

## 8. Important evaluation note about NAB Score

The code uses NAB's labeled **time intervals** and implements the documented Standard-profile timing reward and false-positive weighting for complete anomaly windows in the held-out test partition. It reports this as `nab_style_score`, normalized against null/perfect scores for the evaluated test subset. The official NAB repository describes the complete v1.1 corpus as 58 time-series files and its full benchmark runner/normalization procedure; this course project uses a declared 10-file subset and chronological test portions, so this project score is not an official NAB leaderboard score. The test partition already follows the train/validation warm-up, so the score applies no additional probationary portion by default. See the official NAB repository: https://github.com/numenta/NAB.

Keep the distinction clear in your report: call this the *NAB-style normalized test score*, not the official corpus-wide NAB score. Do not rename `event_recall` or `mean_early_detection_credit` as NAB Score.

## 9. Research limitations to discuss

- The 60/20/20 chronological split may place some labeled anomaly windows fully in train or validation, leaving a particular test file without an anomaly event. The audit CSV makes this visible; do not present a test F1 value from a no-positive test split as evidence of anomaly-detection quality.
- The threshold hyperparameters are tuned with validation labels. The learned autoencoder is unsupervised, but threshold selection is validation-supervised.
- Training uses all training windows, including any labeled anomalous windows in the training partition. That is what this implementation does to follow the current slide description; it may make the autoencoder learn some anomaly patterns. A normal-only training variant should be a separately declared ablation, not silently substituted.
- NAB files are univariate. Conclusions should be limited to single-metric time-series streams unless more datasets are added.
- A single train/validation/test chronological split is one experimental protocol. Repeated blocked time-series splits can be added as a robustness extension.

## 10. Publish the implementation on GitHub

Create an empty GitHub repository (do not initialize it with a README or license), then run these commands from this project folder in PowerShell. Replace the URL with the HTTPS URL of your own repository.

```powershell
git init
git add .
git commit -m "Initial implementation of robust adaptive thresholding"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

The `.gitignore` deliberately excludes downloaded NAB data, generated experiment outputs, virtual environments, caches, and trained model weights. The repository therefore contains the complete source code and configuration; anyone cloning it can reproduce the experiment by installing `requirements.txt` and running `python run_experiment.py`. Do not commit private credentials or personal access tokens.

## 11. References and data source

- Numenta, **Numenta Anomaly Benchmark (NAB)** repository: https://github.com/numenta/NAB
- NAB data-corpus documentation: https://github.com/numenta/NAB/blob/master/data/README.md
- NAB combined anomaly windows: https://github.com/numenta/NAB/blob/master/labels/combined_windows.json
- Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). *Unsupervised real-time anomaly detection for streaming data*. Neurocomputing. https://doi.org/10.1016/j.neucom.2017.04.070

The supplied slides identify a 2026 paper as the base-paper reference but do not include enough bibliographic details to independently reproduce that paper's full pipeline. The `rolling_mean_3sigma` implementation here reproduces the threshold definition written in the slides (rolling mean + 3 standard deviations); it should not be described as an exact reproduction of an unverified paper implementation.
