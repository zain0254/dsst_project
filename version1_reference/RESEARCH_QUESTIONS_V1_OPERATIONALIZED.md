# Research Questions and Benchmark Protocol

This file describes what the implementation measures. It does not contain experimental findings. Findings are produced only after running `python run_experiment.py` on the selected NAB files.

## Main research question

**Can a causal moving-median plus median-absolute-deviation (MAD) threshold improve the robustness of LSTM-autoencoder anomaly detection in streaming time series compared with fixed and rolling mean/standard-deviation thresholds?**

The experiment holds the model, input data, reconstruction errors, and train/validation/test split constant. Only the thresholding strategy changes. This is the controlled comparison required by the proposal.

## RQ1 - Baseline performance

**How do fixed percentile, fixed mean + k sigma, and rolling mean + 3 sigma thresholds perform on the selected NAB files?**

Methods compared:

1. Fixed percentile of training reconstruction errors. The percentile is selected using validation labels.
2. Fixed mean + k population standard deviations of training reconstruction errors. `k` is selected using validation labels.
3. Causal rolling mean + 3 population standard deviations of prior reconstruction errors. The multiplier is fixed at 3, as shown in the proposal.
4. Causal rolling median + lambda raw MAD of prior reconstruction errors. `lambda` is selected using validation labels.

Report precision, recall, F1, false-positive rate (FPR), PR-AUC / average precision, event recall, mean detection delay, and the NAB-style normalized test score. The last metric is a test-subset application of NAB Standard-profile timing rewards; it is not an official corpus-wide NAB leaderboard score.

Primary outputs: `benchmark_method_summary.csv`, `per_dataset_comparison.csv`, `test_metrics.csv`, and `plots/benchmark_*.png`.

## RQ2 - Superiority of moving median + MAD

**Does moving median + lambda MAD produce a higher F1-score and NAB-style score than the three baselines?**

For every NAB file, average over the three independent model seeds and the configured sequence lengths. Treat the NAB file—not an individual seed/window combination—as the paired statistical unit. Compare the proposed threshold with each baseline using paired differences. The reporting module provides Wilcoxon signed-rank tests, 95% bootstrap confidence intervals, and Holm-adjusted p-values for the three baseline comparisons within each metric family.

**H1 from the proposal:** the proposed method has a higher mean test F1 on at least 7 of 10 NAB files than rolling mean + 3 sigma. H1 is marked supported only when all 10 configured files have complete labeled events in their test portions, at least 7 have a higher proposed-method F1, and the paired Wilcoxon test is significant at alpha = 0.05. If one or more files do not contain a complete labeled event in the test partition, the result is reported as inconclusive for the strict 7-of-10 claim.

Primary outputs: `pairwise_proposed_vs_baselines.csv`, `hypothesis_outcomes.csv`, and `research_question_results.md`.

## RQ3 - Robustness during high variance

**Does the proposed method reduce false alarms when normal stream variability shifts?**

The proposal does not define “high variance.” For this implementation, a test point is classed as high variance when its causal rolling standard deviation over the preceding `W` raw observations is at or above the 75th percentile of rolling standard deviations measured on the training portion. The percentile cutoff is derived from training data only. The current test point is not used to calculate its own volatility score.

Calculate high-variance FPR only on high-variance points whose ground-truth label is normal. Compare the proposed method with rolling mean + 3 sigma using the paired t-test specified in the proposal. Also report per-file absolute and relative reductions. A relative reduction is undefined when the baseline FPR is zero; those files are not silently assigned a percentage reduction of zero.

**H2 from the proposal:** mean relative high-variance FPR reduction is at least 15% and the paired t-test has p < 0.05.

Primary outputs: `high_variance_fpr_by_dataset.csv`, `statistical_tests.csv`, and `hypothesis_outcomes.csv`.

## RQ4 - Domain dependency

**Which selected NAB data category benefits most from moving median + MAD?**

Compare the proposed-minus-rolling-mean+3-sigma F1 difference by dataset and then by category: AWS CloudWatch, Twitter, Traffic, and Known Cause. Report the number of evaluable files, mean and median improvement, standard deviation, 95% bootstrap confidence interval, positive-file count, and descriptive rank.

**H3 from the proposal:** AWS CloudWatch has the largest average improvement. Since each category has only two or three selected files, domain results are exploratory/descriptive and must not be generalized as a statistically established property of all cloud monitoring data.

Primary output: `domain_comparison.csv` and `plots/domain_f1_improvement.png`.

## Dataset selection

The proposal slide lists 3 AWS CloudWatch + 2 Twitter + 2 Traffic + 2 Known Cause files, which totals 9, while H1 refers to 10 files and the dataset slide says 10-12. To meet the stated 10-file H1 denominator, the configuration uses a third Known Cause file (`nyc_taxi.csv`). The full list is in `config.json`. This choice should be confirmed with the course supervisor if the intended dataset list is different.

## Shared experimental controls

- Dataset split: chronological 60% training / 20% validation / 20% test.
- Scaling: `MinMaxScaler` fitted only on the 60% training rows.
- Sequence lengths: W = 30, 50, and 100.
- Model: univariate LSTM autoencoder; architecture and training settings are recorded in `config.json`.
- Independent training seeds: 15, 42, and 123.
- Threshold tuning: validation labels only; test labels are never used to select parameters.
- Streaming thresholds: compare each error against a threshold computed from prior errors, then append the current error to the history.
- The same trained model and reconstruction errors are used for all four methods for each dataset/window/seed combination.
- The full experiment comprises 10 datasets x 3 sequence lengths x 3 seeds = 90 model fits. Each fit generates four threshold-method metric rows.

## Arithmetic correction to proposal slide 11

For the slide's displayed error array `[0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 5.0]`, the actual median is 0.15, not 0.20. The raw MAD is 0.05. Thus with lambda = 3 the threshold is 0.15 + 3(0.05) = 0.30, not 0.35. The implementation follows the formula applied to the actual observed values.

## Important interpretation limits

- The NAB corpus itself contains more time series than this selected 10-file subset. Do not present the subset results as a full NAB benchmark leaderboard result.
- A labeled event that crosses a split boundary is not counted as a complete event for event recall or NAB-style test scoring. The data audit records these exclusions. Pointwise test metrics still use the point labels in the test interval, so report split coverage alongside event-level results.
- Training is unsupervised with respect to the reconstruction task, but threshold hyperparameters are validation-supervised because validation anomaly labels select percentile / k / lambda.
- Training uses all training windows, including any labeled anomaly windows that happen to occur in the training period. This follows the current proposal protocol; a normal-only training ablation would be a separate experiment.
- A passing software test or synthetic demo verifies code mechanics only. It is not empirical evidence for H1, H2, or H3.
