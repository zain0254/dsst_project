# Research Questions and Hypotheses — Proposal-Aligned Version

This document keeps the original proposal's research questions and hypotheses separate from later implementation extensions. It describes the study protocol, not empirical findings. Use `../reports/FINAL_RESULTS_AND_METHOD_AUDIT.md` and the `../evaluation/` folder for completed V1 findings.

## Main research question

Can a causal moving Median + Median Absolute Deviation (MAD) threshold improve the reliability of LSTM-Autoencoder anomaly decisions in streaming time series compared with fixed threshold rules?

## Original proposal research questions

**RQ1:** How does an LSTM Autoencoder with fixed percentile and fixed mean-plus-standard-deviation thresholds perform on selected NAB streaming time-series files?

**RQ2:** Does the proposed moving Median + MAD threshold improve precision, recall, F1-score, PR-AUC, or NAB score compared with fixed threshold baselines?

**RQ3:** Does the proposed threshold reduce false-positive anomaly alerts during periods of changing normal data variability?

**RQ4:** Which NAB categories or anomaly types benefit most from robust adaptive thresholding?

## Original proposal hypotheses

- **H1:** The proposed moving Median + MAD threshold will achieve a higher average F1-score than at least one fixed-threshold baseline.
- **H2:** The proposed method will reduce false-positive rates on some time series with changing normal variability.
- **H3:** The proposed method will maintain or improve NAB score by detecting anomalies with fewer false alerts or earlier detection.

These are the exact substantive claims to evaluate in the thesis. They do not impose a 7-of-10 requirement, a 15% FPR-reduction threshold, or an AWS-domain rank requirement.

## Methods

The proposal's three core methods are:

1. Fixed high-percentile threshold based on training reconstruction errors; percentile selected on validation.
2. Fixed mean + k population standard deviations of training reconstruction errors; k selected on validation.
3. Proposed causal moving Median + λ × raw MAD using the preceding reconstruction-error history; λ selected on validation.

The later implementation also includes **rolling mean + 3σ** as an additional comparator. It is useful for evaluating adaptive versus fixed thresholds, but it is not one of the proposal's two fixed baselines. In V2 this is changed to validation-calibrated rolling mean + kσ, so V1 and V2 rolling-baseline results are not directly identical.

## Shared V1 protocol

- Selected NAB files: 10 real-world time series in AWS CloudWatch, Twitter, Traffic, and Known Cause categories.
- Split: chronological 60% training / 20% validation / 20% test.
- Scaling: `MinMaxScaler` fitted on training rows only.
- Sequence lengths: W = 30, 50, and 100.
- Model: univariate LSTM Autoencoder; architecture and training settings are recorded in the archived `config_used.json`.
- Independent model seeds: 15, 42, and 123; 90 total model fits.
- The same trained model and reconstruction-error series feed all threshold methods for each dataset/window/seed combination.
- Threshold parameters are selected using validation data; test data are not used for parameter selection.
- Rolling thresholds use only past errors and exclude the current error from the threshold history.
- The main inferential unit is the dataset; model seeds and window lengths are repeated conditions within each dataset, not independent datasets.

## V1 data and training limitation

V1 does not use NAB labels to filter training windows, so the autoencoder training procedure is label-blind. However, the split audit shows labeled anomaly periods in the training partitions of 9 out of 10 selected files. V1 includes those windows in training, so it is not guaranteed to be a clean normal-only training set. This is anomaly contamination, not test-label leakage. The exact counts are in `../evaluation/dataset_coverage.csv`.

Threshold calibration is validation-supervised. In 54 of the 90 V1 fit-runs, the validation segment had positive labels; in 36 it had none, so the legacy selection fallback relied on false-positive behavior. This is an inconsistency that V2 attempts to address through a shared validation FPR constraint.

## Later operationalized criteria (supplementary, not the original hypotheses)

The V1 auto-generated hypothesis output added a strict ≥7-of-10 comparison, a ≥15% high-variance FPR-reduction criterion against rolling 3σ, and an AWS CloudWatch ranking. V2 further amends the H1 directional criterion to a proportion of event-evaluable datasets with a minimum sample count. These are later analysis decisions; they must be labeled as operationalizations/amendments and not attributed verbatim to the original proposal. H3 in the original proposal concerns NAB score, false alarms, and/or earlier detection; it is not an AWS domain-ranking hypothesis.

## Metric definitions and caveats

- Pointwise precision, recall, F1, accuracy and FPR are calculated from binary predictions.
- `PR-AUC` is implemented as scikit-learn average precision (AP); label it `PR-AUC / AP` for clarity.
- ROC-AUC uses continuous reconstruction errors, not thresholded predictions.
- Event recall and delay use complete events wholly inside the test partition; partial events crossing a split boundary are excluded from event-level scoring and reported separately.
- Delay fraction is normalized by event duration; raw elapsed seconds are also stored but are not directly comparable across differently sampled series.
- NAB-style score is a test-partition calculation on the subset's complete events, not the official full-corpus NAB leaderboard score.
- Accuracy is secondary because of anomaly class imbalance.
