# Version 2 benchmark protocol (submission-focused revision)

## Why Version 2 exists

Version 1 is preserved unchanged as the original 10-file, 3-window, 3-seed benchmark. Version 2 addresses the two largest weaknesses visible in V1: low positive-event test coverage and the proposed method's excessive false-positive rate.

## Dataset expansion

Run `python prepare_expanded_nab_manifest.py --additional 15` from this directory. The script uses the NAB `combined_windows.json` already in `data/nab/labels/`, downloads candidate **real** NAB files from the official NAB GitHub raw repository, and screens them before adding them to `config_v2.json`.

A new file is eligible only if it has:
- at least 600 valid rows;
- at least one complete labeled anomaly event wholly inside the final chronological 20% test partition, with both event endpoints present;
- positive test labels; and
- at least 100 fully normal W=100 training windows and 50 fully normal W=100 validation windows for normal-only training and early stopping.

The script excludes artificial NAB series, attempts to balance additions across AWS CloudWatch, Ad Exchange, Known Cause, Traffic and Twitter, and writes `expanded_dataset_selection.json` with selected and rejected candidates. In the completed run, 13 additional datasets passed. The original 10 V1 datasets were retained without reapplying these criteria: four have complete positive test events and six have no positive test labels. The 23-dataset run therefore has 17 event-evaluable datasets and six negative-only test datasets. Do not describe all 23 as eligible under the event-coverage screen.

## Training protocol

Version 2 uses chronological 60/20/20 partitions. The scaler is fitted only on normal training rows. Training windows are retained only when **every point** in the complete W-length window is labeled normal. Validation and test windows retain causal historical context. For early stopping only, validation loss is calculated on fully normal validation windows; the complete validation partition is still used for threshold calibration. Test windows are never filtered by labels.

This is a label-filtered, contamination-free training experiment using NAB ground truth to construct a clean training set. It is not a fully label-free deployment protocol. Describe it as normal-only / label-filtered training, not purely unsupervised training.

## Threshold calibration

All threshold methods use the same validation selection rule and a predeclared validation FPR budget of 5%:
1. Among candidates meeting the FPR budget, select the highest validation F1.
2. If no candidate meets the budget, select the candidate with the lowest validation FPR, then highest F1.
3. If validation has no positive labels, the rule naturally selects the lowest-FPR candidate.

The rolling mean baseline is calibrated over the same k grid as the fixed mean+sigma baseline. It must be described as rolling mean + k sigma (validation-calibrated), not as a fixed 3-sigma baseline. The internal legacy identifier `rolling_mean_3sigma` is retained for compatibility with saved prediction-column names.

The proposed MAD method uses an expanded lambda grid `[0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, 100]` and a small scale-relative MAD floor to avoid a collapsed threshold when recent-history MAD is zero.

## Model training

The epoch cap is increased from 20 to 40 and early-stopping patience from 4 to 6. This allows the longer datasets to continue improving while retaining validation-loss early stopping. Report actual epochs per fit; do not claim convergence merely because the cap was increased.

## Reproducibility and comparison

Use the same three windows `[30, 50, 100]` and three seeds `[15, 42, 123]`. The completed matrix contains 23 selected datasets × 3 windows × 3 seeds = 207 model fits. Seventeen datasets are event-evaluable in the test partition; six legacy datasets have no positive test labels and are retained for supplementary false-alarm analysis. Methods are evaluated on the same model scores.

Run into a new output directory; never overwrite Version 1:

```bash
python run_experiment.py --config config_v2.json --output-dir outputs/version2_reproduction_fresh --device cuda --skip-eda
```

Do not compare V1 and V2 aggregate values as if they were one experiment. V2 changes dataset coverage, training contamination, calibration objective, rolling baseline tuning and epoch budget. Present V1 as the original benchmark and V2 as a revised protocol / follow-up experiment.

## Hypothesis reporting and alignment to the original proposal

The original proposal's hypotheses are:

- H1: the proposed moving Median + MAD threshold will achieve a higher average F1-score than at least one fixed-threshold baseline.
- H2: the proposed method will reduce false-positive rates on some time series with changing normal variability.
- H3: the proposed method will maintain or improve NAB score by detecting anomalies with fewer false alerts or earlier detection.

These original statements do **not** specify a 7-of-10 rule, a 15% minimum FPR reduction, or an AWS ranking requirement. Any such fixed quantitative criterion is a later operationalization and must be labeled as a supplementary/amended criterion. In particular, the V1 auto-generated outcomes table introduced a 7-of-10 H1 rule, a 15% FPR reduction requirement versus rolling 3-sigma for H2, and an AWS domain ranking for H3. The AWS ranking is an RQ4-style domain analysis, not the original H3.

The V2 output code now reports the original proposal-aligned H1–H3 decisions directly: H1 compares F1 with the two fixed baselines; H2 includes per-file high-variance FPR changes and paired aggregate tests; H3 evaluates NAB-style event score with false-alert and detection-delay context. Domain ranking remains a separate exploratory RQ4 result. Do not apply the optional 70%-of-evaluable-files/minimum-15 rule unless it is explicitly declared as a later amendment before running the benchmark.

## NAB score scope

The evaluator calculates a NAB-style score on complete events inside each held-out test partition. This is not the official full-corpus NAB leaderboard evaluation.

For H2, the 75th-percentile high-variance cutoff is computed from full trailing volatility windows in the training partition whose historical points are all labeled normal. The current point is excluded, preserving causality. This avoids labeled training anomalies inflating the reference distribution. Continue to label it `NAB-style test-partition score`.
