# Review Build Changelog — 15 November 2026

This review build was created from the uploaded `LSTM_AE_CURRENT_FULL_REVIEW.zip` package.

## Corrections made

1. **Dataset cohort disclosure:** Corrected the project documentation to state that the run contains 23 selected datasets, of which 17 are event-evaluable in the held-out test segment and six have no positive test labels. The 13 added datasets passed the screening process; the original 10 V1 datasets were retained without reapplying the new screen. A dataset-by-dataset audit is included in `DATASET_COHORT_AUDIT.csv` and `DATASET_COHORT_AUDIT.md`.
2. **High-variance FPR reference:** Updated the high-variance cutoff to use only full trailing training histories whose historical points are all labeled normal. The current point is excluded from each history window. This prevents labeled training anomalies from inflating the reference distribution for normal variability.
3. **Post-hoc H2 reanalysis:** Recalculated all 828 method-level metric rows from the 207 saved prediction traces. No model was retrained. Only `high_variance_fpr` and `high_variance_normal_points` changed; every other metric reproduced within an absolute difference of `4.6e-13`. The original prediction traces remain in the original benchmark output directory and were not duplicated or modified.
4. **H1 decision logic:** Corrected the hypothesis decision so a positive mean difference and a significant Holm-adjusted p-value must refer to the same baseline comparison.
5. **Configuration consistency:** Passed the configured validation FPR budget and MAD relative floor explicitly into threshold tuning. Their current values match the previous function defaults.
6. **Rolling comparator naming:** Corrected the documentation to describe the V2 comparator as validation-calibrated rolling mean + kσ. The legacy internal identifier `rolling_mean_3sigma` remains unchanged for compatibility with saved output columns.
7. **Reproducibility instructions:** Corrected the ZIP extraction path, WSL virtual-environment activation, test count, dataset-selection explanation, and safe fresh-output reproduction command.

## Verification performed

- Python compilation passed.
- All 20 automated tests passed.
- A one-dataset, one-window, one-seed CPU smoke run completed and produced four method-level rows.
- All 207 saved prediction files and 828 metric rows were independently audited.
- The clean-normal volatility reanalysis recalculated 828 rows and verified unchanged metrics within `4.6e-13`.
- The reanalysis audit JSON parses as strict JSON.
- The original V1 and V2 benchmark evidence remains preserved separately.

## What was not done

The 207-model GPU benchmark was not retrained. The saved model outputs were reused because the corrections affect the high-variance evaluation mask and reporting logic, not the trained LSTM-AE weights or reconstruction-error traces. A fresh full GPU run remains optional for an independent end-to-end reproduction.

## Research interpretation

- H1 is not supported: the fixed baselines significantly outperform proposed MAD on F1 across the 17 event-evaluable datasets.
- H2 has some per-dataset reductions versus fixed baselines, but aggregate evidence is inconclusive after Holm correction on 20 paired datasets. MAD remains worse than the rolling kσ baseline on macro high-variance FPR.
- H3 is descriptive only: the proposed method's custom NAB-style test score is better than the fixed baselines on average but not significantly so, and it does not beat the rolling comparator. Its event recall is lower than the fixed baselines and its mean delay fraction is higher than all three comparators.

The project score remains a **NAB-style test-partition score**, not an official full-corpus NAB leaderboard score.


## 2026-11-15 — Threshold-ablation candidate build

- Added moving median + normal-scaled MAD (1.4826 × raw MAD).
- Added moving median + IQR scale (IQR / 1.3489795).
- Retained raw MAD as a distinct method.
- Corrected validation candidate selection when no positive labels occur: choose the candidate closest to, but not exceeding, the predeclared validation FPR budget; if none meet it, minimize FPR. The same fallback applies to all methods for fair calibration. Positive-validation selection remains maximize F1 under the FPR budget.
- Historical V2 audit found 117/207 model runs with no positive validation labels; this makes the fallback consequential and worth testing.
- Derived robust-scale floors from training reconstruction errors only, so validation/test spikes cannot inflate the scale floor. The rolling history remains causal and is initialized from the preceding train-plus-validation stream.
- Added compressed calibration traces (train/validation/test reconstruction errors) to avoid retraining for future threshold-only ablations.
- Added dataset-macro descriptive threshold-ablation summary; new variants are exploratory and are not added to the original confirmatory hypotheses.
- Automated suite: 22 tests passed. One-dataset CPU smoke experiment completed with six methods and calibration trace output. Full 207-fit GPU candidate benchmark remains pending.
