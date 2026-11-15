# Research-question results generated from the held-out NAB experiment

> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.
> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.

## RQ1 - Baseline performance
How do fixed percentile, fixed mean + k sigma, and validation-calibrated rolling mean + k sigma perform on the selected NAB files?

| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |
|---|---:|---:|---:|---:|
| Fixed percentile | 0.2242 (n=1) | 0.2264 | 0.2523 | -1128.36 (n=1) |
| Fixed mean + k sigma | 0.2129 (n=1) | 0.2136 | 0.2523 | -1058.62 (n=1) |
| Rolling mean + k sigma (validation-calibrated) | 0.0930 (n=1) | 0.0807 | 0.2523 | -332.70 (n=1) |
| Moving median + lambda raw MAD (original) | 0.0952 (n=1) | 0.1072 | 0.2523 | -486.45 (n=1) |
| Moving median + k scaled MAD (1.4826 x MAD) | 0.0939 (n=1) | 0.1137 | 0.2523 | -518.76 (n=1) |
| Moving median + k IQR scale (IQR / 1.349) | 0.0797 (n=1) | 0.0871 | 0.2523 | -366.99 (n=1) |

## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?

| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| f1_score | Fixed percentile | 1 | -0.1289 | [-0.1289, -0.1289] | nan |
| f1_score | Fixed mean + k sigma | 1 | -0.1177 | [-0.1177, -0.1177] | nan |
| f1_score | Rolling mean + k sigma (validation-calibrated) | 1 | 0.0022 | [0.0022, 0.0022] | nan |
| pr_auc | Fixed percentile | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| pr_auc | Fixed mean + k sigma | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| pr_auc | Rolling mean + k sigma (validation-calibrated) | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| nab_style_score | Fixed percentile | 1 | 641.9034 | [641.9034, 641.9034] | nan |
| nab_style_score | Fixed mean + k sigma | 1 | 572.1651 | [572.1651, 572.1651] | nan |
| nab_style_score | Rolling mean + k sigma (validation-calibrated) | 1 | -153.7572 | [-153.7572, -153.7572] | nan |

## RQ3 - False alarms during high-variance periods

| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 1 | 0.1697 | 61.7% | 1 | N/A |
| Fixed mean + k sigma | 1 | 0.1588 | 60.1% | 1 | N/A |
| Rolling mean + k sigma (validation-calibrated) | 1 | -0.0461 | -77.6% | 0 | N/A |

The 15% relative-reduction column is a supplementary descriptive summary, not a threshold stated in the original proposal. Per-series values are available in `high_variance_fpr_by_dataset.csv`.

## RQ4 - Which NAB domain benefits most?

| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |
|---|---:|---:|---:|---:|---:|
| Known Cause | 1 | 0.0022 | [0.0022, 0.0022] | 1 | 1 |

Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.

## Original proposal hypothesis checks

| Hypothesis | Outcome | Evaluable datasets | Interpretation |
|---|---|---:|---|
| H1: Proposed moving Median + MAD achieves higher average F1 than at least one fixed-threshold baseline | Not supported on evaluated datasets | 1 | The inferential sample is event-evaluable datasets only; repeated seeds and windows are averaged within dataset. |
| H2: Proposed threshold reduces false-positive alerts on some series with changing normal variability | Partially observed on some series; aggregate evidence is inconclusive | 1 | A per-series win is not presented as a general benefit when the aggregate false-alarm rate is worse. |
| H3: Proposed method maintains or improves NAB score through fewer false alerts or earlier detection | Potentially supported descriptively; statistical power is limited | 1 | NAB-style scores are computed over complete events in the held-out test segment; they are not official full-corpus NAB leaderboard scores. |

## Interpretation rules
- The validation partition selects threshold hyperparameters; only the test partition is used for final performance reporting.
- Paired statistical comparisons use one averaged observation per dataset; seeds and sequence lengths are not independent datasets.
- H1 is compared with the two fixed-threshold baselines in the proposal. The additional rolling baseline is supplementary.
- H2 reports per-series reductions as well as the aggregate direction; isolated wins are not evidence of a general false-alarm reduction.
- H3 is evaluated using NAB-style event score, false alarms, and detection timing. Domain comparisons are a separate exploratory RQ4 analysis.
- The original proposal does not prescribe 70%, 15%, or a fixed minimum number of evaluable datasets; any such rule must be labelled as an amendment.
- State the number of evaluable files alongside every aggregate result. A file with no test anomaly event cannot support event-level NAB claims.
