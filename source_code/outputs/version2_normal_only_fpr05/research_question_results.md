# Research-question results generated from the held-out NAB experiment

> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.
> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.

## RQ1 - Baseline performance
How do fixed percentile, fixed mean + k sigma, and validation-calibrated rolling mean + k sigma perform on the selected NAB files?

| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |
|---|---:|---:|---:|---:|
| Fixed percentile | 0.3071 (n=23) | 0.1386 | 0.5615 | -368.67 (n=17) |
| Fixed mean + k sigma | 0.3097 (n=23) | 0.1342 | 0.5615 | -330.95 (n=17) |
| Rolling mean + k sigma (validation-calibrated) | 0.0486 (n=23) | 0.0190 | 0.5615 | -23.64 (n=17) |
| Moving median + lambda MAD (proposed) | 0.1280 (n=23) | 0.0299 | 0.5615 | -27.37 (n=17) |

## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?

| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| f1_score | Fixed percentile | 17 | -0.2423 | [-0.3286, -0.1678] | 4.578e-05 |
| f1_score | Fixed mean + k sigma | 17 | -0.2458 | [-0.3333, -0.1679] | 4.578e-05 |
| f1_score | Rolling mean + k sigma (validation-calibrated) | 17 | 0.1074 | [0.0626, 0.1524] | 0.0003815 |
| pr_auc | Fixed percentile | 17 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Fixed mean + k sigma | 17 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Rolling mean + k sigma (validation-calibrated) | 17 | 0.0000 | [0.0000, 0.0000] | 1 |
| nab_style_score | Fixed percentile | 17 | 341.3074 | [17.8530, 769.8340] | 0.2647 |
| nab_style_score | Fixed mean + k sigma | 17 | 303.5812 | [14.0918, 706.0337] | 0.1915 |
| nab_style_score | Rolling mean + k sigma (validation-calibrated) | 17 | -3.7265 | [-57.3851, 70.7333] | 0.2647 |

## RQ3 - False alarms during high-variance periods

| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 21 | 0.1474 | 15.0% | 11 | 0.06277 |
| Fixed mean + k sigma | 21 | 0.1373 | 15.1% | 13 | 0.06277 |
| Rolling mean + k sigma (validation-calibrated) | 21 | -0.0428 | -725.2% | 5 | 0.06277 |

The 15% relative-reduction column is a supplementary descriptive summary, not a threshold stated in the original proposal. Per-series values are available in `high_variance_fpr_by_dataset.csv`.

## RQ4 - Which NAB domain benefits most?

| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |
|---|---:|---:|---:|---:|---:|
| Ad Exchange | 1 | 0.2577 | [0.2577, 0.2577] | 1 | 1 |
| Traffic | 4 | 0.1555 | [0.0906, 0.2274] | 4 | 2 |
| Twitter | 3 | 0.0954 | [0.0147, 0.1396] | 3 | 3 |
| AWS CloudWatch | 4 | 0.0875 | [-0.0113, 0.1703] | 3 | 4 |
| Known Cause | 5 | 0.0622 | [-0.0002, 0.1558] | 3 | 5 |

Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.

## Original proposal hypothesis checks

| Hypothesis | Outcome | Evaluable datasets | Interpretation |
|---|---|---:|---|
| H1: Proposed moving Median + MAD achieves higher average F1 than at least one fixed-threshold baseline | Not supported on evaluated datasets | 17 | The inferential sample is event-evaluable datasets only; repeated seeds and windows are averaged within dataset. |
| H2: Proposed threshold reduces false-positive alerts on some series with changing normal variability | Partially observed on some series; aggregate evidence is inconclusive | 21 | A per-series win is not presented as a general benefit when the aggregate false-alarm rate is worse. |
| H3: Proposed method maintains or improves NAB score through fewer false alerts or earlier detection | Potentially supported descriptively; statistical power is limited | 17 | NAB-style scores are computed over complete events in the held-out test segment; they are not official full-corpus NAB leaderboard scores. |

## Interpretation rules
- The validation partition selects threshold hyperparameters; only the test partition is used for final performance reporting.
- Paired statistical comparisons use one averaged observation per dataset; seeds and sequence lengths are not independent datasets.
- H1 is compared with the two fixed-threshold baselines in the proposal. The additional rolling baseline is supplementary.
- H2 reports per-series reductions as well as the aggregate direction; isolated wins are not evidence of a general false-alarm reduction.
- H3 is evaluated using NAB-style event score, false alarms, and detection timing. Domain comparisons are a separate exploratory RQ4 analysis.
- The original proposal does not prescribe 70%, 15%, or a fixed minimum number of evaluable datasets; any such rule must be labelled as an amendment.
- State the number of evaluable files alongside every aggregate result. A file with no test anomaly event cannot support event-level NAB claims.
