# Research-question results generated from the held-out NAB experiment

> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.
> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.

## RQ1 - Baseline performance
How do fixed percentile, fixed mean + k sigma, and validation-calibrated rolling mean + k sigma perform on the selected NAB files?

| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |
|---|---:|---:|---:|---:|
| Fixed percentile | 0.1590 (n=1) | 0.0938 | 0.6134 | -9.51 (n=1) |
| Fixed mean + k sigma | 0.2269 (n=1) | 0.1457 | 0.6134 | -67.26 (n=1) |
| Rolling mean + k sigma (validation-calibrated) | 0.0196 (n=1) | 0.0049 | 0.6134 | 89.62 (n=1) |
| Moving median + lambda MAD (proposed) | 0.2033 (n=1) | 0.0765 | 0.6134 | 36.68 (n=1) |

## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?

| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| f1_score | Fixed percentile | 1 | 0.0443 | [0.0443, 0.0443] | nan |
| f1_score | Fixed mean + k sigma | 1 | -0.0236 | [-0.0236, -0.0236] | nan |
| f1_score | Rolling mean + k sigma (validation-calibrated) | 1 | 0.1837 | [0.1837, 0.1837] | nan |
| pr_auc | Fixed percentile | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| pr_auc | Fixed mean + k sigma | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| pr_auc | Rolling mean + k sigma (validation-calibrated) | 1 | 0.0000 | [0.0000, 0.0000] | nan |
| nab_style_score | Fixed percentile | 1 | 46.1954 | [46.1954, 46.1954] | nan |
| nab_style_score | Fixed mean + k sigma | 1 | 103.9454 | [103.9454, 103.9454] | nan |
| nab_style_score | Rolling mean + k sigma (validation-calibrated) | 1 | -52.9402 | [-52.9402, -52.9402] | nan |

## RQ3 - False alarms during high-variance periods

| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 1 | 0.0567 | 21.6% | 1 | N/A |
| Fixed mean + k sigma | 1 | 0.2057 | 50.0% | 1 | N/A |
| Rolling mean + k sigma (validation-calibrated) | 1 | -0.1986 | -2800.0% | 0 | N/A |

For H2's relative-reduction criterion, inspect `high_variance_fpr_by_dataset.csv` and the paired comparison output; percentage reductions are only defined when the baseline high-variance FPR is greater than zero.

## RQ4 - Which NAB domain benefits most?

| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |
|---|---:|---:|---:|---:|---:|
| AWS CloudWatch | 1 | 0.1837 | [0.1837, 0.1837] | 1 | 1 |

Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.

## Original proposal hypothesis checks

| Hypothesis | Outcome | Evaluable datasets | Interpretation |
|---|---|---:|---|
| H1: Proposed moving Median + MAD achieves higher average F1 than at least one fixed-threshold baseline | Directional only; not statistically established | 1 | The inferential sample is event-evaluable datasets only; repeated seeds and windows are averaged within dataset. |
| H2: Proposed threshold reduces false-positive alerts on some series with changing normal variability | Partially observed on some series; aggregate evidence is inconclusive | 1 | A per-series win is not presented as a general benefit when the aggregate false-alarm rate is worse. |
| H3: Proposed method maintains or improves NAB score through fewer false alerts or earlier detection | Not supported overall by the available NAB-style event-score evidence | 1 | NAB-style scores are computed over complete events in the held-out test segment; they are not official full-corpus NAB leaderboard scores. |

## Interpretation rules
- The validation partition selects threshold hyperparameters; only the test partition is used for final performance reporting.
- Paired statistical comparisons use one averaged observation per dataset; seeds and sequence lengths are not independent datasets.
- H1 is compared with the two fixed-threshold baselines in the proposal. The additional rolling baseline is supplementary.
- H2 reports per-series reductions as well as the aggregate direction; isolated wins are not evidence of a general false-alarm reduction.
- H3 is evaluated using NAB-style event score, false alarms, and detection timing. Domain comparisons are a separate exploratory RQ4 analysis.
- The original proposal does not prescribe 70%, 15%, or a fixed minimum number of evaluable datasets; any such rule must be labelled as an amendment.
- State the number of evaluable files alongside every aggregate result. A file with no test anomaly event cannot support event-level NAB claims.
