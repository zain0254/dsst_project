# Research-question results generated from the held-out NAB experiment

> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.
> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.

## RQ1 - Baseline performance
How do fixed percentile, fixed mean + k sigma, and validation-calibrated rolling mean + k sigma perform on the selected NAB files?

| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |
|---|---:|---:|---:|---:|
| Fixed percentile | 0.4280 (n=18) | 0.1425 | 0.5557 | -425.26 (n=17) |
| Fixed mean + k sigma | 0.4346 (n=18) | 0.1423 | 0.5557 | -417.18 (n=17) |
| Rolling mean + k sigma (validation-calibrated) | 0.1333 (n=18) | 0.0501 | 0.5557 | -109.72 (n=17) |
| Moving median + lambda raw MAD (original) | 0.2256 (n=18) | 0.0615 | 0.5557 | -92.52 (n=17) |
| Moving median + k scaled MAD (1.4826 x MAD) | 0.2421 (n=18) | 0.0729 | 0.5557 | -143.71 (n=17) |
| Moving median + k IQR scale (IQR / 1.349) | 0.1695 (n=18) | 0.0563 | 0.5557 | -109.49 (n=17) |

## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?

> Holm correction in this RQ2 table is applied across three pairwise comparators: fixed percentile, fixed mean + k sigma, and rolling mean + k sigma. This differs from the proposal-specific H1 correction in statistical_tests.csv, which covers only the two fixed-threshold comparators stated in the original hypothesis. Use the H1-specific adjusted p-values in statistical_tests.csv when reporting H1; the different adjusted values reflect different correction families, not different raw tests.

| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| f1_score | Fixed percentile | 18 | -0.2024 | [-0.3114, -0.1091] | 0.0005722 |
| f1_score | Fixed mean + k sigma | 18 | -0.2090 | [-0.3211, -0.1106] | 0.0005722 |
| f1_score | Rolling mean + k sigma (validation-calibrated) | 18 | 0.0923 | [0.0486, 0.1391] | 0.001579 |
| pr_auc | Fixed percentile | 18 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Fixed mean + k sigma | 18 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Rolling mean + k sigma (validation-calibrated) | 18 | 0.0000 | [0.0000, 0.0000] | 1 |
| nab_style_score | Fixed percentile | 17 | 332.7401 | [13.4121, 741.4925] | 0.3188 |
| nab_style_score | Fixed mean + k sigma | 17 | 324.6524 | [16.5291, 724.7141] | 0.1044 |
| nab_style_score | Rolling mean + k sigma (validation-calibrated) | 17 | 17.1954 | [-44.4313, 95.8375] | 0.9632 |

## RQ3 - False alarms during high-variance periods

| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 42 | 0.1285 | -83.8% | 21 | 0.008436 |
| Fixed mean + k sigma | 42 | 0.1305 | -39.9% | 25 | 0.008436 |
| Rolling mean + k sigma (validation-calibrated) | 42 | -0.0549 | -274.5% | 10 | 0.00034 |

The 15% relative-reduction column is a supplementary descriptive summary, not a threshold stated in the original proposal. Per-series values are available in `high_variance_fpr_by_dataset.csv`.

## RQ4 - Which NAB domain benefits most?

| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |
|---|---:|---:|---:|---:|---:|
| Ad Exchange | 1 | 0.2577 | [0.2577, 0.2577] | 1 | 1 |
| Traffic | 4 | 0.1309 | [0.0581, 0.2168] | 4 | 2 |
| AWS CloudWatch | 5 | 0.0978 | [0.0035, 0.1925] | 4 | 3 |
| Twitter | 3 | 0.0679 | [0.0160, 0.1357] | 3 | 4 |
| Known Cause | 5 | 0.0376 | [-0.0243, 0.1145] | 3 | 5 |

Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.

## Original proposal hypothesis checks

| Hypothesis | Outcome | Evaluable datasets | Interpretation |
|---|---|---:|---|
| H1: Proposed moving Median + MAD achieves higher average F1 than at least one fixed-threshold baseline | Not supported on evaluated datasets | 18 | The inferential sample is point-evaluable datasets only (at least one positive held-out test point); repeated seeds and windows are averaged within dataset. |
| H2: Proposed threshold reduces false-positive alerts on some series with changing normal variability | Mixed: isolated per-series reductions, but significantly higher mean FPR than rolling baseline | 42 | A per-series win is not presented as a general benefit when the aggregate false-alarm rate is worse. |
| H3: Proposed method maintains or improves NAB score through fewer false alerts or earlier detection | Potentially supported descriptively; statistical power is limited | 17 | NAB-style scores are computed over complete events in the held-out test segment; they are not official full-corpus NAB leaderboard scores. |

## Interpretation rules
- The validation partition selects threshold hyperparameters; only the test partition is used for final performance reporting.
- Paired statistical comparisons use one averaged observation per dataset; seeds and sequence lengths are not independent datasets.
- H1 is compared with the two fixed-threshold baselines in the proposal. The additional rolling baseline is supplementary.
- H2 reports per-series reductions as well as the aggregate direction; isolated wins are not evidence of a general false-alarm reduction.
- H3 is evaluated using NAB-style event score, false alarms, and detection timing. Domain comparisons are a separate exploratory RQ4 analysis.
- The original proposal does not prescribe 70%, 15%, or a fixed minimum number of evaluable datasets; any such rule must be labelled as an amendment.
- State the number of evaluable files alongside every aggregate result. A file with no test anomaly event cannot support event-level NAB claims.
