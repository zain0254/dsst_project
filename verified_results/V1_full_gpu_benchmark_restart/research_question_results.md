# Research-question results generated from the held-out NAB experiment

> This file is generated from the actual `test_metrics.csv` output. It must not be filled with guessed or synthetic results.
> `NAB-style test score` applies NAB Standard-profile timing rewards to complete anomaly windows in the test subset; it is not the official full-corpus leaderboard score.

## RQ1 - Baseline performance
How do fixed percentile, fixed mean + k sigma, and rolling mean + 3 sigma perform on the selected NAB files?

| Method | Mean F1 (N) | Mean FPR | Mean PR-AUC | Mean NAB-style test score (N) |
|---|---:|---:|---:|---:|
| Fixed percentile | 0.0905 (n=10) | 0.0603 | 0.4428 | -270.56 (n=4) |
| Fixed mean + k sigma | 0.0925 (n=10) | 0.0893 | 0.4428 | -326.14 (n=4) |
| Rolling mean + 3 sigma (proposal baseline) | 0.0301 (n=10) | 0.0424 | 0.4428 | -330.63 (n=4) |
| Moving median + lambda MAD (proposed) | 0.0966 (n=10) | 0.2348 | 0.4428 | -2027.22 (n=4) |

## RQ2 - Does moving median + lambda MAD improve F1 and NAB-style score?

| Metric | Baseline | N datasets | Mean improvement (directional) | 95% bootstrap CI | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| f1_score | Fixed percentile | 4 | 0.0151 | [-0.0302, 0.0574] | 1 |
| f1_score | Fixed mean + k sigma | 4 | 0.0102 | [-0.0920, 0.1125] | 1 |
| f1_score | Rolling mean + 3 sigma (proposal baseline) | 4 | 0.1663 | [0.0614, 0.2382] | 0.375 |
| pr_auc | Fixed percentile | 4 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Fixed mean + k sigma | 4 | 0.0000 | [0.0000, 0.0000] | 1 |
| pr_auc | Rolling mean + 3 sigma (proposal baseline) | 4 | 0.0000 | [0.0000, 0.0000] | 1 |
| nab_style_score | Fixed percentile | 4 | -1756.6635 | [-5177.6966, 151.9287] | 0.75 |
| nab_style_score | Fixed mean + k sigma | 4 | -1701.0763 | [-5472.5998, 461.3502] | 0.875 |
| nab_style_score | Rolling mean + 3 sigma (proposal baseline) | 4 | -1696.5851 | [-4589.7532, -122.2394] | 0.375 |

## RQ3 - False alarms during high-variance periods

| Baseline | N datasets | Mean absolute FPR reduction | Mean relative reduction | Files with >=15% reduction | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 8 | -0.1968 | -351.1% | 2 | 0.2149 |
| Fixed mean + k sigma | 8 | -0.1409 | -164.5% | 2 | 0.254 |
| Rolling mean + 3 sigma (proposal baseline) | 8 | -0.3211 | -424.1% | 0 | 0.00295 |

For H2's relative-reduction criterion, inspect `high_variance_fpr_by_dataset.csv` and the paired comparison output; percentage reductions are only defined when the baseline high-variance FPR is greater than zero.

## RQ4 - Which NAB domain benefits most?

| Domain | Datasets | Mean F1 improvement | 95% bootstrap CI | Positive datasets | Rank |
|---|---:|---:|---:|---:|---:|
| AWS CloudWatch | 1 | 0.2200 | [0.2200, 0.2200] | 1 | 1 |
| Known Cause | 3 | 0.1484 | [0.0085, 0.2563] | 3 | 2 |

Domain comparisons are exploratory and descriptive because each category contains only a small number of selected files. A first-place mean does not establish a statistically generalizable domain effect.

## Proposal hypothesis checks

| Hypothesis | Outcome | Evidence count | Evaluated / configured | p-value |
|---|---|---:|---:|---:|
| H1: Median + MAD improves F1 on at least 7 of 10 datasets | Not supported / inconclusive | 4 | 4 / 10 | 0.125 |
| H2: Median + MAD reduces high-variance FPR by at least 15% | Not supported / inconclusive | 0 | N/A / 10 | 0.0009834 |
| H3: AWS CloudWatch has the greatest improvement | Not supported descriptively | 1 | N/A / 2 | N/A |

## Interpretation rules
- The validation partition selects threshold hyperparameters; only the test partition is used for the final reported performance.
- Paired statistical comparisons use one averaged observation per dataset, not each seed/window as an independent sample.
- A file with no complete labeled test event is not evidence for anomaly-F1 superiority or event-level NAB-style score.
- `Not supported / inconclusive` means the pre-defined criterion was not met or the number of evaluable files was insufficient; it is not proof that the methods are equivalent.
- State the number of evaluable files alongside every aggregate result.
