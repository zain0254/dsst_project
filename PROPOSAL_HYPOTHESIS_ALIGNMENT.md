# Alignment to the Original Research Proposal

This file preserves the original proposal wording and distinguishes it from later protocol changes. The authoritative completed experiment is Final46, stored in source_code/outputs/final46_six_methods/. V1 and V2 are historical evidence and must not be combined with Final46.

## Original proposal hypotheses

- **H1:** The proposed moving Median + MAD threshold will achieve a higher average F1-score than at least one fixed-threshold baseline.
- **H2:** The proposed method will reduce false-positive rates on some time series with changing normal variability.
- **H3:** The proposed method will maintain or improve NAB score by detecting anomalies with fewer false alerts or earlier detection.

## Authoritative Final46 interpretation

Final46 evaluates 46 NAB datasets, three sequence lengths (30, 50, 100), three model seeds (15, 42, 123), 414 LSTM-AE fits, and six threshold methods (2,484 method-level results). The metric populations differ by eligibility.

### H1 — F1 improvement

**Outcome: Not supported on the evaluated datasets.**

- Evaluated population: 18 datasets with positive test points.
- Proposed method: moving median + raw MAD.
- Comparators: fixed percentile and fixed mean + k standard deviations.
- Mean paired difference (proposed minus fixed percentile): -0.202389.
- Mean paired difference (proposed minus fixed mean + k sigma): -0.209002.
- Wilcoxon two-sided p-values: 0.000252 and 0.000191.
- Holm-adjusted p-values: 0.000381 for both comparisons.
- In the paired dataset comparisons, the fixed baselines were better on 17 of 18 datasets in each comparison.

The evidence does not support the proposal's F1-improvement hypothesis.

### H2 — False-positive reduction during changing normal variability

**Outcome: Mixed.**

- Evaluated population: 42 datasets with eligible normal high-variance test points.
- Raw MAD versus fixed percentile: mean difference -0.128489; paired t-test p=0.004218; Holm-adjusted p=0.008436.
- Raw MAD versus fixed mean + k sigma: mean difference -0.130512; p=0.004276; Holm-adjusted p=0.008436.
- Raw MAD versus rolling mean + k sigma: mean difference +0.054894; p=0.000113; Holm-adjusted p=0.000340.

The negative differences against fixed baselines mean that raw MAD has lower high-variance FPR. The positive difference against the rolling comparator means raw MAD has higher high-variance FPR than rolling mean + k sigma. The result is comparator-dependent, not a general adaptive-thresholding victory.

### H3 — NAB-style event performance

**Outcome: Descriptively favorable against fixed baselines, but not statistically established.**

- Evaluated population: 17 datasets with at least one complete test event.
- Raw MAD versus fixed percentile: mean NAB-style score difference +332.740; Wilcoxon p=0.159378; Holm-adjusted p=0.318756.
- Raw MAD versus fixed mean + k sigma: mean difference +324.652; p=0.034790; Holm-adjusted p=0.104370.
- Raw MAD versus rolling mean + k sigma: mean difference +17.195; p=0.963226; Holm-adjusted p=0.963226.

The reported score is a custom NAB-style score on complete events in each held-out test partition. It is not the official full-corpus NAB leaderboard score. The differences do not establish statistical superiority or formal non-inferiority.

## Final46 method extensions

The final benchmark also evaluates:
- Moving median + scaled MAD (1.4826 × raw MAD).
- Moving median + IQR scale (IQR / 1.3489795).

These are exploratory ablations. The original confirmatory hypotheses are about the proposed raw-MAD method; do not rewrite the original hypotheses after seeing the results or claim that the scaled-MAD/IQR methods confirm the original raw-MAD hypothesis.

## Protocol decisions and limitations

- Chronological 60/20/20 partitioning.
- Normal-only training windows selected using NAB labels. This is label-informed normal-only training, not fully label-free unsupervised training.
- Validation labels are used to select threshold parameters under a shared 5% validation FPR target.
- The threshold fallback for validation partitions without positive labels uses the FPR budget rather than trying to optimize uninformative F1.
- Repeated seeds and window sizes are averaged within each dataset before dataset-level statistical comparisons.
- Point-level metrics use positive-test datasets; event-level metrics require complete test events; high-variance FPR requires eligible normal high-variance test points.
- Dataset selection and test-label coverage must be disclosed; the final selected cohort includes 28 datasets with no positive test points.

## Historical V1 and V2 findings

Earlier V1 and V2 experiments are preserved in the repository. They used different cohort sizes, threshold-method sets, and/or protocols. Their findings are historical development evidence only and must not be combined with Final46 aggregates.

- V1: 10 datasets, 90 model fits, 4 methods, 360 method-level rows.
- V2: 23 datasets, 207 model fits, 4 methods, 828 method-level rows.
- Final46: 46 datasets, 414 model fits, 6 methods, 2,484 method-level rows.

See HISTORICAL_EXPERIMENTS/README.md and FINAL_BENCHMARK/FINAL_RESULTS_AND_FINDINGS.md.