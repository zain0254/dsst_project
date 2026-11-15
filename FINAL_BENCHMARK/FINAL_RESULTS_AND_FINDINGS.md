# Final46 — Results and Findings

## Source and averaging

All descriptive method means below are taken from `source_code/outputs/final46_six_methods/benchmark_method_summary.csv`. The table reports dataset-level means and the valid dataset count for each metric. The valid population changes by metric; do not compare columns as if they share one denominator.

## Six-method results

| Method | F1 mean (n) | FPR mean (n) | Event recall (n) | NAB-style score (n) | High-variance FPR (n) |
|---|---:|---:|---:|---:|---:|
| Fixed percentile | 0.4280 (18) | 0.1425 (46) | 0.9684 (17) | -425.26 (17) | 0.2449 (42) |
| Fixed mean + k sigma | 0.4346 (18) | 0.1423 (46) | 0.9706 (17) | -417.18 (17) | 0.2469 (42) |
| Rolling mean + k sigma, validation-calibrated | 0.1333 (18) | 0.0501 (46) | 0.9466 (17) | -109.72 (17) | 0.0615 (42) |
| Moving median + raw MAD (proposed) | 0.2256 (18) | 0.0615 (46) | 0.8606 (17) | -92.52 (17) | 0.1164 (42) |
| Moving median + scaled MAD | 0.2421 (18) | 0.0729 (46) | 0.9150 (17) | -143.71 (17) | 0.1400 (42) |
| Moving median + IQR scale | 0.1695 (18) | 0.0563 (46) | 0.8780 (17) | -109.49 (17) | 0.0864 (42) |

Higher F1, event recall, and NAB-style score are better; lower FPR is better. Negative NAB-style scores are possible under the custom penalty/reward normalization and do not mean the computation failed.

## Statistical hypotheses

The primary inferential source is `statistical_tests.csv`, with outcomes in `hypothesis_outcomes.csv`. Comparisons use dataset-level paired observations after averaging repeated seeds and window lengths within each dataset.

### H1 — F1 improvement

**Not supported.** On 18 positive-test datasets, raw MAD is significantly worse than both fixed baselines:
- vs fixed percentile: mean paired difference -0.202389; Wilcoxon p=0.0002518; Holm p=0.0003815.
- vs fixed mean + k sigma: mean paired difference -0.209002; p=0.0001907; Holm p=0.0003815.

The fixed baseline is better on 17 of 18 paired datasets in each comparison.

### H2 — high-variance false-positive reduction

**Mixed and comparator-dependent.** On 42 datasets:
- vs fixed percentile: mean difference -0.128489; paired t-test p=0.004218; Holm p=0.008436. Negative means raw MAD has lower high-variance FPR.
- vs fixed mean + k sigma: mean difference -0.130512; p=0.004276; Holm p=0.008436.
- vs rolling mean + k sigma: mean difference +0.054894; p=0.0001133; Holm p=0.0003400. Positive means raw MAD has higher high-variance FPR.

### H3 — event-level NAB-style performance

**Descriptively favorable against the fixed baselines, but not statistically established.** On 17 complete-event datasets:
- vs fixed percentile: mean difference +332.740; Wilcoxon p=0.159378; Holm p=0.318756.
- vs fixed mean + k sigma: mean difference +324.652; p=0.034790; Holm p=0.104370.
- vs rolling comparator: mean difference +17.195; p=0.963226; Holm p=0.963226.

No comparison is significant after Holm correction. This is not a formal non-inferiority test.

## Why two H1 Holm-adjusted values appear

The two values come from **different Holm correction families**, not from different raw p-values or a calculation error:

- The proposal-specific H1 test in `statistical_tests.csv` corrects the two comparisons named by the original hypothesis: raw MAD versus fixed percentile and raw MAD versus fixed mean + k sigma. The adjusted p-value is **0.0003815**.
- The broader RQ2 table in `pairwise_proposed_vs_baselines.csv` includes those two fixed baselines plus the additional rolling mean + k sigma comparator. Holm correction across all three comparisons gives **0.0005722** for the two fixed-baseline rows.

The raw p-values are unchanged. For reporting the original H1 hypothesis, use the H1-specific correction family and quote **0.0003815** from `statistical_tests.csv`. For the broader three-comparator RQ2 analysis, use **0.0005722** and state that the rolling comparator is included in the correction family. The generated RQ report now explains this distinction.

## Interpretation

The raw-MAD method is not an overall winner. It reduces high-variance false alarms against fixed thresholds, but this comes with lower point-level F1 and lower event recall. Its high-variance false-alarm rate is worse than the rolling comparator. Event-level NAB-style averages are descriptively higher than fixed baselines but not statistically significant after correction.

The findings support a nuanced trade-off and identify a need for further research; they do not support a universal superiority claim.
