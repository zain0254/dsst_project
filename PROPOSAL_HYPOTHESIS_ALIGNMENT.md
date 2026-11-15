# Alignment to the Original Research Proposal

This file preserves the proposal's own wording as the primary frame for the thesis conclusions. Later V1 implementation criteria and V2 operational criteria are reported as additional protocol decisions, not silently substituted for the original hypotheses.

## Original proposal hypotheses (verbatim meaning)

- **H1:** The proposed moving Median + MAD threshold will achieve a higher average F1-score than at least one fixed-threshold baseline.
- **H2:** The proposed method will reduce false-positive rates on some time series with changing normal variability.
- **H3:** The proposed method will maintain or improve NAB score by detecting anomalies with fewer false alerts or earlier detection.

## How the completed V1 evidence maps to those hypotheses

- **H1:** Mean F1 is directionally higher for the proposed method than for both fixed thresholds. Inferential comparisons use only four test-positive datasets, and neither comparison is statistically significant after Holm correction. Report as a directional observation, not a confirmed improvement.
- **H2:** A few individual datasets show lower high-variance FPR versus fixed thresholds, but the overall high-variance FPR is much higher. Against rolling mean + 3σ, the increase is significant after Holm correction. Report mixed per-file findings and an adverse aggregate, not general success.
- **H3:** The proposed method detects all seven complete events and has a lower normalized delay fraction. But false alarms increase sharply and the NAB-style test-partition score is markedly worse. The outcome is not supported overall.

## How the completed V2 benchmark maps to those hypotheses

- **Cohort:** V2 contains 23 selected datasets. Seventeen have complete positive test events; six retained legacy datasets have no positive test labels. The six contribute to point-level false-alarm analysis but are excluded from event-based hypothesis denominators. See `reports/DATASET_COHORT_AUDIT.md`.
- **H1:** Not supported. On the 17 event-evaluable datasets, the proposed method has lower mean F1 than both fixed baselines; the fixed baselines are significantly better in paired Wilcoxon tests after Holm correction.
- **H2:** Some dataset-level reductions are observed. After defining high variability from fully normal training histories, the proposed method has lower macro high-variance FPR than both fixed baselines, but higher FPR than the rolling mean + k-sigma comparator. Aggregate tests are inconclusive after Holm correction (20 paired datasets).
- **H3:** Descriptive trade-off only. The proposed method has a higher mean NAB-style test-partition score than the two fixed baselines, but the differences are not significant after Holm correction; it does not outperform the rolling comparator. Its event recall is lower than the fixed baselines and its delay fraction is higher than all three comparators. Do not claim a statistically established overall improvement.

The V2 rolling comparator is validation-calibrated rolling mean + kσ, not a fixed 3σ method. The H2 reanalysis uses the clean-normal training-history reference cutoff and does not retrain the models.

## Later criteria that must remain explicitly labeled as extensions

The generated V1 outcomes table used (a) a 7-of-10 F1 threshold, (b) at least a 15% relative high-variance FPR reduction against rolling 3σ, and (c) an AWS CloudWatch domain ranking. Those are operationalized/revised criteria and are not the original H1/H2/H3 sentences above. V2 further amends H1 to a percentage of eligible datasets, adds a 5% validation FPR budget, and uses label-filtered normal-only training. Keep all these decisions explicit in the methodology.
