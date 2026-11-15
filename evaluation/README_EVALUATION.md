> **Historical V1 notice:** This folder contains the original 10-dataset evaluation matrix and workbook. It is not the Final46 evaluation package. The authoritative completed benchmark is `source_code/outputs/final46_six_methods/`; see [Final Benchmark evaluation guide](../FINAL_BENCHMARK/EVALUATION/README.md).

# Evaluation matrix guide

All values in `evaluation_matrix_summary.csv` are equal-weighted means of per-dataset values. The `n` in each column header is the number of datasets with a valid value for that metric, not the number of model fits.

- F1 across all 10 files includes six test files with no positive labels, where pointwise F1 is zero by convention. Read this alongside the positive-test subset F1 (n=4).
- Precision, recall, ROC-AUC, and PR-AUC/AP are summarized only for the four files with positive test labels and both classes present.
- Accuracy and FPR are summarized across all 10 files. Accuracy is secondary because class imbalance can make it misleading.
- High-variance FPR is available for eight files with normal points in the high-variance test mask.
- Event recall, normalized event delay, and NAB-style score use four event-evaluable files containing seven complete test events.
- ROC-AUC and PR-AUC/AP are threshold-independent because all methods receive the same reconstruction-error scores; their values should be identical across threshold methods.
- `evaluation_matrix_per_dataset.csv` is the wide matrix with all per-file comparisons. `dataset_method_means.csv` is the long-form equivalent.
- `test_metrics_augmented.csv` adds accuracy, ROC-AUC and timestamp-based delay in seconds to each raw method-level row. The source `test_metrics.csv` under `verified_results/` is unchanged.

Detection delay in seconds is retained for transparency, but it is not averaged into the headline matrix because NAB series can have different sampling intervals and event durations. The normalized delay fraction is used for a more interpretable cross-file summary.
