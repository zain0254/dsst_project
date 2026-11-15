# Historical Experiments — V1 and V2

This folder is a navigation guide. Historical source and result folders remain in their original locations to avoid breaking paths and provenance. They are not the authoritative Final46 benchmark.

| Version | Datasets | Model fits | Threshold methods | Method-level rows | Status |
|---|---:|---:|---:|---:|---|
| V1 | 10 | 90 | 4 | 360 | Original historical experiment |
| V2 | 23 | 207 | 4 | 828 | Historical normal-only benchmark |
| Final46 | 46 | 414 | 6 | 2,484 | Authoritative completed benchmark |

## Where historical evidence lives

- V1 restart evidence: `verified_results/V1_full_gpu_benchmark_restart/`
- V1 reference implementation: `version1_reference/`
- Historical V2 outputs: `source_code/outputs/version2_normal_only_fpr05/` and its dated correction/re-audit directories.
- Historical audit and protocol documents: `reports/`.

## How to use it

Use V1/V2 to understand the development path, protocol changes, debugging, and corrections. Do not pool their datasets, model fits, metric rows, or summary statistics with Final46. The experiments have different cohorts, method sets, and protocol generations.

The canonical current result directory is `source_code/outputs/final46_six_methods/`.
