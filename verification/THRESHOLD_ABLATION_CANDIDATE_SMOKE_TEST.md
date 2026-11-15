# Threshold Ablation Candidate — Software Smoke Test

**Date:** 2026-11-15
**Scope:** One dataset, one window size (30), one seed (15), one training epoch, CPU. This is a software smoke test only, not research evidence.

## Verification

- `python run_tests.py`: 22 tests passed.
- `run_experiment.py` completed successfully.
- Six threshold methods produced six rows in `test_metrics.csv`.
- Six validation-tuning rows were written.
- Six dataset-macro summary rows were written.
- One compressed calibration-trace file was saved, containing train errors, validation errors, validation labels, and test errors.
- No full-benchmark F1/NAB conclusions are drawn from this single short run.

The full GPU run remains pending. Follow `reports/REPRODUCIBILITY_AND_RUN_STEPS.md`.
