# Version 2 changes

- Added `config_v2.json`; Version 1 config and outputs are preserved.
- Added `prepare_expanded_nab_manifest.py` to download and pre-screen up to 15 additional real NAB files for complete events in the held-out test partition and adequate clean train/validation windows.
- Added normal-only training: scaler fit on normal training rows; training windows retained only if every point in the window is normal; early stopping uses fully normal validation windows. Full validation labels remain available only for threshold calibration.
- Added a common 5% validation FPR budget and consistent candidate-selection rule for all threshold methods.
- Changed rolling mean baseline from fixed 3-sigma to validation-calibrated k-sigma, with the same k candidate grid as fixed mean+sigma.
- Expanded percentile, k and lambda grids; added a scale-relative MAD floor.
- Increased epoch cap to 40 and early-stopping patience to 6.
- Corrected the statistical reporting to follow the original proposal hypotheses: H1 compares F1 with the two fixed-threshold baselines; H2 reports per-series high-variance FPR outcomes and aggregate tests; H3 evaluates NAB-style event score with false-alert and timing context.
- Kept domain-ranking analysis separate as exploratory RQ4 rather than substituting it for H3. Holm correction is applied within each hypothesis comparison family.
- Added tests for normal-only windows, FPR-constrained calibration, H1 baseline-specific significance logic, and clean-normal high-variance calibration.
- Added a post-hoc V2 metric reanalysis script that recomputes high-variance FPR from fully normal training histories without retraining models.
- Corrected the H1 decision logic so positive direction and adjusted significance must refer to the same baseline comparison.
- Passed the configured validation FPR budget explicitly into threshold tuning.

Validation performed during the review build:
- Python compilation passed.
- All 20 unit tests passed.
- One-file, one-window, one-seed CPU smoke run completed.
- All 828 saved V2 metric rows were independently reproduced from 207 prediction files.
- The clean-normal volatility reanalysis recalculated 828 rows without retraining; all other metrics reproduced within 4.6e-13.

The archived full V2 GPU benchmark contains 207 model fits, 828 method-level rows, and 207 prediction files. The 13 additional selected datasets were already present in the archived package; 6 of the 10 legacy datasets have no positive test labels.
