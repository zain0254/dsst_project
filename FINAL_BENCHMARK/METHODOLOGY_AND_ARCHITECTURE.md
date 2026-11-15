# Final46 — Methodology and Architecture

## End-to-end pipeline

1. Load a selected NAB univariate time series and its anomaly-window labels.
2. Parse timestamps, order observations chronologically, and construct point labels from NAB anomaly windows.
3. Partition each series chronologically into 60% train, 20% validation, and 20% test.
4. Fit scaling using normal training data only.
5. Create sequence windows of length 30, 50, or 100.
6. Retain training windows only when all points in the full window are normal. This uses labels and is therefore label-informed normal-only training, not fully label-free unsupervised training.
7. Train an LSTM Autoencoder with hidden size 64 and latent size 32. The configured maximum is 40 epochs, batch size 256, learning rate 0.001, weight decay 0.00001, and early-stopping patience 6.
8. Obtain reconstruction errors and use them as anomaly scores.
9. Tune threshold parameters on validation data under a shared 5% validation FPR budget.
10. Apply each of six threshold methods to the held-out test score sequence.
11. Calculate point-level, event-level, and high-variance metrics on their eligible populations.
12. Aggregate seeds and window sizes within each dataset for dataset-level comparisons and statistical testing.
13. Write raw metric rows, predictions, calibration traces, audit tables, summaries, and figures to the Final46 output directory.

The executable implementation is in `source_code/run_experiment.py` and the modules under `source_code/src/`. Consult the source when describing exact function signatures or implementation details.

## LSTM Autoencoder

A window is a sequence of consecutive observations. For a univariate series, a length-100 window has shape 100 × 1. The encoder LSTM summarizes temporal information in a 64-unit hidden representation. A Dense layer maps this to a 32-dimensional latent vector using tanh. Decoder initial hidden and cell states are generated from the latent vector; the latent vector is repeated across the window and passed through the decoder LSTM, followed by a reconstruction layer.

The latent dimension is a bottleneck hyperparameter. It encourages a compact representation, but 32 was configured rather than selected by a dedicated latent-size ablation. Do not claim it is optimal.

## Six threshold methods

| Identifier | Concept |
|---|---|
| `fixed_percentile` | A fixed threshold set from a percentile of training reconstruction errors |
| `fixed_mean_k_sigma` | Training mean plus a tuned multiple of training standard deviation |
| `rolling_mean_3sigma` | Causal rolling mean plus a validation-calibrated k times rolling standard deviation; the identifier is legacy and does not mean a hard-coded 3σ |
| `moving_median_lambda_mad` | Moving median plus lambda times raw median absolute deviation; original proposed method |
| `moving_median_scaled_mad` | Moving median plus k times 1.4826 × MAD |
| `moving_median_iqr_scale` | Moving median plus k times IQR / 1.3489795 |

For a history H, raw MAD is median(|e − median(H)|). Scaled MAD multiplies raw MAD by 1.4826 as a normal-consistency scale estimate. IQR scale divides the interquartile range by approximately 1.349. These are alternative robust scale estimators, not identical formulas.

## Calibration and causal application

Candidate parameters are selected using validation data and a common 0.05 FPR budget. When validation contains no positive labels, F1 cannot meaningfully guide selection; the recorded fallback selects a candidate with FPR closest to the prespecified budget from below, or the lowest-FPR candidate if none meets the budget.

A 5% validation target is not a guarantee of 5% test FPR. Test false-alarm rates are measured independently on held-out test data.

The adaptive threshold uses preceding reconstruction-error history. The current point is excluded from the history used to determine its own threshold, preventing the current score from directly raising its own threshold.

## Statistical unit

The model has three seeds and three window sizes per dataset. These are repeated experimental conditions, not 9 independent datasets. The inferential comparisons use dataset-level observations after aggregating the repeated conditions within each dataset.
