# V1 Full GPU Benchmark — Archived Evidence

This directory contains the completed Version 1 benchmark outputs from the recorded WSL2 / TensorFlow 2.21.0 / NVIDIA RTX 4050 Laptop GPU run.

- 10 NAB time series
- 3 window lengths: 30, 50, 100
- 3 model seeds: 15, 42, 123
- 90 LSTM-AE fits
- 360 method-level test rows across four threshold rules
- 90 timestamp-aligned prediction CSVs in `predictions/`

The original CSV/JSON/Markdown/plot outputs are preserved as V1 historical evidence. The package-level `evaluation/` directory holds recomputed and augmented metrics, including ROC-AUC and accuracy. These derived metrics were validated against prediction traces; the original `test_metrics.csv` was not modified.

Interpret the proposed Median + MAD results cautiously: higher sensitivity and earlier event detections were accompanied by much higher false-positive rates and a markedly worse NAB-style score. Only four files had positive labels in the held-out test partition and only seven complete events were evaluable.

`nab_style_score` is a test-partition NAB-style score over complete events, not an official full-corpus NAB leaderboard score. See the package-level final audit report.
