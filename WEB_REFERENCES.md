# External technical references checked for this package

1. Numenta Anomaly Benchmark (official GitHub repository): https://github.com/numenta/NAB
   - Describes NAB v1.1 as a streaming anomaly-detection benchmark and the corpus as 58 time-series files.
   - Its README describes the official scoreboard normalization as 100 for a perfect detector and 0 for the null detector. The project's score is explicitly labelled *NAB-style test-partition score*, not an official full-corpus NAB leaderboard score.

2. TensorFlow tested source-build compatibility matrix: https://www.tensorflow.org/install/source
   - Lists TensorFlow 2.21.0 with Python 3.10–3.13 and TensorFlow 2.15.0 with Python 3.9–3.11. This is why the included main requirements match the recorded Python 3.12.3 + TensorFlow 2.21.0 WSL environment, while TensorFlow 2.15.1 is an optional older reference only.
