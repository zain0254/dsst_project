# Final46 Reproducibility Guide

## Inspect existing results without retraining

Start in `source_code/outputs/final46_six_methods/` and inspect:

1. `run_summary.json`
2. `config_used.json`
3. `environment.json`
4. `test_metrics.csv`
5. `benchmark_method_summary.csv`
6. `statistical_tests.csv`
7. `hypothesis_outcomes.csv`

This inspection does not require running model training.

## Run software tests

From the repository's `source_code/` directory, activate the project environment and run:

~~~bash
source ~/venvs/lstm_tf221_gpu/bin/activate
python run_tests.py
~~~

Record the actual output from the checkout being reviewed. Do not rely only on an older test-count statement.

## Before attempting a full reproduction

A full reproduction retrains 414 LSTM Autoencoders. It is not needed to read or understand the saved results.

First inspect the command-line parser in `run_experiment.py` and the saved configuration. Confirm the supported config and output arguments before running. Use a new output directory, for example `outputs/final46_reproduction_YYYYMMDD/`; never write into or overwrite `outputs/final46_six_methods/`.

Do not assume the Final46 output directory contains model weight checkpoints. The saved predictions and calibration traces support analysis of threshold decisions, but retraining the neural networks requires the source data, code, configuration, environment, and random seeds.

## Environment and hardware claims

Use `environment.json` and actual training logs to report software/hardware versions. A configuration field requesting CUDA is not, by itself, proof of GPU execution for every model fit.
