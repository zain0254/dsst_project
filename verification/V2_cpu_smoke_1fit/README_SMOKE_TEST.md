# V2 CPU Smoke Test — Verified Output, Not a Benchmark

This folder contains the actual output from a single V2 pipeline smoke run executed while assembling this package.

- Dataset: `realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv`
- Window length: 30
- Seed: 15
- Epochs: 1
- Device: CPU
- Models trained: 1
- Threshold methods: 4
- Test rows: 807

The run successfully completed and wrote test metrics, predictions, validation tuning, split audit, environment/config snapshots, pairwise/statistical output, and plots (plots and core prediction trace are included here). It verifies basic execution after the final code changes.

**Do not use these four method rows as thesis evidence.** A one-epoch, one-series, one-window CPU smoke test is not a statistically meaningful benchmark and does not test full GPU runtime. The full V2 expanded dataset-selection and GPU benchmark remain pending.

The hypothesis output is generated from a single eligible file in this smoke test; all inferential p-values are therefore unavailable. This is expected and must not be interpreted as a result.
