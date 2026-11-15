# Final46 Audit and Verification

## Primary audit evidence

All paths below are relative to `source_code/outputs/final46_six_methods/`:

- `run_summary.json`: 46 selected datasets, 414 model fits, 2,484 method-level metric rows, six methods, metric population counts.
- `config_used.json`: saved configuration, selected dataset/domain mapping, model hyperparameters, threshold grids, chronological split.
- `environment.json`: recorded runtime environment.
- `dataset_split_audit.csv`: per-dataset split and label eligibility.
- `validation_tuning.csv`: selected threshold parameters and validation behavior.
- `calibration_traces/`: saved score/error traces supporting threshold re-analysis.
- `predictions/`: held-out predictions and thresholds.
- `statistical_tests.csv` and `hypothesis_outcomes.csv`: inferential outcomes and interpretation.

## Verification facts

- Run summary records 46 selected datasets, 414 trained models, and 2,484 method-level rows.
- 46 × 3 windows × 3 seeds = 414 fits.
- 414 × 6 methods = 2,484 rows.
- The selected domains are AWS CloudWatch (16), Twitter (10), Traffic (7), Known Cause (7), and Ad Exchange (6).
- Metric populations are 18 positive-test datasets, 17 complete-event datasets, and 42 high-variance FPR-evaluable datasets.
- The saved run summary records requested device `cuda`; this field alone is not proof that each fit executed on a GPU. Use per-run logs/environment evidence for that claim.

## Issues to reconcile before paper submission

1. `research_question_results.md` reports H1 Holm p=0.0005722 in its RQ2 table, while `statistical_tests.csv` records H1 Holm p=0.0003814697. The report-generation outputs must be reconciled.
2. `benchmark_method_summary.csv` provides dataset-level metric means and valid dataset counts. Preserve those denominators when quoting values.
3. The Final46 configuration has latent size 32, but no latent-size ablation establishes optimality.
4. Confirm whether model weight checkpoints exist before claiming all trained weights are archived.
5. Confirm current test-suite output on the exact reviewed commit before saying tests pass.
6. Verify the final reproduction command against the CLI parser before using it; a saved config is an experiment record, not automatically a portable run command.

## Software correctness versus scientific validity

Passing unit tests checks specified software behavior; it does not prove that the research design is unbiased, that a metric is scientifically appropriate, or that a hypothesis is supported. The paper still needs a critical methods and limitations discussion.
