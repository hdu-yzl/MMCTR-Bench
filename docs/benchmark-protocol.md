# Run and evaluate experiments

## Model selection and final evaluation

Fit parameters on training data. Use validation AUC for early stopping, checkpoint selection,
and hyperparameter selection. Freeze the configuration and weights before computing test AUC
and LogLoss. The candidate-search CLI evaluates a predeclared
list. To implement the paper's Optuna random search (seed 2025, up to 50 trials or 50 GPU-hours)
with an external driver, install `.[optuna]`.

## Datasets and reference results

Use the dataset versions and split definitions in the [data guide](data.md). Each of the four
[README reference tables](../README.md#reference-results) covers 21 models and seeds 2025–2029.
AUC and LogLoss are reported as mean ± sample standard deviation (`ddof=1`). Keep per-seed
scores, resolved settings, and the data fingerprint with the run records.

## Tune one model

Set `MMCTR_PYTHON` to the active environment's absolute interpreter and run from the repository
root:

```bash
"$MMCTR_PYTHON" -m mmctr.cli tune --dataset-name tiktok --model-name dmf \
  --candidates configs/experiments/tuning.example.yaml --study-id tiktok-dmf-search \
  --output-root outputs/tuning --cuda 0
```

Candidate YAML accepts `model` and `training` overrides. Ties keep the first candidate;
the selected weights are tested without retraining. Use a fresh study ID for each search.
