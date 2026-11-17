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

## Diagnostic experiments

Each invocation runs one dataset, model, seed, and setting. Cold start and the native control
are available on all four datasets. Robustness, alignment, and fusion runs use MicroLens or
TikTok.

### Tuning and comparisons

Declare the candidate hyperparameters before running each setting and independently tune its
matched `control`. Use the same dataset fingerprint, backbone, and paired seeds, with comparable
search spaces and budgets. Select by validation AUC (ties keep the first candidate), freeze
the best weights, and compare final test results. Do not reuse the main table's settings or
scores. This CLI searches the supplied candidates; it does not schedule an Optuna budget.

For each seed, record test AUC, LogLoss, the signed AUC difference (`setting - control`), and the
selected configurations. Use mean and sample standard deviation for paired differences across
seeds. Relative AUC change is undefined if the control AUC is zero.

### Modality robustness

The executor jointly masks all non-ID target and history modalities per sampled row in
train and test. Validation stays clean. Use `--probability 0.1|0.3|0.5|0.7` with one value
per setting. The mask seed is 42; keep the batch size fixed across rates to reuse the same
test draws.

### Alignment and fusion

Use `alignment --method` with `kl`, `contrastive` (InfoNCE), `cosine`, `mmd`, or
`adversarial`, and `--weight` with 0.1, 0.3, 0.5, or 0.7. Keep Adam fixed. Alignment
uses projected target features, or masked, pooled codebook histories for QARM and PSRQ.

Use `fusion --method` with CAT, MAF, LMF, MTFN, SRC, FQ-Former, SimCEN, DTA, or DMF.
SRC needs at least two inputs and is unavailable for MicroLens PAMD. LMF, MTFN, GMMF, and
SimCEN fusion studies use a shared wrapper, so differences from a native control can also
reflect the wrapper. To isolate an operator, tune a reference within the same wrapper and
substitution site.

For a **DMF backbone**, set `model: {similarity_source: legacy_projected}` in every alignment
or fusion candidate and in its matching control. These diagnostics use projected features;
the default frozen-feature configuration is used for ordinary training.

### Run one setting

From the repository root, use the active environment's absolute interpreter:

```bash
MMCTR_PYTHON="$(command -v python)"

"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution robustness \
  --dataset-name tiktok --model-name dmf --probability 0.3 --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-drop03-s2025 --output-root outputs/analysis --cuda 0

"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution control \
  --dataset-name tiktok --model-name dmf --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-control-s2025 --output-root outputs/analysis --cuda 0
```

Use a unique study ID for each setting and seed. Each study writes `trial_history.json`,
`frozen_selection.json`, and `final_test_result.json` under `<output-root>/<study-id>/`.
Use these JSON files for reporting and visualization with external tools.
Cold-start results include full-test metrics and the subset audit. Pass `--dataset-config`
for a local catalog and see each subcommand's `--help` for its arguments.

### Cold-start evaluation

The mixed user/item protocol (`paper-mixed-graph-oov-v2`) trains on the full training split
and selects hyperparameters and checkpoints on full validation.

- **Few-shot:** include rows whose user or item appears in fewer than five training candidate
  rows, including zero. Count both positive and negative candidates; history appearances do
  not add to these counts.
- **Zero-shot:** include rows whose user or item is absent from the training graph. For items,
  the graph includes candidate rows and valid history positions.

After selection, map unseen user and item IDs to separate zero-initialized OOV embeddings.
Use the same mapping and frozen weights for the full test and the subset. Retain non-ID
features and padding. Report the subset counts and AUC change; empty or single-class subsets
are `not_evaluable`. The difference also reflects the subset's composition.

| Dataset | Zero-shot rows | Few-shot rows |
|---|---:|---:|
| AntM2C | 386,775 | 410,705 |
| MicroLens | 2,862 | 360,000 (full test) |
| TikTok | 30 | 30 |

TikTok's two subsets are identical, with five positive and 25 negative rows.

```bash
"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution cold-start \
  --dataset-name tiktok --model-name dmf --regime zero --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-cold-zero-s2025 --output-root outputs/analysis --cuda 0
```

Use `--regime low_lt5` for few-shot.
