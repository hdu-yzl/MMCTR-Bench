# Run and evaluate experiments

## Model selection and final evaluation

Fit parameters on training data. Use validation AUC for early stopping, checkpoint selection,
and hyperparameter selection. Freeze the configuration and weights before computing test AUC
and LogLoss. Install `.[optuna]` for the built-in random search. By default, each dataset/model pair gets at most
**50 attempted trials or 50 allocated GPU-hours**, whichever comes first. Set
`--max-trials` and `--max-gpu-hours` to change these limits; `--max-hours` additionally
limits trial time on either CPU or GPU. Failed and
budget-interrupted trials count toward the trial limit.

## Datasets and reference results

Use the dataset versions and split definitions in the [data guide](data.md). Each of the four
[README reference tables](../README.md#reference-results) covers 21 models and seeds 2025–2029.
AUC and LogLoss are reported as mean ± sample standard deviation (`ddof=1`). Keep per-seed
scores, resolved settings, and the data fingerprint with the run records.

## Batch training

Run `scripts/train_all_models.sh --dataset tiktok --gpus 0,1` for all 23 models, or
select a subset with `--models dmf,qarm,psrq`. Each model runs once using its saved
configuration. Reference tables summarize earlier five-seed runs; individual results
can vary with the seed and environment.

Before CTR workers start, `mmctr.quantization.ensure` checks QARM's per-modality RQ
artifacts and PSRQ's joint artifact with the production loader. Missing, corrupt or
structurally incompatible artifacts trigger default pretraining and a second check.
Valid artifacts are reused, and preparation is locked per dataset to prevent duplicate
pretraining by simultaneous batch launchers. Failure stops the batch before CTR starts.
RQ uses `configs/models/<dataset>/rq.yaml`; PSRQ uses its model configuration and
`pretraining` settings. SID codes are computed from these frozen quantizers during model
execution; there is no separate SID completion flag. `--dry-run` checks existing artifacts
and prints any required pretraining commands without fitting them.

Standalone training/tuning still require prepared artifacts. To prepare them separately:

```bash
"$MMCTR_PYTHON" -m mmctr.quantization.ensure --dataset-name tiktok --cuda 0
```

## Tune one model

Set `MMCTR_PYTHON` to the active environment's absolute interpreter and run from the repository
root:

```bash
"$MMCTR_PYTHON" -m mmctr.cli tune --dataset-name tiktok --model-name dmf \
  --study-id tiktok-dmf-search \
  --output-root outputs/tuning --cuda 0
```

After the frozen validation winner completes its final evaluation successfully, `tune`
atomically replaces `configs/models/<dataset>/<model>.yaml` with its model and training
parameters plus validation provenance. Single-model and batch training read this same file.
Run seeds, devices and output paths remain controlled by the new run's arguments.
Test metrics never affect selection or enter the parameter file. Failed searches leave
existing parameters unchanged.

A new search starts with the current model file and retains unsampled settings. The
requested search overrides sampled parameters; it may establish new selection provenance
for a changed dataset. Ordinary training checks any saved selection's data fingerprint.
Explicit Trainer overrides take precedence over the file.

Pass `--no-write-back` for exploratory searches. Both `train` and `tune` accept
`--model-config-dir` to choose another parameter root. There is no separate tuned layer.
To select several methods or datasets, use:

```bash
scripts/tune.sh --datasets tiktok,ebnerd --models dmf,simcen \
  --max-trials 20 --max-gpu-hours 4 --cuda 0
```

Each pair receives the specified budget, and pairs run sequentially. Use `--dry-run`
to inspect commands. See the [parameter guide](../configs/models/README.md).

All models use this fixed public search space:

| Hyperparameter | Values |
|---|---|
| Learning rate (`training.lr`) | 0.001, 0.0001, 0.00001 |
| Weight decay (`training.l2`) | 0.00001, 0.000001, 0.0000001 |
| Dropout (`model.dropout`) | 0, 0.1, 0.2, 0.3, 0.5 |

Optuna's `RandomSampler` draws the requested number of categorical trials with replacement using search seed 2025,
independently of the training `--seed`. Repeated combinations are allowed; the space has
45 distinct shared combinations. The original tuner configuration adds the following
model-specific dimensions automatically; all other settings retain their configured defaults.
The combined space shares one budget per dataset/model pair. Unsampled optimizer and
training settings remain fixed at the values in that pair's parameter file (initially Adam,
batch size 512, maximum 20 epochs, and patience 5).

| Model | Additional parameters and values |
|---|---|
| DCN | `cross_num`: [3, 4, 5] |
| LMF | `rank`: [3, 5, 7, 10]; `fusion_dim`: [32, 64, 128] |
| Diff-MSIN | `lambda1`, `lambda2`: [0.001, 0.01, 0.1, 0.5]; `T`: [3, 5, 7, 10]; `num_cross_layers`: [3, 4, 5]; `heads`: [1, 2, 4] |
| MARN | `lambda0`: [0.001, 0.01, 0.05, 0.1, 0.5] |
| DMF | `tier_num`: [10, 20, 30, 50]; `attention_dim`: [32, 64, 128]; `num_buckets`: [30, 50, 70]; `alpha`: [0.05, 0.1, 0.3, 0.5] |
| SimCEN | `hidden_unit`: [[480, 480, 480]]; `cl_temperature`, `alpha`: [0.05, 0.1, 0.3, 0.5] |
| MAKE | `tier_num`: [10, 20, 30, 50] |
| EM3 | `query_num`: [3, 5, 7, 10]; `cic_tau`, `cic_weight`: [0.05, 0.1, 0.3, 0.5] |
| GMMF | `lambdas`: all image/text/audio weights jointly 0.1 or jointly 0.3; `N`: [1, 2, 3]; `lr_disc`, `lr_gen`: [0.0001, 0.00001, 0.000001, 0.0000001] |
| QARM | `cross_num`: [3, 4, 5] |

The executable ranges live in [model_search.py](../src/mmctr/experiments/model_search.py).
Models without an explicit original search range use only the shared space. Shared draws
are paired across models using a separate random stream from model-specific draws
(model-specific seed 2026).
Structured choices are sampled by stable index and materialized as lists/mappings in
candidate configs and provenance. GMMF uses the shared L2 in all three optimizer phases;
unused modalities in its reconstruction-weight mapping are ignored.

Two original SimCEN choices, [512, 512, 512] and [128, 128, 128], are excluded because
the implemented three-way experts require each hidden width to be divisible by three.
The exclusion and reason are recorded in search metadata; widths are not silently rounded.

RQ and PSRQ original ranges belong to **quantizer pretraining**, which the CTR tuner does
not run. They are retained separately in `PRETRAINING_SEARCH_SPACES`:

| Pretrainer | Original ranges (not sampled by CTR tuning) |
|---|---|
| RQ | `n_levels`: [2, 3, 4]; `codebook_size`: [256, 512, 1024]; `batch_size`: [256, 512, 1024]; `n_init`: [3, 5, 10]; `max_iter`: [10, 20, 30] |
| PSRQ | `n_levels`: [2, 3, 4, 5]; `codebook_size`: [256, 512, 1024, 2048]; `quant_loss_weight`: [0.5, 1.0, 2.0]; `mu`: [0.25, 0.5, 0.99]; `batch_size`: [512, 1024, 2048] |

Pretraining must satisfy `batch_size >= codebook_size` and produce compatible frozen
artifacts before CTR training. PSRQ CTR search varies downstream dropout while retaining
the configured `quantizer_dropout` for the frozen artifact's compatibility
check. An explicit `quantizer_dropout` override must match the pretraining artifact.

Optionally pass `--candidates configs/experiments/tuning.example.yaml`: a YAML containing
`model: {...}` fixes model variant controls while retaining the public search. It cannot
override any searched parameter, including through a dataset-scoped mapping. Legacy YAML
with a nonempty `candidates` list of `model` and
`training` overrides remains available for exploratory searches and small smoke runs; it is
labelled `explicit_candidates` in artifacts and uses the same configurable budget limits.

GPU-hours sum each trial's wall time on its single allocated CUDA device, including
initialization, training, validation, and cleanup. The deadline is checked between batches
and after fitting; in-flight work may finish after the limit. A trial interrupted by the
budget cannot win. If no trial completed, history is retained and the command fails without
testing or writing parameters. CPU time (`--cuda -1`) is recorded separately and consumes
no GPU-hours; use `--max-hours` for a CPU time limit. The final
test is outside the search budget. Search settings, per-trial timings, cumulative usage,
and the stopping reason are saved with the history and frozen selection.

Ties keep the first candidate; the selected weights are tested without retraining.
Use a fresh study ID for each search, including after failed or interrupted searches.

## Entry points and extensions

Use `scripts/train_model.sh` for one model, `scripts/train_all_models.sh` for a model set,
and `scripts/tune.sh` for budgeted parameter search. The batch script includes quantized
models by default; the obsolete `--include-quantized` option is removed. Use
`scripts/pretrain_quantizers.sh` to deliberately refit RQ and PSRQ; ordinary batch training
reuses compatible artifacts and prepares missing ones automatically.

The unused `ExperimentRunner`, `ExperimentTask`, matrix serialization functions and
`ValidationOnlyTuner.final_test_task()` are removed. Use the current training/tuning entry
points. Custom tuning code can use `load_tuning_search()` and `run_validation_search()`;
selection freezing and separate final-test result recording remain available.
