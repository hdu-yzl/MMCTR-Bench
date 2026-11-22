# Run and evaluate experiments

## Model selection and final evaluation

Fit parameters on training data. Use validation AUC for early stopping, checkpoint selection,
and hyperparameter selection. Freeze the configuration and weights before computing test AUC
and LogLoss. Install `.[optuna]` for the built-in random search. Each model gets at most
**50 attempted trials or 50 allocated GPU-hours**, whichever comes first. Failed and
budget-interrupted trials count toward the trial limit.

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
  --study-id tiktok-dmf-search \
  --output-root outputs/tuning --cuda 0
```

After the frozen validation winner completes its final evaluation successfully, `tune`
automatically writes its settings to `configs/tuned/<dataset>/<model>.yaml`. This is a
dataset/model-specific configuration layer over the base catalogs. It includes the
resolved model parameters, training hyperparameters, data fingerprint, and validation
selection provenance; final-test metrics do not affect the selection or write-back.
Subsequent ordinary training and main-table jobs automatically load that file. Their
run seed, device, and output directory remain controlled by the new run's arguments.
The main-table source hashes also include these YAML files.

The most recent successful search replaces the same pair's saved configuration;
failed searches do not change it. New searches use the base catalogs and their search
overrides, independently of earlier saved configurations. Explicit Trainer overrides
take priority over saved defaults. A different data fingerprint is rejected to prevent
accidental reuse across data versions.

Pass `tune --no-write-back` to keep an exploratory search from updating defaults, or
`train --ignore-tuned-config` to train from the base catalogs. Both commands accept
`--tuned-config-dir` for an alternate saved-defaults directory. The main-table launcher
uses the default `configs/tuned/` directory. See the [configuration guide](../configs/tuned/README.md).

All models use this fixed public search space:

| Hyperparameter | Values |
|---|---|
| Learning rate (`training.lr`) | 0.001, 0.0001, 0.00001 |
| Weight decay (`training.l2`) | 0.00001, 0.000001, 0.0000001 |
| Dropout (`model.dropout`) | 0, 0.1, 0.2, 0.3, 0.5 |

Optuna's `RandomSampler` draws 50 categorical trials with replacement using search seed 2025,
independently of the training `--seed`. Repeated combinations are allowed; the space has
45 distinct shared combinations. The original tuner configuration adds the following
model-specific dimensions automatically; all other settings retain their configured defaults.
The combined space shares one budget of 50 attempts / 50 GPU-hours per model, rather than
allocating a separate budget to each dimension. Adam, batch size 512, maximum 20 epochs,
and patience 5 remain fixed.

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
the original catalog dropout as `quantizer_dropout` for the frozen artifact's compatibility
check. An explicit `quantizer_dropout` override must match the pretraining artifact.

Optionally pass `--candidates configs/experiments/tuning.example.yaml`: a YAML containing
`model: {...}` fixes model variant controls while retaining the public search. It cannot
override any searched parameter, including through a dataset-scoped mapping. Legacy YAML
with a nonempty `candidates` list of `model` and
`training` overrides remains available for exploratory searches and small smoke runs; it is
labelled `explicit_candidates` in artifacts and has the same hard budget caps.

GPU-hours sum each trial's wall time on its single allocated CUDA device, including
initialization, training, validation, and cleanup. The deadline is checked between batches
and after fitting; in-flight work may finish after the limit. A trial interrupted by the
budget cannot win. If no trial completed, history is retained and the command fails without
testing. CPU time (`--cuda -1`) is recorded separately and consumes no GPU-hours. The final
test is outside the search budget. Search settings, per-trial timings, cumulative usage,
and the stopping reason are saved with the history and frozen selection.

Ties keep the first candidate; the selected weights are tested without retraining.
Use a fresh study ID for each search, including after failed or interrupted searches.
