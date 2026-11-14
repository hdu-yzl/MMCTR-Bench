# Experiment matrices

`mmctr.experiments.ExperimentRunner` is the orchestration boundary above the canonical training
engine. A matrix contains immutable `ExperimentTask` values; every task declares its dataset,
model, seed, fully resolved config, and `data_fingerprint`. Task identity is the SHA-256 of those
values, so changing a seed, dataset version, or resolved option creates a different task rather
than silently resuming an incompatible run.

The runner receives an application executor with the signature:

```python
executor(task, run_context, device) -> {metric_name: finite_float}
```

Devices are explicit strings such as `cpu`, `cuda:0`, and `cuda:1`. At most one active worker owns
each configured device. Every task receives a normal isolated run directory containing resolved
config, runtime metadata, metrics, checkpoints, summary, and a versioned `result.json`. A task
exception is converted to a failed `RunResult` and does not cancel sibling tasks.

Matrix state lives under `outputs/.matrices/{experiment}.json` and is atomically replaced. Resume
only reuses completed results whose task identity and result metadata still agree. Failed tasks are
retried in a new run directory, preserving the failed run as evidence. The returned results remain
in caller task order even when workers complete out of order.

This runner schedules and records experiments; it does not select hyperparameters or decide when
test evaluation is allowed.

## Validation-only tuning

The candidate-search command uses the production Trainer with test disabled for
every trial, freezes the highest validation-AUC candidate, then loads its validation-best weights
for exactly one final test evaluation (no retraining). It writes no `.pt` files:

```bash
<BM_PYTHON> -m mmctr.cli tune --dataset-name tiktok --model-name dmf \
  --candidates configs/experiments/tuning.example.yaml --study-id tiktok-dmf-v2 \
  --output-root outputs/tuning --cuda 0
```

Candidate YAML accepts `model` and `training` override mappings only. Every candidate's full
resolved model/training/data configuration is frozen with its validation metrics. Study IDs
cannot be reused after selection; existing artifacts are preserved. `train --validation-only`
also makes standalone validation-only runs available. Default `train` is a final single run.

The candidate-list interface evaluates the supplied configurations and records each trial.
Paper §3.5 and Appendix C specify Optuna random search with seed `2025` and a stopping budget of
**50 trials or 50 GPU-hours, whichever occurs first**. That search strategy and time limit require
an external search driver; the candidate-list CLI does not schedule them.

`ValidationOnlyTuner` consumes complete `TuningTrial` records and rejects any trial that already
contains a `test*` metric. Selection uses strictly higher `val_auc`; `val_log_loss` is recorded but
does not change the tie rule: equal AUC keeps the first candidate. The study writes complete
`trial_history.json` plus an atomic `frozen_selection.json` containing experiment ID, selected run,
validation metrics, seeds, data fingerprint, and the exact frozen candidate config. These files
remain under the configured output root and are never promoted directly into tracked YAML.

Only a persisted `FrozenSelection` can create a `stage: final_test` task. Its final `test_*` metrics
are written to a separate `final_test_result.json`; they cannot alter or overwrite the selection
artifact. This keeps configuration search and final reporting mechanically separated.

## Modal robustness

The paper protocol independently masks textual and visual non-ID modalities at the instance level with
rates `{0.1, 0.3, 0.5, 0.7}`; ID features remain present. The same mechanism is applied to
train/validation/test, and a separate model is retrained and validation-selected for each rate. Reported
AUC drop is relative to the original complete-modality model, so the complete setting is the baseline,
not an additional masking rate. The public example enumerates exactly these four rates.

`ModalityDropout` is the single canonical missing-modality transform. It accepts named non-ID
modalities and a probability, then samples one missing flag per example/modality and applies the
same flag to target and complete history tokens. Inputs are never modified. The derived random seed
includes the experiment seed, split, batch index, and modality, so repeating evaluation produces
the same perturbation while different batches do not reuse one mask. Metadata records protocol,
probability, seed, and realized missing counts.

`TransformedDataLoader` applies this transform only to explicit train/validation/test splits and
otherwise delegates the production canonical loader. Robustness experiments therefore use the same
model and data path as formal training instead of maintaining a robustness model copy.

Define the task grid with `configs/experiments/robustness.example.yaml`, then validate and persist
it with:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli plan-robustness-study \
  --config configs/experiments/robustness.example.yaml \
  --output outputs/analysis/robustness/tasks.json
```

The strict config fixes the production model/data config, canonical manifest fingerprint,
droppable modalities, affected splits, probabilities, and seeds. The command writes an atomic
`robustness-study-matrix-v1`; `load_robustness_study_matrix` verifies its content fingerprint and
returns tasks for `ExperimentRunner`. The diagnostic executor also offers a
[joint-modality protocol](analysis/cross-dataset-diagnostics.md#modality-robustness) with explicit
mask coupling and affected splits.

## Cold-start protocol

**Paper reproduction uses `paper-mixed-graph-oov-v2`**, implemented by
`cold_populations`, `PaperColdStartLoader` and the diagnostic executor. Follow the runnable
workflow in [benchmark protocol](benchmark-protocol.md#cold-start-evaluation).
It uses the user/item **union**, few-shot candidate support **<5 including zero**, zero-shot
absence from both train targets and train histories, full train/full validation, and shared
OOV identity indices at final evaluation only. The comparator is the full test AUC of the same
selected weights with the same OOV policy. The executor requires matching v2 matrix and
baseline fingerprints.

The event-partition API below is an entity-support audit/planner, not the paper mixed cold-start
execution path. Its v2 few-shot rule is also strict `<N` including zero; pass
`max_support_interactions=5` as an exclusive threshold.

`ColdStartProtocol` validates user- or item-targeted `cold_start`, `zero_shot`, and `few_shot`
partitions from canonical event, user, and item IDs. Train and evaluation event sets must be
disjoint. Zero-shot targets must be absent from training; for item audits supply all training
history IDs through `train_history_item_ids`. Every few-shot evaluation target must
have fewer than the declared threshold of training interactions, including zero. These checks are made
before evaluation, so an accidental warm target or duplicated event fails the experiment instead
of being folded into a metric.

Each audit records split sizes, sorted per-target support counts, and a deterministic fingerprint
of the complete source partitions. `save_cold_start_audit` atomically writes the versioned audit
manifest with a second integrity fingerprint; `load_cold_start_audit` rejects modified or malformed
artifacts. Formal runs should retain that manifest beside their resolved config and results.

After producing the audit, reference it from `configs/experiments/cold-start.example.yaml` and
build the evaluation matrix with:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli plan-cold-start-study \
  --config configs/experiments/cold-start.example.yaml \
  --output outputs/analysis/cold-start/tasks.json
```

Planning reloads and integrity-checks the audit, embeds its protocol fingerprint and file SHA-256,
and writes `cold-start-study-matrix-v2`. Cold-start evaluation uses canonical dataset versions
and the shared runner/training engine.

## Efficiency protocol

`EfficiencyProtocol` is the shared boundary for parameter count, step latency, throughput, and
peak allocated CUDA memory. A caller supplies one already-configured training or inference step,
the exact examples per step, model parameter source, explicit device, and the data/input
fingerprint. Warm-up calls are excluded from timing. CUDA measurements synchronize before and
after the measured region and reset peak-memory statistics after warm-up; CPU reports leave peak
memory unset instead of presenting a non-comparable process-memory estimate.

The resulting `EfficiencyReport` records warm-up/measured counts, total and trainable parameters,
elapsed time, per-step latency, examples/second, peak bytes, device, and input fingerprint.
The runtime fields also preserve accelerator name plus PyTorch/CUDA versions.
`save_efficiency_report` writes an atomic versioned artifact, and its loader verifies both the
manifest and report fingerprints. Formal tables must use the same step definition and batch size
across compared models and retain the report artifacts used to generate them.

## Alignment protocol

The paper evaluates five objectives at a unified cross-modal interface: Kullback–Leibler divergence
(`KL`), `InfoNCE`, cosine similarity (`Cosine`), maximum mean discrepancy (`MMD`), and adversarial
alignment (`Adv`). Only the objective and its validation-selected coefficient vary.

The generic `plan-alignment-study` API supports `cosine` and `mse`; `MSE` is an implementation
diagnostic. The [diagnostic executor](analysis/cross-dataset-diagnostics.md#alignment) supports
the five-objective matrix using explicit representation adapters and optimization phases.

Alignment studies use the production model unchanged. `ActivationCapture` installs temporary
forward hooks on explicitly named modules, retains their differentiable tensor outputs for one
forward pass, and always removes the hooks when its context exits. Alternatively, models that
already publish `ModelOutput.representations` need no hooks.

`AlignmentAuxiliary` validates at least two same-shaped, batch-aligned floating representations
and attaches a named `alignment_cosine` or `alignment_mse` scalar to the existing
`ModelOutput.auxiliary_losses`. It does not replace logits, wrap the model, or own an optimizer;
the normal training engine applies the configured auxiliary-loss weight and run provenance.

Use `configs/experiments/alignment.example.yaml` to declare production model/data config, the
canonical manifest fingerprint, alignment methods/weights, model-specific representation module
paths, and seeds. Build the verified matrix with:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli plan-alignment-study \\
  --config configs/experiments/alignment.example.yaml \\
  --output outputs/analysis/alignment/tasks.json
```

The output uses `alignment-study-matrix-v1` and the shared task-matrix integrity protocol.
Alignment execution injects `ActivationCapture` and
`AlignmentAuxiliary` from the task's `analysis` config into the shared training executor.

## Plot inputs and provenance

Plots must enter through `load_standard_results`, which accepts only completed version-1
ExperimentRunner results, finite metrics, and the explicitly required metric names. This prevents
analysis code from silently mixing failed runs, ad-hoc CSV columns, and values copied into plotting
scripts.

Every generated figure should have a sibling provenance JSON written by
`save_figure_provenance`. It records the complete plotting config, script version, ordered input
paths, sizes, SHA-256 hashes, and an aggregate fingerprint.

For standard single-metric comparisons, the canonical CLI performs ingestion, seed aggregation,
rendering, and provenance in one operation:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli plot-results \
  --inputs outputs/experiment/*/*/*/*/result.json \
  --output outputs/figures/validation-auc.png \
  --metric val_auc --kind bar --group-by model
```

`render_metric_figure` accepts only completed `result-v1` inputs, groups by model/dataset/task/seed,
uses the arithmetic mean for repeated runs, writes PNG/PDF/SVG atomically, and always emits a
sibling provenance manifest containing the plotted values and source hashes. Generated
figures belong under ignored `outputs/figures/`; runtime source under `src/` contains no static
research artifacts.

## Fusion studies

The paper independently retrains each model after replacing only its fusion module with one of nine
operators: Concat (`CAT`), `DMF`, decoupled target attention (`DTA`), `FQ-Former`, low-rank multimodal
fusion (`LMF`), multi-head attention fusion (`MAF`), multimodal tensor fusion network (`MTFN`),
`SimCEN`, and stochastic reverse cross-modal fusion (`SRC`). Standard linear projections may adapt
dimensions; the remaining architecture and formal tuning budget stay fixed.

The generic registry exposes six operators: `concatenate`/`cat`, `sum`, `mean`, `maf`, `lmf`, and
`mtfn`. `CAT`, `MAF`, `LMF`, and `MTFN` overlap the paper nine; `sum` and `mean` are additional
runtime choices. The [diagnostic executor](analysis/cross-dataset-diagnostics.md#fusion) provides
all nine operators through model-specific adapters with explicit substitution policies.

`build_fusion_study_tasks` creates immutable ExperimentRunner tasks that vary registered fusion
components on the production `dnn_mm`, `dnn_mm_seq`, `dmf`, and `make` implementations. Each task
contains the exact model/data config, canonical fusion name, seed, and data fingerprint. Registry
aliases such as `cat`/`concatenate` and `add`/`sum` are normalized in provenance while retaining the
temporary implementation spelling needed by existing production constructors.

Paper-private models are rejected from this sweep: replacing their FQ-Former, SRC, expert gates,
quantization, or GAN structure with an approximate generic fusion would change the studied model.
Those models require an explicitly registered equivalent component before entering this matrix.

The tracked `configs/experiments/fusion.example.yaml` shows the strict study schema. Replace its
fixture data and fingerprint with the resolved values from a canonical dataset manifest, then
materialize an immutable task matrix with:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli plan-fusion-study \
  --config configs/experiments/fusion.example.yaml \
  --output outputs/analysis/fusion/tasks.json
```

The command validates all model/fusion combinations through the production registries and writes
`fusion-study-matrix-v1` atomically. The artifact includes every resolved task and a SHA-256
integrity fingerprint; `load_fusion_study_matrix` verifies it and returns the exact
`ExperimentTask` values accepted by `ExperimentRunner`. The canonical command/API shares model
implementations with production training.
