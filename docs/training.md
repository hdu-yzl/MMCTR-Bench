# Training engine

`mmctr.training.TrainingEngine` owns the training lifecycle. Models only implement
`forward(Batch) -> ModelOutput`; they do not construct optimizers, choose devices, compute metrics,
or write files.

## Paper protocol and public defaults

The formal protocol in paper §3.5 and Appendix C / Table 4 uses Adam, batch size `512`, at most `20`
epochs, validation-AUC early stopping with patience `5`, and random seed `2025`. The public
`configs/training/default.yaml` exposes those values. Shared representation defaults are ID embedding
dimension `128`, modality projection dimension `128`, and prediction MLP `[1024, 512, 256]`; these live
in the model catalog rather than the training schema.

The paper also specifies Xavier parameter initialization. This is a reproduction requirement, not a claim
that the engine globally reinitializes every model: initialization remains model-owned, and only modules
with an explicit Xavier initializer currently enforce it instead of their PyTorch constructor default.

The formal hyperparameter search is Optuna random search with seed `2025`, with a per-model budget of
**50 trials or 50 GPU-hours, whichever is reached first**. The current training config describes one
resolved run and does not enforce a study-level GPU-hour budget. Smaller examples are smoke/planning
examples, not the paper protocol.

The engine performs these steps:

1. consume canonical train batches and optimise BCE-with-logits plus named auxiliary losses;
2. evaluate only the validation split for early stopping and model selection;
3. atomically write run-local `checkpoints/last.pt` and `checkpoints/best.pt` payloads;
4. restore the best model before returning a structured `RunResult`;
5. leave test evaluation as a separate, explicit call after configuration is frozen.

Checkpoint payloads are schema-versioned and contain model state, optimizer state, epoch,
validation metric, and caller metadata. `resume()` restores `last.pt` and returns the next epoch.
It also restores validation best/epoch, accumulated bad epochs, patience, and run identity so a
resumed job cannot silently reset early stopping or cross run boundaries.
The engine accepts a metric writer such as `RunContext.append_metrics`, keeping artifact layout and
training behavior separate.

Models with genuinely alternating objectives use an explicit composition rather than owning a
private `fit()` loop. `PhasedAdam` stores disjoint named parameter groups in one serializable Adam
state. Its normal `step()` updates only `main`; `step_phase(name)` temporarily hides every other
group's gradients. `AlternatingPhase(name, start_epoch, objective)` tells `TrainingEngine` which
scalar objective to run and the inclusive epoch at which it becomes active. The engine always runs
the main BCE/auxiliary objective first, followed by active phases in declared order.

GMMF is the first consumer. The Trainer composes `main`, `discriminator`, and `generator` groups
from registry metadata, preserves their separate learning rates, and runs discriminator then
generator after `epoch >= N`. All three per-parameter Adam states are saved in the same run-local
checkpoint, so resume cannot silently reset the adversarial phases. Ordinary models continue to
use the unchanged single-optimizer path.

Every formal registry entry resolves to a canonical `Batch -> ModelOutput` model. The old
model-owned `fit`, misspelled `evalate`, `save`, and `load` interfaces and their compatibility
imports have been physically removed; numerical regression is retained through self-contained
fixtures rather than a second runtime implementation.

## One-click launchers

The repository-level Bash scripts are thin orchestration wrappers; they do not implement a second
Trainer. Activate the validated environment, resolve its interpreter, and preview a command before
launching it:

```bash
export MMCTR_PYTHON="$(command -v python)"

scripts/train_model.sh --dataset antm2c --model dnn_mm_seq --gpu 0 --dry-run
scripts/train_model.sh --dataset antm2c --model dnn_mm_seq --gpu 0
```

`train_all_models.sh` creates one sequential worker per unique GPU and assigns models round-robin.
The default set contains the 21 models that do not require quantization artifacts:

```bash
scripts/train_all_models.sh --dataset antm2c --gpus 0,1,2,3,4,5,6,7 --dry-run
scripts/train_all_models.sh --dataset antm2c --gpus 0,1,2,3,4,5,6,7
```

Use `--models dnn,dcn,deepfm` for a subset. QARM/PSRQ require
`scripts/pretrain_quantizers.sh` and the explicit `--include-quantized` option. All launchers
validate registry names, accept `--use-local-data`, propagate worker failures, and resolve paths
relative to the repository rather than the caller's current directory.
