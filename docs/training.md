# Training engine

`mmctr.training.TrainingEngine` owns the training lifecycle. Models only implement
`forward(Batch) -> ModelOutput`; they do not construct optimizers, choose devices, compute metrics,
or write files.

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
