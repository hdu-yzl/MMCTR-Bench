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

Legacy models still exposing `fit`, `evalate`, `save`, and `load` remain behind compatibility
imports until the single-base-model migration is complete. New code must not call those methods.
