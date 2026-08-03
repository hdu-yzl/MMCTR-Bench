# Architecture

MMCTR is migrating from legacy top-level packages to a single `mmctr` namespace. New code follows
this dependency direction:

```text
cli -> experiments -> training/evaluation -> models + data + quantization -> core/config/utils
```

## Core contracts

`mmctr.core` is the boundary shared by data adapters, models, training, and experiment runners.

- `Batch` owns named user, target-item, and history feature mappings, a boolean `[B, L]` history
  mask, float `[B]` labels, and optional metadata. IDs use `torch.long`; continuous features use a
  floating dtype. Construction validates batch and sequence dimensions without mutating caller
  mappings.
- `ModelOutput` owns float `[B]` logits, named scalar auxiliary losses, and optional named
  representations. Training code decides how auxiliary losses are weighted.
- `RunResult` is the in-memory result returned by runners and contains status, finite metrics,
  artifact location, error, and metadata.

Legacy loaders and models are accepted only through `Batch.from_legacy` and
`ModelOutput.from_legacy`. This keeps tuple ordering and the historical `pred`/`au_loss` keys out
of new core code while model families are migrated incrementally.

The core contracts do not create optimizers, select devices, write checkpoints, compute sklearn
metrics, or parse command-line arguments. Those responsibilities belong to the training,
evaluation, experiment, and CLI layers.

Quantization premodels form a sibling boundary to CTR models. RQ/PSRQ consume item
feature tables and produce versioned artifacts, while QARM/MCCA remain ordinary
`BaseSeqModel` consumers after the composition layer injects those loaded
artifacts. This keeps pretraining objectives and artifact I/O out of
`Batch -> ModelOutput` and out of model constructors.
