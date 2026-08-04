# Architecture

MMCTR uses a single installable `mmctr` namespace. Runtime code follows this dependency direction:

```text
cli -> experiments -> training/evaluation -> models + data + quantization -> core/config/utils
```

## Paper protocol versus executable surface

The paper's formal protocol is broader than the currently executable generic analysis surface. It defines
five alignment objectives (`KL`, `InfoNCE`, `Cosine`, `MMD`, `Adv`) while the runtime accepts only
`cosine` and diagnostic `mse`; it evaluates nine fusion replacements (`CAT`, `DMF`, `DTA`, `FQ-Former`,
`LMF`, `MAF`, `MTFN`, `SimCEN`, `SRC`) while the generic registry exposes `concatenate`, `sum`, `mean`,
`maf`, `lmf`, and `mtfn`. Protocol documentation must preserve the paper scope, whereas configurations
must contain only accepted runtime names. Missing implementations are explicit gaps, never silently
approximated operators.

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

Every formal registry model and dataset is canonical. `Batch.from_legacy` and
`ModelOutput.from_legacy` remain explicit format adapters for frozen regression fixtures and the
audited AntM2C TFRecord conversion path; no legacy model or loader package is published.

The core contracts do not create optimizers, select devices, write checkpoints, compute sklearn
metrics, or parse command-line arguments. Those responsibilities belong to the training,
evaluation, experiment, and CLI layers.

Quantization premodels form a sibling boundary to CTR models. RQ/PSRQ consume item
feature tables and produce versioned artifacts, while QARM and the PSRQ CTR model remain ordinary
`BaseSeqModel` consumers after the composition layer injects those loaded
artifacts. This keeps pretraining objectives and artifact I/O out of
`Batch -> ModelOutput` and out of model constructors.
