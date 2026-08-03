# Model component contracts

`mmctr.models.components` is the public boundary for projection, dimension adaptation, and masks.
It operates on tensors that have already passed model-specific ID embedding, quantization, or
paper-specific encoders. It does not inspect `Batch` or silently infer which field belongs to which
branch.

## Named projection

`NamedFeatureProjector` receives a non-empty mapping from feature name to input dimension and one
common output dimension. A call must provide exactly those names. Every tensor must be floating
point, rank 2 or 3, and match its declared final dimension. The projector returns a new dictionary
and never mutates the input mapping or tensors.

Each named field owns a learned linear projection even when its input and output dimensions match.
This preserves the canonical models' existing learned projection semantics. Target and history
may share one projector instance when the model previously shared parameters; pooled target and
history branches may retain separate instances.

An optional boolean presence mapping has the same prefix shape as its field (`[B]` or `[B, L]`).
It is applied after the learned linear layer, so a zero/missing modality remains zero rather than
turning into the projection bias. Missing or unknown feature names, integer tensors, unsupported
ranks, and dimension mismatches raise `ContractError` at the component boundary.

## Dimension adaptation

`DimensionAdapter` changes only the final tensor dimension and preserves every prefix dimension.
It defaults to identity when input and output dimensions are equal; callers can request a learned
same-size transform with `identity_if_same=False`. The same dtype, rank, and final-dimension checks
apply before either identity or linear execution.

## Mask utilities

- `validate_sequence_mask` enforces a boolean `[B, L]` mask and checks an optional sequence prefix.
- `apply_sequence_mask` zeros padded positions in floating `[B, L, ...]` tensors.
- `feature_presence` defines the current missing-modality compatibility rule: all-zero final vectors
  are absent. It is not a replacement for the authoritative history padding mask.
- `apply_feature_mask` applies an explicit boolean prefix mask after projection or encoding.
- `masked_softmax` excludes masked entries and returns zeros for an all-masked slice.

Canonical pooled and sequence multimodal backbones now use these components. History validity is
the intersection of `Batch.history_mask` and non-ID modality presence; ID history uses the explicit
history mask only. The all-zero compatibility rule remains explicit so a future data schema can
replace it with first-class modality-presence fields without changing every model.

## Pooling

Every public pooling component uses:

```text
forward(sequence: FloatTensor[B,L,D], mask: BoolTensor[B,L],
        target: Optional[FloatTensor[B,D]] = None) -> FloatTensor[B,D]
```

The lazy `POOLING_REGISTRY` exposes `mean`, `sum`, `max`, `attention`, `din`, and
`cross_attention`; `average` is an alias for `mean`. Each registry entry declares input/output
rank, mandatory mask support, whether a target is required, and the output-dimension rule. The
runtime class exposes the same immutable `PoolingCapability`, plus `input_dim` and `output_dim`.

Mean/sum/max always return zeros for an all-padding row. Learned attention uses a masked softmax
whose all-padding result is also zero. DIN requires a target and preserves its local-activation
formula. Cross-attention uses target queries and multi-head history keys/values; it rejects a head
count that does not divide the feature dimension. All implementations validate dtype, rank,
dimension, mask shape, device, and target compatibility before model operations.

`BaseSeqModel.masked_pool`, the existing `DinAttention` compatibility name, and NAML's private
encoder now delegate to state-key-compatible public implementations. Model-private attention is
left unchanged when it represents a different paper formula. Fusion, auxiliary losses, and modal
pipeline topology remain separate follow-up contracts.
