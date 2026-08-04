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
left unchanged when it represents a different paper formula.

## Fusion

Public fusion consumes an exact mapping of configured modality names. Every modality must have the
same rank (2 or 3), prefix shape, floating dtype, device, and final input dimension. An optional
presence mapping must contain the same names and boolean prefix shapes. Missing values are masked
before fusion, including after MAF's learned bias, so absent fields cannot silently become learned
bias vectors.

The lazy `FUSION_REGISTRY` exposes `concatenate`, `sum`, `mean`, `maf`, `lmf`, and `mtfn`.
Compatibility aliases are `cat`, `add`, and `average`. Registry metadata and each immutable
`FusionCapability` declare supported ranks, modality-count bounds, presence support, output
dimension rules, and stable auxiliary-loss names.

Every call returns `FusionOutput(fused, auxiliary_losses)`. The representation keeps the input
prefix and changes only its final dimension. Auxiliary losses must be named floating scalar tensors
on the same device; the six initial compatibility algorithms currently return an empty mapping.
Concatenation reports `modalities * input_dim`, LMF reports its configured output dimension, and the
other initial algorithms preserve the common input dimension.

The DNN-MM, DNN-MM-Seq, NAML, DMF, MARN, LMF, and MTFN compatibility presets now delegate to these
public implementations while retaining parameter names for learned MAF/LMF/MTFN state. Other
paper-private fusion formulas remain local until their equations and checkpoint keys have explicit
regression coverage.

## Configurable modal pipeline

`ModalPipelineSet.from_mapping()` parses the resolved `modal_pipeline` mapping into independent
`target`, `history`, and `user` branches. Target and user branches use `feature_fusion` and consume
rank-two feature mappings. A history branch must select exactly one of these topologies:

- `pool_then_fuse`: each modality owns a pooling component, then the pooled vectors are fused;
- `fuse_then_pool`: token-level modalities are fused first, then one pooling component consumes the
  fused sequence;
- `sequence_fusion`: token-level modalities are fused and remain rank three for a sequence-aware
  model head.

The YAML-like input is strict and does not mutate the caller's mapping. Component entries accept a
short name or a `name/options` mapping. For example:

```yaml
modal_pipeline:
  target:
    modalities: [id, image, text]
    input_dimensions: {id: 128, image: 512, text: 768}
    projection_dim: 128
    fusion: {name: concatenate}
  history:
    topology: pool_then_fuse
    modalities: [id, image, text]
    input_dimensions: {id: 128, image: 512, text: 768}
    projection_dim: 128
    pooling:
      id: {name: din, options: {hidden_dims: [64, 32]}}
      image: {name: mean}
      text: {name: attention}
    fusion: {name: maf}
    output_dim: 128
```

The pipeline performs projection, explicit mask/presence intersection, configured pooling/fusion,
and an optional final-dimension adapter. Target-aware pooling requires an exact same-modality target
mapping; configurations that do not use a target reject one instead of silently ignoring it. A
`ModalPipelineOutput` contains the final representation, its authoritative boolean presence mask,
and immutable named auxiliary losses propagated by fusion. All-padding rows or tokens remain zero
even when projection, fusion, or adapter layers have learned biases.

Configuration structure is checked before forward execution: branch/topology compatibility,
modality uniqueness, exact dimension and pooling keys, positive dimensions, registered component
names, and component constructor constraints fail early. Runtime checks still enforce exact names,
floating dtype, ranks, final dimensions, common device/prefix, boolean masks, and target shape. The
pipeline deliberately does not inspect `Batch`; dataset field ownership and ID embedding remain at
the model boundary.
