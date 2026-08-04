# Model contract and migration

The shared projection, dimension, sequence-mask, and missing-modality rules are specified in
[`model-components.md`](model-components.md). Canonical backbones consume those public components;
pooling and fusion remain separate staged interfaces.

## Paper-wide representation defaults

Under paper §3.5 and Table 4, sparse ID embeddings use dimension `128`, every non-ID modality is
projected to dimension `128`, and the shared prediction MLP is `[1024, 512, 256]` unless a model's
original architecture requires a documented exception. The model catalog exposes these values.
The paper protocol also requires Xavier initialization. Initialization is model-owned in the current
runtime: explicit Xavier calls exist in several components, but there is no global post-construction
initializer proving that every registered parameter follows the paper rule. Formal reproduction must audit
that model-specific boundary rather than infer compliance from the YAML dimensions.

## Default modal-pipeline presets

`mmctr.models.default_pipeline_preset(model_name, model_config, data_config)` resolves a model's
default branch behavior after configuration merging. The returned frozen `ModelPipelinePreset`
either builds a `ModalPipelineSet` or records why a paper-specific path cannot yet be expressed by
the public components. `model_pipeline_coverage()` covers every canonical registry name, and the
legacy alias `dnn_seq` resolves to the `dnn_mm_seq` preset.

Thirteen models currently have executable, numerically checked presets:

- `dnn`, `dcn`, `deepfm`, and `autoint`: separate target/history ID projection with masked mean
  history pooling before each model-specific interaction head;
- `dnn_mm` and `dnn_mm_seq`: target fusion plus per-modality mean history pooling before fusion;
- `lmf` and `mtfn`: target fusion plus token fusion before masked mean history pooling;
- `naml`: shared target/history MAF and learned attention history pooling;
- `make`: shared configurable fusion and target-aware DIN history pooling;
- `simcen`: ordered projected target/history fields before its private segmentation and expert body;
- `qarm`: quantized target fields plus per-field masked mean history pooling after code embedding.
- `dmf`: shared non-ID target/history modality-center projection and fusion before its private DTA
  and similarity-tier head, plus the canonical user branch.

Sequence presets retain the existing shared target/history projector. NAML, MAKE, and DMF also
retain their shared fusion instance; this is part of parameter semantics, not an implementation detail.
Preset regression tests load the existing branch weights into the equivalent pipeline and compare
target, history, and user representations at `1e-7` absolute tolerance.

All other registered models have an explicit non-executable decision with a concrete reason. This
includes paper-private SRC/FQ-Former, expert/MoE, PGD, decomposition, quantization, or GMMF DSN
fusion paths. DIN also remains model-specific: a regression attempt found that its current
Dice normalization observes padded-position projection bias, while the public pipeline masks that
bias before pooling; the resulting effective representation difference is not silently accepted as
compatible. Calling `build()` on a non-executable preset raises `ContractError`. Such models must
not be approximated with a convenient public mean/cat component; the corresponding algorithm or
compatibility decision must first gain checkpoint and numeric regression coverage.

The only public model base is `mmctr.models.BaseSeqModel`. Despite the name, it supports both
non-sequential models that consume pooled history and models that retain sequence tokens. Each
model declares exactly one `HistoryCapability`:

- `pooled_history`: use `BaseSeqModel.masked_pool` to reduce `[B, L, D]` before the model-specific
  body;
- `sequence_tokens`: retain `[B, L, D]` and use `Batch.history_mask` in sequence interactions.

The base owns no optimizer, device selection, logger, metric implementation, checkpoint path, or
training loop. Its public forward signature is always `forward(batch: Batch) -> ModelOutput`.

The former `models` package, dual training-owning bases, and forward-signature adapter are no
longer distributed. Historical numerical evidence is retained as fixed canonical fixtures instead
of keeping a second executable model implementation in the wheel.

## Migrated baseline family

The formal registry now resolves `dnn`, `dcn`, `deepfm`, `autoint`, and `din` to pure implementations
under `mmctr.models.baselines`. DNN/DCN/DeepFM/AutoInt concatenate separate user and target-item IDs,
then apply mask-aware mean history pooling before their original model-specific body. DIN retains
history tokens and applies its target-aware attention with the explicit history mask. All five keep
logits at `[B]`, including batch size one.

Callers construct models with `mmctr.models.registry.create_model(model_name, model_config,
data_config)` and let `TrainingEngine` own optimizer, device, checkpoint, metrics, and run state.
The DNN migration fixture still reproduces the pre-migration logits, loss, and 205-parameter count.

## Migrated simple multimodal family

The formal registry resolves `dnn_mm`, `lmf`, and `mtfn` to pure canonical implementations in
`mmctr.models.multimodal`, while `dnn_mm_seq` lives in `mmctr.models.sequence`. All project target
and history features by name, zero projected padding tokens explicitly, and pool or attend with
`Batch.history_mask`. `dnn_mm_seq` keeps user, target item, and history fields separate and never
uses rank-destroying bare `squeeze()`.

The currently required cat/add/mean/MAF/LMF/MTFN operations are private migration components, not
the future public fusion registry. This preserves the `FUSE-001` boundary while allowing the four
models to leave the former training-owning bases.

`simcen` is the first complex auxiliary-loss migration. Its segmentation and multi-level experts
remain model-private, while contrastive training is exposed as the scalar named loss
`simcen_contrastive` and its ego/two-view tensors are returned through `ModelOutput.representations`.
The stable log-sum-exp formulation avoids the legacy exponentiation overflow path.

## Migrated sequence-token family

`dnn_mm_seq`, `naml`, and `make` now share the pure encoding boundary in
`mmctr.models.sequence`. The boundary projects user, target-item, and history fields by name,
preserves `[B, L, D]` history tensors, applies the explicit boolean history mask after projection,
and never mutates the input batch.

NAML keeps its MAF target/history fusion and learned user-interest attention, but padded tokens are
excluded with masked softmax and an all-padding row returns zero interest. MAKE keeps configurable
migration fusion, target-aware DIN pooling, cosine similarity tiers, and its MLP head; both DIN and
the tier histogram consume the same mask. Similarity diagnostics are returned as representations,
not written by the model. The formal registry points only to these canonical implementations.

DMF and MARN also use this sequence boundary. DMF preserves its non-ID modality-center similarity,
discretised similarity embeddings, decoupled target attention, tier histogram, and weighted dual
interest branches. Its DTA and tier counts now consume `Batch.history_mask`, so padded buckets do
not contribute. MARN preserves modality-specific/shared projections, domain uncertainty weighting,
gradient reversal, MAF, and target-aware history attention. Its three training objectives are named
`marn_domain_classifier`, `marn_adversarial_invariance`, and `marn_specific_classifier` in
`ModelOutput.auxiliary_losses`; the configured `lambda0` weight is applied exactly where the legacy
total applied it. The canonical module does not enable global autograd anomaly detection or create
training state during import/construction.

## Migrated advanced sequence family

`em3` and `diff_msin` live in `mmctr.models.advanced_sequence` and reuse the same named projection
boundary. EM3 preserves its learned FQ-Former query tokens, target-aware DIN history pooling,
content projection, and bidirectional content/item CIC objective. Fused padded history positions
are explicitly zeroed before masked DIN pooling, and the weighted CIC term is exposed as
`em3_content_item_contrastive`.

Diff-MSIN preserves per-modality DIN pooling, specific/shared experts, modal gates, stochastic
reverse cross-modal fusion, the final gate/cross network, and label-aware synthesis hinge loss.
Its former aggregate `au_loss` is exposed as the weighted scalar terms `diff_msin_synthesis` and
`diff_msin_contrastive`; labels come directly from `Batch.labels`.

GMMF now lives in `mmctr.models.gmmf` as a pure canonical model. It preserves its modality
autoencoders, conditional generators/discriminators, automatic difference modules, cosine-weighted
history interests, user-conditioned gates, and weighted reconstruction objective. Reconstruction
is exposed as `gmmf_reconstruction`; discriminator and generator objectives are explicit scalar
methods consumed by the training composition. A checkpoint-compatible phased Adam preserves the
legacy per-batch main → discriminator → generator order and the inclusive `epoch >= N` start rule.
The fixed-seed formula fixture freezes canonical logits, reconstruction, discriminator, generator
losses, and parameter count at `1e-7`. Padded histories are additionally masked in the canonical
path. The DSN/CGAN fusion remains model-specific until it becomes a registered public
fusion component, so GMMF's modal-pipeline preset is still intentionally non-executable.

## Migrated specialized model family

`mb`, `pamd`, `mmmlp`, and `m3srec` are implemented in
`mmctr.models.specialized`. MB keeps its ID and modality scoring branches,
attention fusion, and training-only PGD modality-balancing objective. The weighted
objective is exposed as `mb_modality_balance`, while modality weights and branch
scores are available through `ModelOutput.representations`.

PAMD keeps pairwise common/specific decomposition, cross reconstruction,
orthogonality and ranking terms for both target and pooled history. Their weighted
sum is exposed as `pamd_disentanglement`. MMMLP retains modality-specific and fused
MLP-Mixer stacks. M3SRec retains shared self-attention, modality-specific MoE,
cross-modal attention/MoE, positional/modality embeddings, and attention fusion.

All four models consume the canonical `Batch` without mutation. Pooled models use
explicit masked history means; token models mask padded tokens throughout their
sequence path and locate the last valid token from mask positions, including
left-padded histories. Registry metadata names only the canonical implementation and capabilities.

## Quantization premodels and quantized CTR models

RQ and PSRQ are pretraining components under `mmctr.quantization`, not CTR
`BaseSeqModel` subclasses. The model and quantizer registries are separate namespaces: the
quantizer name `psrq` produces semantic-code artifacts, while the canonical CTR model name `psrq`
uses those artifacts through an internal MCCA consumer. The legacy CTR name `mcca` is an alias only.
RQ fits deterministic residual K-means codebooks and keeps the resulting
`[levels, codes, dimension]` tensor as a registered buffer. PSRQ pretraining owns modality and joint
autoencoders and returns a `PSRQOutput` containing reconstruction/quantization objectives and codes;
it deliberately does not emit click logits or own an optimizer, device, checkpoint path, or training loop.

QARM and the PSRQ CTR model are canonical sequence-token predictors in
`mmctr.models.quantized`. Their constructors accept already-loaded RQ/PSRQ dependencies. They never
resolve paths, create directories, or load checkpoints.
`create_model_from_artifacts` is the composition boundary used by the main
Trainer: it loads the dataset-specific artifacts, validates modalities, raw
dimensions, level count, codebook size and PSRQ architecture, then injects them.

Both models preserve the original discrete-code embedding bodies while using the
canonical `Batch.history_mask`. QARM performs masked mean pooling. The PSRQ model's internal MCCA consumer applies a
masked query/history attention and pins its frozen PSRQ encoder in evaluation mode
even while the CTR head trains. Original zero-valued modalities stay zero after
code lookup, so a nearest code cannot silently turn a missing modality into a
present one. Validation-only tuning and canonical RQ/PSRQ artifact training replace the removed
legacy codebook tuner.
