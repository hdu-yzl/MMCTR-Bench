# Model contract and migration

The only public model base is `mmctr.models.BaseSeqModel`. Despite the name, it supports both
non-sequential models that consume pooled history and models that retain sequence tokens. Each
model declares exactly one `HistoryCapability`:

- `pooled_history`: use `BaseSeqModel.masked_pool` to reduce `[B, L, D]` before the model-specific
  body;
- `sequence_tokens`: retain `[B, L, D]` and use `Batch.history_mask` in sequence interactions.

The base owns no optimizer, device selection, logger, metric implementation, checkpoint path, or
training loop. Its public forward signature is always `forward(batch: Batch) -> ModelOutput`.

`LegacyModelAdapter` is a temporary migration bridge. It copies all feature dictionaries before
calling an old model, joins separate user/item IDs only for old pooled models, and converts
`pred`/`au_loss` output dictionaries into `ModelOutput`. This prevents old in-place mutations from
changing the caller's `Batch`; it does not make the wrapped constructor side-effect free.

The old `models.base_model.BaseModel` and `models.base_seq_model.BaseSeqModel` emit deprecation
warnings and remain solely so unmigrated research models keep working. Model-family tasks replace
them with pure implementations before the compatibility packages can be removed.

## Migrated baseline family

The formal registry now resolves `dnn`, `dcn`, `deepfm`, `autoint`, and `din` to pure implementations
under `mmctr.models.baselines`. DNN/DCN/DeepFM/AutoInt concatenate separate user and target-item IDs,
then apply mask-aware mean history pooling before their original model-specific body. DIN retains
history tokens and applies its target-aware attention with the explicit history mask. All five keep
logits at `[B]`, including batch size one.

`mmctr.utils.helper.getModel` and `helper.resolve_model_class` intentionally continue to resolve the
frozen legacy classes for historical scripts and numerical fixtures. This is a compatibility API,
not the formal registry. New callers use `mmctr.models.registry.create_model` with `(model_config,
data_config)` and let `TrainingEngine` own all training state.

## Migrated simple multimodal family

The formal registry resolves `dnn_mm`, `lmf`, and `mtfn` to pure canonical implementations in
`mmctr.models.multimodal`, while `dnn_mm_seq` lives in `mmctr.models.sequence`. All project target
and history features by name, zero projected padding tokens explicitly, and pool or attend with
`Batch.history_mask`. `dnn_mm_seq` keeps user, target item, and history fields separate and never
uses rank-destroying bare `squeeze()`.

The currently required cat/add/mean/MAF/LMF/MTFN operations are private migration components, not
the future public fusion registry. This preserves the `FUSE-001` boundary while allowing the four
models to leave the training-owning legacy bases. Legacy helper resolution remains frozen for
server numerical regression.

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
not written by the model. The formal registry points to these canonical implementations while the
legacy helper metadata remains available for server regression.

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
`diff_msin_contrastive`; labels come directly from `Batch.labels`. GMMF is intentionally still
legacy-only because its algorithm couples three optimizers with an alternating GAN schedule. It
must migrate together with an explicit multi-optimizer training-engine protocol, not as an
incomplete forward-only model.
