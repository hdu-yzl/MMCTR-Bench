# EB-NeRD canonical data

The provider is [Ekstra Bladet](https://recsys.eb.dk/dataset/). This adapter uses the
official **small** bundle and the Google multilingual BERT and Ekstra Bladet image embeddings.
Obtain the seven Parquet inputs under `data/raw/ebnerd/`; see the raw placement README.

## Reproducible conversion

Activate the existing project environment. Set `MMCTR_PYTHON` to that environment's absolute
interpreter path, then run from the repository root:

```bash
"$MMCTR_PYTHON" -m mmctr.data.datasets.ebnerd
"$MMCTR_PYTHON" -m mmctr.data.datasets.ebnerd --verify-only
```

The default output is `data/processed/ebnerd/canonical-v1`, with history length 5.
Use `--raw-dir`, `--output-dir`, and `--sequence-length` for an explicit new version.
`--config-output PATH` writes a dataset catalog entry derived from the generated manifest.
When using different inputs or history length, update the EB-NeRD entry in
`configs/datasets/catalog.yaml` from that generated entry. Local path injection alone
does not change the catalog's dimensions or fingerprint. A mismatched existing output is
rejected; a matching existing output is fully hash-verified rather than rebuilt.
Conversion uses a temporary directory on the output filesystem and publishes only on success.

## Evaluation and label protocol

- Sort official training impressions by timestamp, then impression ID. The timestamp at
  the 90% row boundary starts local validation; all impressions at that timestamp stay
  together. Actual row ratios can differ slightly from 90/10 because of ties.
- Local `train` precedes local `val`. Official validation forms the full local `test` parent in canonical-v1.
  The active canonical-v2 retains a fixed 10% sample of its impressions. This is a labeled **local holdout**, not the official challenge test.
- Expand each displayed candidate into one binary CTR sample. Membership in the official
  clicked list determines the label; repeated clicks of the same candidate collapse to
  one label and are counted in the manifest. Preserve candidate order and all negatives.
  No catalog-negative sampling, impression duplication, or balancing is performed.
- All candidates in one impression remain in one split. Batches can span impression
  boundaries; grouping metrics must regroup by impression ID across batches.
- Use the provider's fixed history for each source split, sort by historical timestamp,
  retain the last L events, and left-pad with zero. Histories must strictly predate the
  first impression of their source split. Do not append current clicks while expanding
  candidates. Local validation uses the original training history; local test uses the
  original official-validation history, which may legitimately include earlier training clicks.
- No global popularity, engagement totals, next-read-time, or post-impression field is used.
  The existing evaluator reports candidate-level ROC-AUC/LogLoss, not news-ranking
  MRR/nDCG or the official challenge score.

## IDs, modalities, and missing data

User IDs are sorted and compacted into 1..U. Item IDs occupy U+1..U+I; zero is padding.
The vocabulary uses the union of source candidates and histories, without fitting labels.
Raw-to-encoded mappings are retained in `mappings/original_user_ids.npy` and
`item_features/original_item_ids.npy`; the corresponding row position identifies the
mapping. Static feature tables may be shared across splits.

The feature store contains text (768 dimensions for the downloaded release) and image
(1024 dimensions), aligned by article ID, never by source file row order.
Absent embeddings produce zero vectors and explicit `text_present.npy` /
`image_present.npy` boolean arrays. Missing images do not remove examples or history items.
Real items with missing images retain their history mask; padding has all masks false.
The manifest records missing counts. Existing model masking uses the zero input convention;
analysis can inspect the explicit presence metadata.

Each split includes user/item/history IDs, float32 labels, impression IDs, microsecond
timestamps and original candidate positions. Loader metadata carries those audit fields
and `modality_presence[modality][item|history]`. A unique candidate event ID is
`impression_id:candidate_position` for cold-start and event overlap audits.

The manifest fingerprints all seven sources, generated arrays, split statistics, protocol,
ID range and modality dimensions. `--verify-only` checks every generated array's SHA-256.
Data and generated arrays stay out of Git.

## Active canonical-v2: 10% test impressions

The dataset catalog and EB-NeRD analysis examples now use
`data/processed/ebnerd/canonical-v2`. Retain canonical-v1 as the full-test parent.
Generate the active version after preparing canonical-v1:

```bash
"$MMCTR_PYTHON" -m mmctr.data.datasets.ebnerd \
  --sample-test-from data/processed/ebnerd/canonical-v1 \
  --output-dir data/processed/ebnerd/canonical-v2 \
  --test-fraction 0.1 --sampling-seed 2025
"$MMCTR_PYTHON" -m mmctr.data.datasets.ebnerd \
  --output-dir data/processed/ebnerd/canonical-v2 --verify-only
```

Sampling is uniform without replacement over sorted unique impression IDs, using
NumPy RandomState seed 2025. The retained impression count is ceil(N * 0.1).
Every candidate and label belonging to a selected impression is retained, in original
order. Sampling never inspects labels or scores. Training, validation, histories,
user/item mappings and feature tables remain byte-identical to the parent.

The sampled test has **24,465 impressions, 292,271 candidate samples and 24,559 positive
labels** (9.9787% of the full candidate count). Training still has 2,325,624 samples,
validation 260,123, and history length remains 5. The source parent and sampling
parameters enter the new fingerprint. Report this sampled protocol explicitly when
comparing experiments; old full-test metrics are not the same evaluation protocol.

## Training and analysis

```bash
"$MMCTR_PYTHON" -m mmctr.cli train --dataset-name ebnerd --model-name dnn_mm_seq
bash scripts/train_all_models.sh --dataset ebnerd --python "$MMCTR_PYTHON" --dry-run
"$MMCTR_PYTHON" -m mmctr.cli plan-robustness-study \
  --config configs/experiments/ebnerd.robustness.example.yaml \
  --output outputs/ebnerd/robustness-matrix.json
"$MMCTR_PYTHON" -m mmctr.cli plan-fusion-study \
  --config configs/experiments/ebnerd.fusion.example.yaml \
  --output outputs/ebnerd/fusion-matrix.json
"$MMCTR_PYTHON" -m mmctr.cli plan-alignment-study \
  --config configs/experiments/ebnerd.alignment.example.yaml \
  --output outputs/ebnerd/alignment-matrix.json
```

EB-NeRD is registered through the same loader, catalog and `Batch` contract as the existing
datasets. `MMCTR_EBNERD_DATA_DIR` is supported with `--use-local-data`.
Study commands produce task matrices; they do not themselves execute full studies.
Efficiency and plotting use the existing generic APIs and result schema.
Cold-start requires a separately audited subset; this ordinary chronological split is
not itself a cold-start split.

QARM and PSRQ use `get_multi_modal()` through the existing quantizer entry points.
They require their own EB-NeRD artifacts; artifacts from another dataset must not be reused:

```bash
bash scripts/pretrain_quantizers.sh --dataset ebnerd --python "$MMCTR_PYTHON" --gpu 0
```

This conversion does not claim that quantizers or full benchmark experiments have been trained.


## Full-test canonical-v1 statistics

| Split | Candidate samples | Positive labels |
|---|---:|---:|
| train | 2,325,624 | 210,401 |
| val | 260,123 | 23,362 |
| test (official validation) | 2,928,942 | 245,622 |

The complete vocabulary contains 18,827 users and 13,105 articles.
There are 2,447 articles without image embeddings and no missing text embeddings;
1,181 repeated click entries were collapsed to binary labels. The generated store uses
approximately 511 MiB. Fingerprint:
`68392299795171edfa0180f701ff04f7ab436c24fd0e86081849e2339c47ab0f`.

## canonical-v3: structured feature enhancement

The opt-in `residual-mlp-v1` protocol adds **517 floating-point inputs** in
`Batch.context_features["structured"]`, alongside the original user/item IDs,
five history tokens and text/image embeddings. These are encoded dimensions,
not 517 independent raw columns. Exact ordered names and exclusions are embedded
in the fingerprinted manifest.

| Block | Dimensions | Inputs |
|---|---:|---|
| Candidate article | 228 | Category, subcategory, topics, NER clusters, entity groups, article type, sentiment label/score, premium, title/subtitle/body lengths, image count, missing indicators |
| Full fixed-history profile | 240 | Mean article attributes across all official prior clicks; click/unique counts, diversity, time span, prior read/scroll means/maxima/missing rates, fractions near the latest historical click |
| Impression context | 38 | Device, gender/postcode with missing buckets, SSO, age/missingness, cyclic hour/weekday, candidate-set size, ages of first/latest historical click |
| Candidate/profile interactions | 11 | Publication age/missing/future flag, historical candidate frequency, seven hashed attribute-profile overlaps |

Categoricals use deterministic namespaced SHA-256 hashing, with normalized
multi-hot values and explicit missing tokens. Hash collisions are possible;
overlaps are hashed similarities, not exact category equality indicators.
Numerics use fixed clipping/log scales, never validation/test-fitted statistics.
Profile read/scroll values come exclusively from `*_fixed` history arrays.
History recency fractions are relative to the last fixed click; context recency
uses the actual exposure timestamp. No within-split label updates are performed.

Excluded: click labels as features, current/next read and scroll outcomes,
article lifetime engagement aggregates, last-modification time, subscriber
snapshot status, raw session ID, current-page article ID, URL and candidate order.
Official article metadata and pretrained vectors are snapshots rather than
point-in-time article revisions; this inherited limitation is explicit.

All original v2 arrays remain byte-identical, including its 10% whole-impression
test sample. A separate `structured/` store deduplicates articles, history
profiles and impressions; the loader assembles the vector per batch. The original
five-token sequence and modality/quantizer interfaces remain unchanged.
The existing EB-NeRD quantizers can be reused because their input embeddings and
item mappings are byte-identical.

The central model factory attaches the same 517→128→1 ReLU residual logit branch
to all 23 registered backbones when `structured_feature_dim: 517` and
`structured_feature_protocol: residual-mlp-v1` are configured.
Training, canonical analysis and artifact-based model factories use this path.
Direct class construction does not opt in; use `create_model` or
`create_model_from_artifacts`. Missing/mismatched structured inputs fail loudly.

This is a **feature-enhanced model protocol**, including for formerly ID-only
baselines, and must be labeled accordingly in tables. It is not an unchanged
reproduction of each paper's architecture. Old configs instantiate precisely the
old architecture. Modality dropout continues to target text/image embeddings;
it does not remove these separate structured attributes.

Reproduce with the existing server interpreter (absolute path resolved in bm):

```bash
/path/to/bm/bin/python -m mmctr.data.datasets.ebnerd \
  --enrich-from data/processed/ebnerd/canonical-v2 \
  --raw-dir data/raw/ebnerd \
  --output-dir data/processed/ebnerd/canonical-v3 \
  --config-output outputs/ebnerd-v3-config.yaml
```

Enrichment checks raw source hashes against the parent, publishes atomically,
and validates existing outputs on repeat runs. Original v1/v2 artifacts,
completed main-table results and the frozen recovery runtime are retained.

## canonical-v4: categorical and bucketed ID fields (current)

**The v3 residual prediction protocol above is retired.** V3 is retained only as
an immutable preprocessing intermediate and audit record. Model factories reject
the old residual configuration. V4 has no structured float input and no added
prediction head.

The current schema contains **56 additional ID fields**, plus the original
user ID and item ID: **58 ID fields in total**.

- 19 categorical fields: seven candidate article attributes, seven preferred
  historical attribute sets, device, gender, postcode, exposure hour and weekday.
- 37 scalar fields converted to bucket IDs: eight article numeric/indicator
  attributes, twelve history statistics, six exposure statistics, and eleven
  candidate-history interaction statistics.

Each field has its own disjoint range within the existing global ID namespace.
Vocabulary fitting uses only local-training candidates, local-training users'
prior fixed histories, and local-training impressions. Validation/test-only
categories map to the field's OOV ID. Missing and OOV are distinct from padding.
Categorical token IDs have no hashing collisions. Candidate sets are deduplicated
and sorted; history sets select frequent tokens with lexical tie breaks, capped
at 16 tokens per field. Individual token embeddings are masked-mean pooled
within that field.

Continuous values use fixed bin edges on the previously documented fixed
log/clipping scales, equivalent to deterministic non-uniform boundaries in
original units. All exact edges, vocabularies, field offsets, widths and names
are stored in the fingerprinted manifest. Explicit missing-indicator fields
distinguish missing numeric values where applicable. The seven attribute overlap
statistics inherit v3's hashed profile similarity definition; they are numeric
statistics bucketized into IDs, not categorical vocabularies.

The loader exposes additional fields as integer
`Batch.user_features["attribute_ids"] [B,56,16]`, alongside the original singleton
`user_features["id"]`. The group contains **exposure-side ID attributes**, including
candidate/context attributes; it is not a claim that all fields are static user
properties. Numeric and single-value fields use one non-padding token; remaining
slots are zero. Actual user/item identities and history IDs remain unmodified
for analysis grouping and embedding lookup.

Every new token is looked up in the **same existing FeatureEmbedding table** as
user ID and item ID. Every field therefore yields the same `latent_dim` width.
For example, with latent_dim=128, the 58 fields each have width 128 before the
existing ID projection/fusion. No 517-dimensional dense residual branch exists.

Pooled models concatenate all identity/attribute embeddings into their existing
target ID projector. Sequence models concatenate user/exposure attribute fields
into their existing user ID projector, retaining singleton target/history news
IDs and their attention semantics. DIN, MB and executable pipeline presets use
the corresponding field-aware widths. Original predictor and auxiliary-loss
formulas are retained. Configs without additional fields preserve the old path.

V4 keeps the original 28 v2 data arrays byte-identical, including train/val/test
and 10% test sampling. Text/image quantizers remain reusable. Old checkpoints
require matching embedding/projector dimensions when loading.

Build with the absolute server interpreter:

```bash
/path/to/bm/bin/python -m mmctr.data.datasets.ebnerd \
  --ids-from data/processed/ebnerd/canonical-v3 \
  --raw-dir data/raw/ebnerd \
  --output-dir data/processed/ebnerd/canonical-v4 \
  --config-output outputs/ebnerd-v4-config.yaml
```

### ID embedding width

EB-NeRD sets `id_embedding_dim: 16` in its dataset configuration.
The shared factory, Trainer's resolved model configuration, and pipeline presets
apply this to `latent_dim` for this dataset only. User/item/additional field
embeddings share width 16. Projection dimensions and all other training
hyperparameters retain their existing values. Other datasets keep their model
catalog defaults.

Generated manifests include fitted attribute vocabularies and remain local with the dataset.
Reproduce them with the preprocessing commands; do not publish them as source files.
