# Data contracts

Training code consumes `mmctr.core.Batch` objects from a loader implementing
`mmctr.data.DataLoaderProtocol`. It does not call TensorFlow or interpret positional tuples.
`Batch.context_features` keeps interaction-owned context separate from target item features;
models must opt in to those named fields rather than relying on concatenation offsets.

All four dataset registry entries resolve to canonical named-array loaders. AntM2C uses a sharded
layout because its interaction contexts are much larger; MicroLens, TikTok, and EB-NeRD use named
arrays per split. All preserve their audited manifest and emit `Batch` without a TensorFlow or positional-
tuple hop. Compatibility iterators remain for analysis and quantization callers during migration.

Every processed dataset is described by a `DatasetManifest` containing:

- schema, dataset, and data versions;
- storage format and sequence length;
- padding/OOV IDs and explicit ID offsets;
- named feature dimensions;
- per-split counts and optional SHA-256 digests;
- a stable manifest fingerprint used in run metadata.

Manifests synthesized from current YAML are marked `legacy-config-only`. They are compatibility
metadata, not proof that source files or split statistics were audited.

AntM2C also has a restartable batch-extraction boundary and a sharded named-array publisher.
Extraction loads an encoder once per unfinished job, atomically checkpoints each shard, validates
shape/finiteness/checksums, records missing keys, and resumes only when source fingerprints match.
The canonical store writes every interaction context independently and title/image once in a shared
item store. Its loader memory-maps those arrays, gathers target/history modalities from the same
rows, preserves padding while applying explicit ID offsets, and yields canonical batches. Published
shards and logical source files carry SHA-256 evidence, and incomplete conversion state cannot be
opened as the final dataset.

The full Linux artifact is `data/processed/antm2c/canonical-v1/`, fingerprint
`ece3afcc876853caadd4e277421c938a59426e2e003577b1fd02b90e2d3b2aca`. It contains 8,536,558
train, 721,500 validation, and 1,932,927 test events in 1,059/91/239 atomic shards. Its 15,284 files
occupy 173,233,549,382 bytes. The manifest anchors both item feature hashes and the aggregate shard
metadata hash; the full verifier checks every file's hash, size, dtype, shape, and split count.

## MicroLens canonical-v1

`prepare_microlens(raw_dir, output_dir)` consumes `item_feature.parquet` and `train.parquet`.
It streams the 3.6M interactions, keeps the last five source history items, shifts item IDs by
1,000,000, and creates a deterministic 80/10/10 interaction split with seed 42. Text and image
embeddings are written once in an item store; target and history items gather from the same rows.
The parquet item features are already upstream BERT/CLIP embeddings: this processor does not encode
original video frames or raw text. Their source-file SHA-256 values were rechecked against the
read-only reference and exactly match the canonical manifest.
The split definition and sample assignment are recorded in the canonical-v1 manifest.

The server artifact is `data/processed/microlens/canonical-v1/`. Its manifest fingerprint is
`1e28692d5b7f5c722c6621f78533a1974eef9f9b64256a6a3e51e8a27889221f` and records 2,880,000 train,
360,000 validation, and 360,000 test interactions.

## TikTok canonical-v1

`prepare_tiktok(raw_dir, output_dir)` consumes the official `train.json`, `val.json`, and
`test.json` plus the three modality arrays. Each official positive event
uses its prefix in the supplied list order: validation inherits train history, and test inherits train plus
validation history. Source files provide no reliable timestamps, so this is not proof of temporal
causality. Five deterministic negatives are sampled from items never observed by that
user across the three source splits. The manifest records the split, history, and sampling rules.

The text/image/audio arrays are pre-extracted upstream features rather than raw media. All six
source-file SHA-256 values were rechecked against the read-only reference and exactly match the
canonical manifest. Canonical-v1 stores generated CTR samples alongside these upstream features.

The server artifact is `data/processed/tiktok/canonical-v1/`. Its manifest fingerprint is
`a908bb25f28f899bf2cf21f81483ae67d2106adea582e0e5576a2e9cb1ff05bc` and records 357,246 train,
18,306 validation, and 36,780 test CTR samples.

The current arrays contain **412,332** samples in total, **9,308 distinct users** and
**6,710 distinct target items** across all splits, with a positive:negative ratio of **1:5**.
Counts below are measured from emitted arrays, rather than user keys in source JSON:

| Split | CTR rows | Positives | Negatives | Distinct users | Distinct target items |
|---|---:|---:|---:|---:|---:|
| Train | 357,246 | 59,541 | 297,705 | 9,298 | 6,710 |
| Validation | 18,306 | 3,051 | 15,255 | 3,051 | 6,176 |
| Test | 36,780 | 6,130 | 30,650 | 6,130 | 6,660 |

User and item counts above refer to distinct IDs in emitted candidate arrays. Source JSON user
keys with empty positive lists do not contribute CTR rows. See the
[benchmark protocol](benchmark-protocol.md) for evaluation and aggregation details.

Both processors are idempotent for the same source fingerprint and options. They refuse to reuse
an existing output directory built from different inputs. Raw/private arrays and generated output
remain ignored by Git; only the schema, code, protocol, and manifest evidence are versioned.
