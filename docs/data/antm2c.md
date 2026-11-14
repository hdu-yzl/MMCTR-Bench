# AntM2C data protocol

This document defines feature ownership, split rules, storage, and preprocessing for the
AntM2C canonical dataset.

## Feature ownership

| Raw field | Canonical field | Owner | Split sharing | Current evidence |
|---|---|---|---|---|
| `service_entity_seq` | `service_text` | interaction context | forbidden | embedded once per interaction and split |
| `query_entity_seq` | `query_text` | interaction context | forbidden | embedded once per interaction and split |
| `bill_entity_seq` | `bill_text` | interaction context | forbidden | embedded once per interaction and split |
| `log_time` | `time_context` | interaction context | forbidden | event timestamp represented as text today |
| `item_title` | `title_text` | item | allowed | current extractor already deduplicates by item ID |
| image named by `original_item_id` | `image` | item | allowed | current extractor builds one item feature matrix |
| `item_entity_names` | `entity_text` | interaction context | forbidden | 247 of 2,820 sampled items have multiple encoded values; one item has 19 versions |

The stratified benchmark sample contains 6,144 rows from train/validation/test and all source shard
groups. It found 247 conflicting items among 2,820 observed items, so one counterexample was enough
to reject item ownership. `entity_text` is therefore interaction context and is never deduplicated
by item ID. The audit still records conflicts and missing values as source-quality evidence.

## Interaction identity and split rules

Every raw row receives the stable identity `(data_version, split, source_file_index, source_row)`
before filtering or concatenation; the split shards store the numeric
`event_id=[source_file_index, source_row]`, while the dataset version and split come from the
manifest/loader metadata. Histories are ordered by `(user_id, timestamp, event_id)` and built in one scan. Only
earlier positive-label events may enter a row's history; the current event and all later events are
forbidden. Repeated item IDs remain distinct events and therefore cannot be located with
`items.index(item_id)`.

The split protocol uses exact cutoffs `2023-08-03 00:00:00` and
`2023-08-05 00:00:00`: train is `<=` the first cutoff, validation is after the first and `<=` the
second, and test is later. Changing these midnight boundaries creates a new data version.

Padding ID is `0`; contiguous `item_index` values start at `1`. The manifest
must record user/item mappings, padding/OOV policy, source hashes, split counts and timestamps.

## Target storage layout

Interaction rows store IDs, label, scene, history item indices, and the five interaction-context
embeddings (service/query/bill/entity/time). The item feature store owns only title and image.
Target and history modalities gather from the same item-indexed arrays. No interaction row stores a
4608-dimensional concatenation or repeats item-owned vectors.

Each split uses its own service/query/bill/entity/time embeddings, stored in layered named
NumPy arrays.

The pure preprocessing primitives are `build_histories`, `build_item_index`, and
`build_feature_store`. Item indices are assigned by first appearance in the stable versioned event
stream, start at one, and reserve row zero for padding. Histories preserve repeated item events and
append a positive item only after emitting the current event. All item modalities use one feature
table; missing rows are zero-filled and returned in `FeatureStoreAudit` rather than silently lost.

`run_batch_extraction` is the extraction boundary. It receives an encoder factory, invokes it
once for the unfinished job, validates every `[batch, dimension]` result, writes atomic `.npy`
shards plus ordered keys, and advances a checksummed resume manifest only after both files exist.
Empty text or missing images receive zeros and their stable keys remain in `missing_keys`.

`InteractionTable` and `AntM2CArrayLoader` remove the serializer/loader dependency on packed
feature positions. Interaction-owned service/query/bill/time features live in
`Batch.context_features`; item-owned title/image features are gathered for both target and history
from `ItemFeatureStore`. `entity_text` remains a fifth interaction context. For existing model
configs that explicitly request user feature `text`, the loader composes a virtual 3840-dimensional
view in the fixed named order service/query/bill/entity/time. Those five arrays remain separate on
disk, and canonical sequence models opt into the context fallback by configured feature name.

The full-data publisher is available as:

```bash
<SERVER_ENV>/bin/python -m mmctr.data.datasets.antm2c.canonical \
  --source-dir <READ_ONLY_LEGACY_ANTM2C> \
  --output-dir data/processed/antm2c/canonical-v1
```

It writes `sharded-named-npy-v1`: each event is identified by split plus
`[source_file_index, source_row]`, each output shard is atomically renamed only after all of its
arrays have SHA-256 metadata, and `canonical-v1.incomplete/` is a resumable staging directory rather
than a training dataset. Item title/image arrays are copied once into a shared store. If a legacy
packed target vector conflicts with that store, the interaction is retained and the shared store is
authoritative; title/image conflict counts remain in the final manifest. `verify_canonical_dataset`
can perform a full post-publication hash audit.

## Full canonical-v1 publication

The publication input was the already-derived legacy dataset available at conversion time: 28
train/validation/test TFRecords plus the shared `text_feature.npy` and `image_feature.npy` item
arrays. This step did not rerun the earlier raw-event, text-encoder, or image-encoder pipeline. The
canonical manifest records every source TFRecord SHA-256, both item-array SHA-256 values, and an
aggregate source fingerprint, so the derived-to-canonical migration is auditable even though the
raw inputs are managed separately.

The Linux server publication completed all 28 authoritative source files and retained every source
event under the shared-item-store-wins policy:

| Split | Events | Positives | Shards | Title conflicts | Image conflicts |
|---|---:|---:|---:|---:|---:|
| train | 8,536,558 | 2,765,092 | 1,059 | 16,348 | 0 |
| validation | 721,500 | 267,870 | 91 | 4,029 | 0 |
| test | 1,932,927 | 734,363 | 239 | 1,068 | 0 |

The final store contains 15,284 files and 173,233,549,382 bytes (about 161.34 GiB). Its manifest
fingerprint is `ece3afcc876853caadd4e277421c938a59426e2e003577b1fd02b90e2d3b2aca`; the manifest directly
anchors the shared title/image SHA-256 values and the aggregate shard metadata hash.

Use `verify_canonical_dataset` to audit array hashes, sizes, dtypes, shapes, and split counts.

## Raw preprocessing and extraction

The raw pipeline consumes the `antm2c_10m_part*` event files, `AntM2C_image.tar.gz`, and local
BERT/ChineseCLIP checkpoints. `iter_raw_events` streams CSV rows, assigns source-row event IDs,
and applies the split cutoffs. `replay_raw_events` builds the item index and causal histories.
Text/image encoder adapters feed `run_batch_extraction`, which writes checksummed resumable
shards and retains missing-input audits.

The format benchmark API compares identical events, performs warm-up, and records median CPU
passes with semantic checksums. Save benchmark and raw-replay reports under ignored `outputs/`
paths. A complete raw re-encoding produces a separately versioned dataset rather than silently
overwriting canonical-v1.
