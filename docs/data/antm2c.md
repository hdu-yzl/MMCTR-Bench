# AntM2C data protocol

This document records the semantic contract for the rewritten preprocessing pipeline. The initial
ownership proposal came from tracked code; it has now also been checked against a stratified real
TFRecord sample on the Linux experiment server. The original dataset directories remained read-only.

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

The tracked legacy split code uses exact cutoffs `2023-08-03 00:00:00` and
`2023-08-05 00:00:00`: train is `<=` the first cutoff, validation is after the first and `<=` the
second, and test is later. These exact midnight boundaries must be confirmed against the intended
benchmark before regeneration; changing them creates a new data version.

Padding ID is `0`; future contiguous `item_index` values start at `1`. The hard-coded legacy
minimum mapped item ID `67625` is compatibility evidence, not a new loader contract. The manifest
must record user/item mappings, padding/OOV policy, source hashes, split counts and timestamps.

## Target storage layout

Interaction rows store IDs, label, scene, history item indices, and the five interaction-context
embeddings (service/query/bill/entity/time). The item feature store owns only title and image.
Target and history modalities gather from the same item-indexed arrays. No interaction row stores a
4608-dimensional concatenation or repeats item-owned vectors.

Each split uses its own service/query/bill/entity/time embeddings. The legacy serializer always indexes
variables suffixed `_train` inside the train/val/test loop; this is a confirmed bug to remove in
the canonical path. `ANT-006` selected layered named NumPy arrays after the real server benchmark.

The new pure preprocessing primitives are `build_histories`, `build_item_index`, and
`build_feature_store`. Item indices are assigned by first appearance in the stable versioned event
stream, start at one, and reserve row zero for padding. Histories preserve repeated item events and
append a positive item only after emitting the current event. All item modalities use one feature
table; missing rows are zero-filled and returned in `FeatureStoreAudit` rather than silently lost.

`run_batch_extraction` is the new extraction boundary. It receives an encoder factory, invokes it
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
upstream raw inputs are no longer present.

The Linux server publication completed all 28 authoritative source files and retained every source
event under the shared-item-store-wins policy:

| Split | Events | Positives | Shards | Title conflicts | Image conflicts |
|---|---:|---:|---:|---:|---:|
| train | 8,536,558 | 2,765,092 | 1,059 | 16,348 | 0 |
| validation | 721,500 | 267,870 | 91 | 4,029 | 0 |
| test | 1,932,927 | 734,363 | 239 | 1,068 | 0 |

The final store contains 15,284 files and 173,233,549,382 bytes (about 161.34 GiB). Its manifest
fingerprint is `ece3afcc876853caadd4e277421c938a59426e2e003577b1fd02b90e2d3b2aca`; the manifest directly
anchors the shared title/image SHA-256 values and the aggregate shard metadata hash. Full
post-publication verification hashed every array and checked bytes, dtype, shape, and split counts
in 711.558 seconds. Real train/validation/test batches each completed finite `dnn_mm_seq` BCE forward,
backward, and Adam steps on CPU. A 256-event CUDA training step on V100 completed with finite loss
and gradients; the versioned inference efficiency report is retained under the ignored run output
root.

## Linux format benchmark

`ANT-006` materialized the same 2,048 consistent events per split as sampled TFRecord and layered
named arrays under the ignored `data/processed/antm2c/benchmark-v1/`. Sampling is stratified across
source shards, uses batch size 256, performs one complete warm-up per format, and reports the median
of five complete CPU passes. Both paths parse/gather/assemble PyTorch `Batch` objects and produced
the same semantic checksum.

| Measure | Legacy TFRecord | Layered named arrays |
|---|---:|---:|
| Interaction bytes for 6,144 events | 126,903,807 | 95,408,256 |
| Median complete pass | 0.551770 s | 0.090615 s |
| Throughput | 11,135 samples/s | 67,803 samples/s |

The named interaction layout is 24.82% smaller and 6.09× faster in this warm-cache CPU protocol.
Its shared item store is 435,510,515 bytes and is paid once, not once per split. Applying the sampled
interaction ratio to the 231,139,326,681-byte authoritative TFRecord set projected about 174.21 GB
for full named interactions plus the item store, a 57.36 GB reduction. This projection is evidence
for format selection; the final manifest reports measured bytes and counts after full publication.

Four train and ten validation records encountered while filling the stratified sample had packed
target title vectors inconsistent with `text_feature.npy`; none had image conflicts. They were
excluded and counted so both benchmark formats consume identical semantics. The benchmark process
had no visible CUDA device, and the loader protocol contains no H2D transfer, so GPU wait is reported
as unavailable rather than inferred from CPU time. Exact fingerprints, timings, versions, and audit
counts are stored in `antm2c-format-benchmark.json`.

## Raw-pipeline revalidation

The initial prerequisite audit searched only the original read-only reference tree and incorrectly concluded
that the upstream inputs were unavailable. The maintainer subsequently identified the authoritative
read-only inputs on maintainer-provided external storage: three `antm2c_10m_part*` event files,
`AntM2C_image.tar.gz`, and local
BERT/ChineseCLIP checkpoints. No source file or archive is copied into Git or modified.

The public raw replay streams CSV rows, assigns `source-file:source-row` event IDs, applies the exact
midnight cutoffs, builds a first-appearance item index, and runs the causal history scan. A real
30,000-event tracer (10,000 rows from each part) covered train/validation/test with
22,739/1,967/5,294 events, 20,337 users, 6,972 items, and history lengths through five; leakage
validation passed. A second real item-store tracer encoded 25 distinct titles and verified finite
shared target/history gathers with an all-zero padding row.

The approved BERT checkpoint was loaded on V100 and replayed against eight existing
`bill_embeddings_train.npy` rows: maximum/mean absolute deltas were `8.389354e-6` and
`2.527670e-7`. ChineseCLIP loaded on a second V100 and encoded a real 640x640 PNG streamed directly
from the compressed archive into a finite 512-dimensional feature. Both adapters then completed
and resumed checksummed `run_batch_extraction` jobs without reloading their model. Checkpoint and
source identities plus the ignored manifests are retained below
`outputs/data/antm2c-raw-replay-v1/`.

This validates the new raw history/item/extraction boundaries but does not replace the existing
11,190,985-event canonical store with a second full encoding run. Full re-encoding would process
every interaction text field and roughly 84.9K item images; it must be published as a new data
version rather than silently overwriting canonical-v1.
