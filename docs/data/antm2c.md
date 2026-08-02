# AntM2C data protocol

This document records the semantic contract before the preprocessing pipeline is rewritten. It is
based on the tracked raw-column list and the current extraction, serialization, and loader code.
No real dataset was read during this audit.

## Feature ownership

| Raw field | Canonical field | Owner | Split sharing | Current evidence |
|---|---|---|---|---|
| `service_entity_seq` | `service_text` | interaction context | forbidden | embedded once per interaction and split |
| `query_entity_seq` | `query_text` | interaction context | forbidden | embedded once per interaction and split |
| `bill_entity_seq` | `bill_text` | interaction context | forbidden | embedded once per interaction and split |
| `log_time` | `time_context` | interaction context | forbidden | event timestamp represented as text today |
| `item_title` | `title_text` | item | allowed | current extractor already deduplicates by item ID |
| image named by `original_item_id` | `image` | item | allowed | current extractor builds one item feature matrix |
| `item_entity_names` | `entity_text` | item candidate | not until audited | name suggests item metadata, but current extractor embeds every interaction |

`item_entity_names` must be audited over all splits. Promotion to item ownership requires every
non-empty value for a given mapped `item_id` to be identical. Conflicting item IDs and missing
values go into the manifest; conflicts require either an interaction-context assignment or an
explicit canonicalization rule approved as a data-version change.

## Interaction identity and split rules

Every raw row receives `event_id = {data_version}:{source_shard}:{source_row}` before filtering or
concatenation. Histories are ordered by `(user_id, timestamp, event_id)` and built in one scan. Only
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

Interaction rows store IDs, timestamp, label, scene, history item indices, and the four interaction
context embeddings. The item feature store owns title, audited entity text, and image features.
Target and history modalities gather from the same item-indexed arrays. No interaction row stores a
4608-dimensional concatenation or repeats item-owned vectors.

Each split uses its own service/query/bill/time embeddings. The current serializer always indexes
variables suffixed `_train` inside the train/val/test loop; this is a confirmed bug to remove in
`ANT-005`. Final storage format remains undecided until `ANT-006` benchmarks it on the server.

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
from `ItemFeatureStore`. The `named-npy-candidate-v1` writer is intentionally a benchmark candidate:
`ANT-006` still decides whether it or TFRecord/a different layered format becomes authoritative.
