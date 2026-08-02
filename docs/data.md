# Data contracts

New training code consumes `mmctr.core.Batch` objects from a loader implementing
`mmctr.data.DataLoaderProtocol`. It does not call TensorFlow or interpret positional tuples.
`Batch.context_features` keeps interaction-owned context separate from target item features;
models must opt in to those named fields rather than relying on concatenation offsets.

`CanonicalDataLoader` is the migration boundary for the AntM2C, MicroLens, and TikTok legacy
loaders. It selects the old sequence or pooled-compatible iterator, separates combined user/item
IDs when possible, creates the history mask from the manifest padding ID, and adds dataset/split
provenance to batch metadata. This adapter does not change the underlying TFRecord schema.

Every processed dataset is described by a `DatasetManifest` containing:

- schema, dataset, and data versions;
- storage format and sequence length;
- padding/OOV IDs and explicit ID offsets;
- named feature dimensions;
- per-split counts and optional SHA-256 digests;
- a stable manifest fingerprint used in run metadata.

Manifests synthesized from current YAML are marked `legacy-config-only`. They are compatibility
metadata, not proof that source files or split statistics were audited.

AntM2C now also has a restartable batch-extraction boundary and a candidate named-array store.
Extraction loads an encoder once per unfinished job, atomically checkpoints each shard, validates
shape/finiteness/checksums, records missing keys, and resumes only when source fingerprints match.
The candidate store writes every interaction context and item modality as an independently named
array. Its loader memory-maps those arrays, gathers target/history modalities from one item table,
preserves padding while applying explicit ID offsets, and yields canonical batches. The format is
an implementation candidate for server benchmarking, not the final `ANT-006` storage decision.
