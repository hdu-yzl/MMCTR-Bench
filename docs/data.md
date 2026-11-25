# Datasets

Obtain data from the providers below and see the [input layout](../data/raw/README.md).
The benchmark does not redistribute data. Each processed dataset has a manifest with its
split counts, dimensions, and fingerprint.

| Dataset | Canonical directory | Train / validation / test rows |
|---|---|---:|
| [AntM2C](#antm2c) | `data/processed/antm2c/canonical-v1/` | 8,536,558 / 721,500 / 1,932,927 |
| MicroLens | `data/processed/microlens/canonical-v1/` | 2,880,000 / 360,000 / 360,000 |
| [TikTok](#tiktok-canonical-v1) | `data/processed/tiktok/canonical-v1/` | 357,246 / 18,306 / 36,780 |
| [EB-NeRD](#eb-nerd) | `data/processed/ebnerd/canonical-v4/` | 2,325,624 / 260,123 / 292,271 |

## AntM2C

[ATEC dataset portal](https://www.atecup.cn/OfficalDataSet) · Huan et al., *Exploring
Multi-Scenario Multi-Modal CTR Prediction with a Large-Scale Dataset*, SIGIR 2024.

Titles and images are shared by item; service, query, bill, entity, and time features are
per interaction. A history includes only earlier positive events. Train runs through
2023-08-03 00:00:00, validation through 2023-08-05 00:00:00, and test covers later events.

Set `ANTM2C_TFRECORD_DIR` to the directory with the split TFRecords and the shared
`text_feature.npy` and `image_feature.npy` item arrays. The output uses about 161 GiB:

```bash
"$MMCTR_PYTHON" -m mmctr.data.datasets.antm2c.canonical \
  --source-dir "$ANTM2C_TFRECORD_DIR" --output-dir data/processed/antm2c/canonical-v1
```

See [AntM2C source files](../data/raw/antm2c/README.md) for raw events and encoder inputs.

## MicroLens

[Official repository](https://github.com/westlake-repl/MicroLens) · Ni et al.,
*A Content-Driven Micro-Video Recommendation Dataset at Scale*, CIKM 2025.

`prepare_microlens(raw_dir, output_dir)` reads `train.parquet` and
`item_feature.parquet`. It keeps up to five source history items and creates a deterministic
80/10/10 interaction split (seed 42). Target and history items share an item store with
upstream BERT/CLIP embeddings.

## TikTok canonical-v1

[HKUDS/MMSSL](https://github.com/HKUDS/MMSSL) · Wei et al.,
*Multi-Modal Self-Supervised Learning for Recommendation*, WWW 2023.

`prepare_tiktok(raw_dir, output_dir)` uses the official train, validation and test JSON
files and upstream text, image, and audio arrays. Histories follow source list order:
validation inherits train history, and test inherits train and validation history. The
source lacks reliable timestamps, so this order does not establish temporal causality.
Each positive has five deterministic sampled negatives, excluding items observed by the user
in any supplied split.

The processed data contains **412,332 CTR rows**, **9,308 users**, and **6,710 target items**.
User and item counts below are distinct IDs in the emitted rows.

| Split | CTR rows | Positives | Negatives | Users | Target items |
|---|---:|---:|---:|---:|---:|
| Train | 357,246 | 59,541 | 297,705 | 9,298 | 6,710 |
| Validation | 18,306 | 3,051 | 15,255 | 3,051 | 6,176 |
| Test | 36,780 | 6,130 | 30,650 | 6,130 | 6,660 |

## EB-NeRD

The benchmark uses [EB-NeRD small](https://recsys.eb.dk/dataset/) at
`data/processed/ebnerd/canonical-v4/`, with five history items, 16-dimensional ID
embeddings, and article text (768 dimensions) and image (1,024 dimensions) embeddings.
See [the input layout](../data/raw/ebnerd/README.md) for the seven Parquet files.

The official training impressions are sorted by timestamp and impression ID and split locally
at the 90% timestamp boundary, keeping timestamp ties together. Local test samples **10% of
official validation impressions** uniformly without replacement (seed 2025; 24,465 impressions).
This is a labeled holdout; it does not use the official challenge test.

Every displayed candidate contributes one binary CTR row, labeled by membership in the clicked
list. All candidates in an impression stay together, with no added negatives. Histories use the
provider's fixed clicks from before the source split, keep the last five items, and left-pad
with zeros. Missing embeddings are zero-filled without dropping rows or valid history positions.
The benchmark reports candidate-level ROC-AUC and LogLoss.

The dataset has **58 ID fields**: user and article IDs plus 19 categorical and 37 bucketed
article, history, and impression fields. Category vocabularies are fitted on training inputs
only; unseen categories use field-specific OOV IDs. Numeric fields use fixed bucket edges.
Missing categories, unknown categories, and padding have distinct IDs.

The loader exposes the additional fields as `Batch.user_features["attribute_ids"]` with shape
`[B, 56, 16]` (up to 16 token slots per field). Tokens use the shared ID embedding table and are
mean-pooled within each field, ignoring padding. Pooled models use the existing target ID
projection; sequence models use the existing user ID projection and retain separate target and
history article IDs.

### Prepare EB-NeRD

Activate the project environment and set `MMCTR_PYTHON` to its absolute interpreter path.
Prepare and verify the current dataset with one command:

```bash
"$MMCTR_PYTHON" -m mmctr.data.datasets.ebnerd --prepare \
  --raw-dir data/raw/ebnerd \
  --output-dir data/processed/ebnerd/canonical-v4 \
  --work-dir .tmp/ebnerd-prepare \
  --sequence-length 5 --test-fraction 0.1 --sampling-seed 2025 \
  --config-output .tmp/ebnerd-prepare/catalog.yaml
```

The command reuses the existing impression construction, test sampling, attribute preparation
and ID encoding stages, then validates the final output. Intermediate files stay under
`--work-dir`; rerunning with the same inputs and parameters reuses completed stages after
validation. Use a different work/output directory when changing the preprocessing settings.

For diagnosis, the individual `--sample-test-from`, `--enrich-from`, `--ids-from` and
`--verify-only` modes remain available. These modes are mutually exclusive with `--prepare`.

Use the tracked catalog for the benchmark dataset. For a custom build, pass the generated
catalog with `--dataset-config`; it records the actual dimensions, ID ranges, and fingerprint.
Keep generated arrays, mappings, and fitted vocabularies local.

### Adding a dataset loader

A loader exposes `dataset_name`, an audited `DatasetManifest`, and `iter_batches(split)`
yielding `mmctr.core.Batch` objects with explicit history masks. Quantization also uses
`get_multi_modal()` to read shared item features. Register the loader in the dataset registry.
The old `get_data` / `get_data_seq`, tuple adapters and `DatasetManifest.from_config` are
removed; construct the manifest from the processed data and return `Batch` directly.
