# Local datasets

MMCTR-Bench does not redistribute third-party datasets, raw media, extracted features, or generated
canonical arrays. Obtain data from the original provider, review its current terms, and place it in
an ignored local path. A paper license, a code-repository license, or a public download URL is not a
data redistribution license.

## Source and access matrix

| Dataset | Original source and citation | Acquisition | Verified data-license status | Repository policy |
|---|---|---|---|---|
| AntM2C | [ATEC homepage](https://www.atecup.com/home); [Huan et al., *AntM²C: A Large Scale Dataset For Multi-Scenario Multi-Modal CTR Prediction*](https://arxiv.org/abs/2308.16437) | Request/download through the provider's current ATEC flow. The repository does not provide a mirror. | No independent, machine-readable dataset license was verified on 2026-08-04. The Creative Commons link on arXiv applies to the paper, not automatically to its industrial data. | Keep every source event, image, text field, credential, and generated array out of Git. Confirm provider terms before use or sharing. |
| MicroLens | [official author repository](https://github.com/westlake-repl/MicroLens); [Ni et al., *A Content-Driven Micro-Video Recommendation Dataset at Scale*](https://arxiv.org/abs/2309.15379) | Use the download portal linked by the authors. Full-data access may require contacting them. | The official repository says privately modified datasets must not be offered as secondary downloads. No broader redistribution grant was inferred. | Publish preprocessing code and fingerprints only. Do not mirror original or modified MicroLens data. |
| TikTok | [InvRL official implementation](https://github.com/nickwzk/InvRL), whose `tiktok/` directory is the identifiable upstream location for this processed layout; [Du et al., *Invariant Representation Learning for Multimedia Recommendation*](https://doi.org/10.1145/3503161.3548405) | Obtain the data through an authorized upstream route and record its provenance locally. The InvRL README does not document a data download or terms. | No independent data license or authoritative redistribution permission was verified. The precise provider lineage of the current local arrays still requires maintainer confirmation. | Treat the current files as local research inputs, never as redistributable repository assets. |

The TikTok DOI above is intentionally not used as a licensing statement. If its landing page changes
or the identifier cannot be resolved, use the paper title and the official InvRL repository when
requesting provenance from the maintainers. Dataset and model citations are collected in
[`docs/references.md`](../docs/references.md).

## Local placement

Raw and generated datasets are intentionally ignored by Git. On the Linux experiment server, the
canonical outputs produced by the current preprocessing code live at:

- user-downloaded inputs: `data/raw/{antm2c,microlens,tiktok}/`;
- `data/processed/microlens/canonical-v1/`
- `data/processed/tiktok/canonical-v1/`
- `data/processed/antm2c/canonical-v1/`
- `data/processed/antm2c/benchmark-v1/` (format benchmark only; not a full training dataset)

Each canonical directory contains `manifest.json`, a shared item-feature store, and named split
arrays. AntM2C further shards each split and uses `canonical-v1.incomplete/` only for restart state.
Exact filenames that users must download or obtain are specified in
[`data/raw/README.md`](raw/README.md) and the three dataset-specific README files below it.
See [`docs/data.md`](../docs/data.md) for split, negative-sampling, ID mapping, and fingerprint
details. Configure machine-specific roots through ignored `configs/local/paths.yaml`, the documented
`MMCTR_*_DATA_DIR` environment variables, or explicit CLI overrides. Do not copy private source data
or generated arrays into Git history.

The current AntM2C full publication fingerprint is
`ece3afcc876853caadd4e277421c938a59426e2e003577b1fd02b90e2d3b2aca`; its tracked manifest is
evidence only, while the 173,233,549,382-byte array store remains ignored.

The current MicroLens and TikTok canonical stores were regenerated from the provider/reference
interaction files and their already-extracted modality features. MicroLens uses
`train.parquet` plus BERT/CLIP columns in `item_feature.parquet`; TikTok uses official split JSON
plus text/image/audio arrays. Their manifest source hashes exactly match the read-only inputs, but
neither pipeline claims to have rerun encoders from original media.

## Provenance record required for a local run

For every source snapshot, keep an ignored record containing:

- provider URL and access date;
- provider-issued version or archive name, if any;
- exact source-file SHA-256 values;
- terms/license text or a link captured at access time;
- every transformation command and canonical manifest fingerprint.

If any permission or lineage field is unknown, mark it unknown and do not redistribute the data.
