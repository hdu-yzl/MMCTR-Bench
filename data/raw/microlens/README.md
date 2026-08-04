# MicroLens raw inputs

The user must obtain MicroLens through the official author repository/download flow and comply with
its current terms. The repository does not provide a mirror.

Place the provider files exactly as follows:

```text
data/raw/microlens/
├── train.parquet
└── item_feature.parquet
```

`train.parquet` supplies interactions. `item_feature.parquet` already contains upstream BERT text
and CLIP image embeddings; MMCTR-Bench does not reconstruct those features from raw videos.

Generate the canonical store with `prepare_microlens(Path("data/raw/microlens"),
Path("data/processed/microlens/canonical-v1"))` from
`mmctr.data.datasets.microlens.preprocessing`.
