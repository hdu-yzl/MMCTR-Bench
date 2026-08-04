# TikTok raw inputs

The user must obtain this TikTok layout through the official
[HKUDS/MMSSL repository](https://github.com/HKUDS/MMSSL) associated with Wei et al., *Multi-Modal
Self-Supervised Learning for Recommendation* (WWW 2023), or another authorized route. No
independent redistribution permission is supplied by MMCTR-Bench.

Place the official split files and upstream features exactly as follows:

```text
data/raw/tiktok/
├── train.json
├── val.json
├── test.json
├── text_feat.npy
├── image_feat.npy
└── audio_feat.npy
```

The NumPy files are already-extracted modality features, not raw media. The canonical processor
rebuilds causal CTR samples and storage layout; it does not rerun text, image, or audio encoders.

Generate the canonical store with `prepare_tiktok(Path("data/raw/tiktok"),
Path("data/processed/tiktok/canonical-v1"))` from
`mmctr.data.datasets.tiktok.preprocessing`.
