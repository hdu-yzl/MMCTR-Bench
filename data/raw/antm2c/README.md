# AntM2C raw inputs

The user must obtain AntM2C from the original ATEC/provider flow. This repository does not download
or redistribute it. Keep the provider URL, access date, terms, archive name, and SHA-256 values in
an ignored local provenance record.

Recommended placement:

```text
data/raw/antm2c/
├── events/
│   ├── antm2c_10m_part0
│   ├── antm2c_10m_part1
│   └── antm2c_10m_part2
├── images/
│   └── AntM2C_image.tar.gz
├── encoders/
│   ├── bert/          # provider-approved local BERT checkpoint
│   └── chinese-clip/  # provider-approved local ChineseCLIP checkpoint
└── legacy-v1/         # optional: 28 TFRecords + text_feature.npy + image_feature.npy
```

The raw-event/history and streaming encoder implementations live in
`mmctr.data.datasets.antm2c.raw` and `mmctr.data.datasets.antm2c.encoders`. They validate the raw
boundary but do not silently overwrite the published `canonical-v1` store. A complete raw
re-encoding must use a new version such as `canonical-v2`.

The optional `legacy-v1/` layout is only for reproducing the existing TFRecord-to-named-array
conversion:

```bash
<SERVER_ENV>/bin/python -m mmctr.data.datasets.antm2c.canonical \
  --source-dir data/raw/antm2c/legacy-v1 \
  --output-dir data/processed/antm2c/canonical-v1
```

Do not extract the 24+ GB image archive into Git. The image adapter can stream selected members
directly from the archive.
