# Raw data placement

This directory is the filesystem interface for inputs obtained by each user from the original
providers. Git tracks only these placement documents; every dataset file, media archive, extracted
feature, credential, and encoder checkpoint below this directory is ignored.

Download only datasets whose provider terms permit your intended use, then place or symlink them
under the matching directory:

```text
data/raw/
├── antm2c/    # user supplies events, image archive, and optional local encoders
├── microlens/ # user supplies the two provider parquet files
└── tiktok/    # user supplies official splits and upstream modality arrays
```

Exact filenames and provenance requirements are documented in each dataset README. Generated
canonical arrays belong under `data/processed/`, never under `data/raw/`.
