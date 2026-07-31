# Local path configuration

Public dataset YAML files contain portable project-relative defaults. Machine-specific absolute
paths must not be committed.

To configure a machine, copy the tracked example:

```bash
cp configs/local/paths.example.yaml configs/local/paths.yaml
```

Fill only the datasets used on that machine and optionally `output_root`. The copied `paths.yaml`
is ignored by Git. Dataset paths must be absolute existing directories; `output_root` must be
absolute but may be created by the run-context layer.

Environment variables override the local file:

| Variable | Meaning |
|---|---|
| `MMCTR_ANTM2C_DATA_DIR` | AntM2C processed dataset directory |
| `MMCTR_MICROLENS_DATA_DIR` | MicroLens processed dataset directory |
| `MMCTR_TIKTOK_DATA_DIR` | TikTok processed dataset directory |
| `MMCTR_OUTPUT_ROOT` | Root for isolated run outputs |

The primary legacy trainer reads these overrides only when `--use_local_data 1` is passed. If the
selected dataset has no local path, it fails with an explicit configuration error. Without that
flag, canonical dataset paths are resolved against the repository root and never against the
current working directory.

The example deliberately contains `null` values. They are placeholders, not paths, and do not
grant permission to download or redistribute any dataset.
