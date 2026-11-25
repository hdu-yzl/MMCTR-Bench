# Dataset-specific model parameters

Each dataset has one directory: `antm2c/`, `microlens/`, `tiktok/`, and `ebnerd/`.
Each directory contains the 23 registered CTR models and the `rq` pretrainer.
For example, `tiktok/dmf.yaml` is the authoritative parameter file for TikTok DMF.

Each file contains:

- `model`: the model parameters for this dataset, without nested dataset overrides.
- `training`: learning rate, weight decay, batch size, optimizer, epochs, and patience.
- `pretraining`: independent PSRQ quantizer training settings (PSRQ files only).
- `selection`: validation-selection provenance, or `{}` for configured defaults.
- Dataset/model identity and schema version; selected parameters also retain the data fingerprint.

Model files are the sole source of CTR training hyperparameters. Explicit Trainer
and tuning-candidate overrides take precedence. There is no shared training YAML.
Run-local seed/device defaults are 2025/GPU 1 when an entry point does not supply them;
wrapper scripts can set their own defaults. Output defaults to `outputs/`, quantizer
artifacts to `outputs/quantization/artifacts/`. These paths resolve against the project
root, and run-specific checkpoint/log directories are allocated automatically.

RQ reads its `model` and `training` mappings here. PSRQ quantizer pretraining reads the
same file's `model` and independent `pretraining` mapping. CTR tuning preserves this
pretraining mapping when writing the winner. PSRQ pretraining uses `quantizer_dropout`,
independently of downstream CTR `dropout`.

## Tune and write back

Activate the existing environment, then select the pairs and budget:

```bash
scripts/tune.sh --datasets tiktok --models dmf,simcen \
  --max-trials 20 --max-gpu-hours 4 --cuda 0

scripts/tune.sh --datasets all --models dnn,dcn \
  --max-trials 10 --max-hours 2 --cuda 0 --dry-run
```

Budgets apply **per dataset/model pair**. Searches run sequentially on the selected device.
`--models paper` selects the 21 paper models; `--models all` selects all 23 CTR models.
QARM and PSRQ require existing compatible quantizer artifacts. The script does not fit them.

`--max-trials` includes failed attempts. `--max-gpu-hours` limits allocated GPU trial time;
`--max-hours` limits trial time on either CPU or GPU. The first reached limit stops the
search. Checks occur at batch boundaries, so in-flight work may overrun a time limit.
Final evaluation is outside these search budgets. Defaults are 50 attempts / 50 GPU-hours.

Successful tuning selects the highest validation AUC and atomically overwrites only the
corresponding model YAML after the frozen winner's final evaluation succeeds. Test metrics
are never used to choose parameters or copied into this file. Failed searches leave the
existing file unchanged. `--no-write-back` keeps the search evidence without updating it.
History, timing, frozen selection and final-test results remain under `outputs/tuning/`.

New searches start from the current file: sampled parameters are overridden by each trial,
and unsampled settings are retained. Ordinary training, local diagnostics and batch training
read the same parameters automatically. Finish tuning before launching a batch
so its models all use the intended saved settings.

## Migration

The old global model catalog and separate `configs/tuned` directory have been merged here.
Existing effective settings and the three recorded TikTok selections were preserved.
Use `--model-config-dir` with `mmctr train`, `mmctr tune`, or `scripts/tune.sh` to select
another complete parameter root. The old `--tuned-config-dir`, `--ignore-tuned-config`
and diagnostic `--no-tuned-config` options have been removed. To use an earlier parameter
set, restore its model YAML or supply an explicit parameter root.
