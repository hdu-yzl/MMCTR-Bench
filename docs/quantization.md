# Quantization artifacts and pretraining

Quantization is a separate lifecycle from CTR training. RQ and PSRQ learn item
representation codebooks; QARM and MCCA consume those artifacts. The production
boundary is:

1. fit a premodel from the dataset item feature table;
2. save one versioned artifact under `quantization_artifact_dir`;
3. load and validate artifacts in the model composition layer;
4. inject the loaded module into a pure canonical CTR model;
5. save the normal CTR checkpoint in its isolated run directory.

`MODEL_REGISTRY` contains CTR predictors only. `QUANTIZER_REGISTRY` independently
registers `rq` and `psrq`; `mmctr list-models` and `mmctr list-quantizers` therefore
cannot accidentally route a pretraining objective through the binary CTR engine.

## Artifact layout

`configs/training/default.yaml` declares `quantization_artifact_dir`, resolved relative to the
repository root by `TrainingConfig`. The stable layout is:

```text
outputs/quantization/artifacts/
├── rq/<dataset>/<modality>.npz
└── psrq/<dataset>/model.npz
```

This directory contains generated research artifacts and remains outside Git.
Changing the normal per-run `checkpoint_dir` does not change where pretrained
quantizers are found.

Every artifact is an NPZ with `allow_pickle=False`. Its JSON manifest records the
format/version, artifact kind, architecture and modality metadata, plus shape,
dtype and SHA-256 for every array. Writes use a same-directory temporary file and
atomic replacement. Load rejects missing arrays, extra arrays, wrong versions,
kind mismatches, shape/dtype changes and checksum failures before constructing a
model.

## Pretraining entry points

On the authoritative Linux server, activate `bm`, record `command -v python`, and
use that absolute interpreter for these modules:

```bash
scripts/pretrain_quantizers.sh --dataset antm2c --gpu 0 --dry-run
scripts/pretrain_quantizers.sh --dataset antm2c --gpu 0

# Equivalent canonical module entry points:
<SERVER_ENV>/bin/python -m mmctr.quantization.rq_entrypoint --dataset-name antm2c --use-local-data
<SERVER_ENV>/bin/python -m mmctr.quantization.psrq_entrypoint --dataset-name antm2c --cuda 0 --use-local-data
```

After both artifact families exist, `scripts/train_all_models.sh --include-quantized` adds QARM and
MCCA to the normal multi-GPU model queue. Without this explicit opt-in, batch training excludes
them so a missing or incompatible artifact cannot fail an otherwise standard model sweep.

RQ L2-normalizes each non-ID item modality before fitting and stores the actual
raw modality dimension in each artifact. PSRQ trains the configured modality and
joint autoencoders with a first batch at least as large as the codebook. Neither
entry point parses arguments or changes thread environment variables at import
time.

RQ/PSRQ structure must match the QARM/MCCA dataset-specific configuration. The
loader verifies level count and codebook size; PSRQ additionally verifies ordered
modalities, every raw feature dimension, latent projection dimension and encoder
hidden dimensions. An artifact from another dataset or modality is rejected.

The former multi-GPU codebook tuner and frozen premodels have been removed. Use the canonical
RQ/PSRQ entry points above to create validated artifacts, then use the validation-only experiment
tuner for QARM/MCCA recommendation configuration selection.

## Real artifact validation

The Linux server gate generated all three TikTok RQ artifacts with the production `3 × 1024`
structure over the complete 6,711-row item store, and trained the production `3 × 256` PSRQ for
five epochs on V100 (`final_loss=8.485107`). Strict disk reload then composed QARM and MCCA and ran
one real 32-event TikTok forward/backward/Adam step for each. The ignored local report at
`outputs/quantization/tiktok-real-validation.json` records the dataset fingerprint, exact artifact
hashes/sizes, losses and parameter counts; the artifacts themselves remain outside Git.
