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

`config/train.yaml` declares `quantization_artifact_dir`, resolved relative to the
repository root by `TrainingConfig`. The stable layout is:

```text
experiments/quantization/
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
<SERVER_ENV>/bin/python -m trainers.RQ_trainer --dataset-name antm2c --use-local-data
<SERVER_ENV>/bin/python -m trainers.PSRQ_trainer --dataset-name antm2c --cuda 0 --use-local-data
```

RQ L2-normalizes each non-ID item modality before fitting and stores the actual
raw modality dimension in each artifact. PSRQ trains the configured modality and
joint autoencoders with a first batch at least as large as the codebook. Neither
entry point parses arguments or changes thread environment variables at import
time.

RQ/PSRQ structure must match the QARM/MCCA dataset-specific configuration. The
loader verifies level count and codebook size; PSRQ additionally verifies ordered
modalities, every raw feature dimension, latent projection dimension and encoder
hidden dimensions. An artifact from another dataset or modality is rejected.

The historical multi-GPU `Codebook_Tuner.py` still uses frozen legacy premodels
and recommendation classes so existing paper-result reproduction remains
available. Its migration to canonical pretraining plus `TrainingEngine` belongs
to the experiment/tuning consolidation task; it is not a second production
artifact format.
