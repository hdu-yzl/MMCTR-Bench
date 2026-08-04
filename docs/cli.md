# Command-line interface

The public entry point is available as either:

```bash
mmctr --help
python -m mmctr.cli --help
```

For repository checkouts, `scripts/train_model.sh`, `scripts/train_all_models.sh`, and
`scripts/pretrain_quantizers.sh` provide validated one-click wrappers around these canonical
module entry points. They contain no alternate model or training implementation; see the root
README and [training.md](training.md).

Current subcommands:

```text
train             Run isolated canonical training
validate-config   Validate and print a resolved training YAML file
list-models       List registered CTR model names
list-quantizers   List registered quantization premodels
list-datasets     List registered dataset names
plan-*-study      Materialize versioned analysis task matrices
plot-results      Render figures from standard completed results
```

Examples:

```bash
python -m mmctr.cli list-models
python -m mmctr.cli list-quantizers
python -m mmctr.cli validate-config --config configs/training/default.yaml
python -m mmctr.cli train --model-name dnn --dataset-name antm2c --cuda 0
```

Use `--use-local-data` with the local path mechanism documented in
[local-paths.md](local-paths.md). Every train invocation creates an isolated run directory.

The CLI module is dependency-light: help, version, listing, and configuration validation do not
import PyTorch or TensorFlow. The train command applies `--num-threads` to process environment
variables before it imports the training runtime.

The `train` subcommand composes the canonical dataset loader, registry model, training engine, and
isolated run context. Real AntM2C/MicroLens/TikTok CPU steps and a V100 AntM2C step are part of the
recorded server evidence; availability of `--help` alone is still not evidence that a new formal
experiment completed.
