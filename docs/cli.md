# Command-line interface

The public entry point is available as either:

```bash
mmctr --help
python -m mmctr.cli --help
```

Current subcommands:

```text
train             Run isolated canonical or compatibility training
validate-config   Validate and print a resolved training YAML file
list-models       List registered CTR model names
list-quantizers   List registered quantization premodels
list-datasets     List registered dataset names
```

Examples:

```bash
python -m mmctr.cli list-models
python -m mmctr.cli list-quantizers
python -m mmctr.cli validate-config --config config/train.yaml
python -m mmctr.cli train --model-name dnn --dataset-name antm2c --cuda 0
```

Use `--use-local-data` with the local path mechanism documented in
[local-paths.md](local-paths.md). Every train invocation creates an isolated run directory.

The CLI module is dependency-light: help, version, listing, and configuration validation do not
import PyTorch or TensorFlow. The train command applies `--num-threads` to process environment
variables before it imports the training runtime.

The `train` subcommand currently wraps the legacy trainer. Its full Linux/CUDA and real-data path
remains a deferred server validation gate; availability of `--help` is not evidence that a formal
experiment completed.
