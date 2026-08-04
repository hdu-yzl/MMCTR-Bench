# Configuration contract

`mmctr.config` provides strict YAML loading, deterministic layer merging, and an immutable typed
schema for runtime-critical training settings.

Configuration precedence is low to high:

```text
training defaults < dataset < model < experiment < explicit CLI overrides
```

`merge_config_layers` recursively merges mappings without mutating its inputs. Lists and scalar
values are replaced as complete values; they are not concatenated implicitly.

The repository has one tracked configuration root:

```text
configs/
├── datasets/catalog.yaml
├── models/catalog.yaml
├── training/default.yaml
├── experiments/*.example.yaml
└── local/paths.example.yaml
```

The former `config/` tree and duplicate `seq_data.yaml` were removed. Sequence and pooled models
resolve the same dataset catalog; history behavior is selected by model capability rather than by
loading a second copy of dataset metadata.

## Training schema

`TrainingConfig` rejects missing and unknown keys, booleans passed as integers, unsupported
optimizers, invalid numeric ranges, and an early-stop patience larger than `max_epochs`. It is a
frozen dataclass, so runtime code cannot silently mutate the validated object. `to_dict()` is the
explicit compatibility boundary for legacy code that still consumes dictionaries.

Relative paths are resolved against the project root containing `pyproject.toml`, never against
the caller's current working directory. Callers may supply an explicit project root when embedding
the library. `checkpoint_dir` stores ordinary training checkpoints, while
`quantization_artifact_dir` is the independent stable root for RQ/PSRQ artifacts consumed across
isolated CTR runs; see [quantization.md](quantization.md).

## YAML safety

`load_yaml_mapping` requires a mapping at the document root and rejects duplicate YAML keys.
Duplicate keys are not accepted with a last-value-wins interpretation because that makes reviewed
configuration differ from the effective run.

The current strict typed surface covers shared training fields. Algorithm-specific model and
dataset fields remain mappings until their dedicated model/data refactor tasks establish regression
evidence. This boundary is deliberate: configuration cleanup must not silently change paper model
formulas or data semantics.

The former `Tuner.yaml` and unconsumed `best_param.yaml` were removed. Historical test-selected
records remain explicitly quarantined in `docs/legacy_tuning_history.yaml` and are not executable
configuration. New tuning output belongs under ignored `outputs/` paths; a reviewed frozen
selection must retain its experiment ID, validation metrics, seeds, and data fingerprint.
