# Run directory contract

Every training run must own one directory under:

```text
outputs/{experiment_name}/{dataset}/{model}/{run_id}/
```

The `run_id` is generated as:

```text
{UTC timestamp with microseconds}-{10-character config hash}-{8-character random entropy}
```

The timestamp and hash make the run traceable. Random entropy keeps simultaneous runs with the
same configuration distinct, while atomic directory creation (`exist_ok=False`) makes a collision
fail instead of reusing an existing directory.

## Files

The run-context utility creates this initial layout:

```text
{run_id}/
├─ resolved_config.yaml
├─ run_metadata.json
├─ metrics.jsonl
└─ checkpoints/
```

The primary legacy trainer also writes `run.log` and, on successful completion, `summary.json`.
Its existing model-specific checkpoint filename is retained inside the isolated `checkpoints/`
directory until the common training engine is introduced by `TRAIN-001`.

`run_metadata.json` starts with status `running` and records the run/config identity, Git commit
when available, Python/OS and framework versions, command, working directory, seed, requested
CUDA device, data version/fingerprint when supplied, and start time. The trainer changes the
status to `completed` or `failed` and records an end time. YAML and JSON files use atomic
temporary-file replacement.

## Current adoption boundary

`src/trainers/Trainers.py` uses this contract. Legacy tuning and analysis entry points still have
their own output conventions; their migration belongs to `TUNE-001` and `EXP-001`. Until those
tasks are complete, do not treat their output layout as the formal run contract.

Use `--output_root` to override the configured `output_root` for the primary legacy trainer. The
override selects only the root; the trainer always creates the remaining hierarchy and run ID.
