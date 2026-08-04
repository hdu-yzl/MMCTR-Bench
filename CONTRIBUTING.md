# Contributing to MMCTR-Bench

Thank you for improving MMCTR-Bench. The repository is undergoing a staged refactor, so changes
must preserve research behavior while the engineering interfaces are replaced incrementally.

## Before making changes

1. Open an issue or agree on a clearly scoped task before implementation.
2. Confirm its dependencies, expected files, owner, and validation evidence.
3. Record the rationale before changing package layout, public interfaces, configuration hierarchy,
   metric semantics, data splits, or experiment protocols.

Do not combine unrelated cleanup with a task. Public metadata, registries, schemas, and root
configuration files are high-conflict areas and require explicit coordination.

## Environment boundary

Linux is the authority for installation, framework compatibility, CUDA, real-data training,
performance, and formal regression. Replace documentation placeholders with real absolute paths
in execution records:

```bash
<SERVER_ENV>/bin/python -m pytest
<SERVER_ENV>/bin/python -m build
```

The configured Windows workspace interpreter is:

```powershell
& 'd:\anaconda\envs\py312\python.exe' --version
```

Windows checks may validate UTF-8, AST, TOML, Markdown, and dependency-light units. They do not
prove Linux or CUDA compatibility. Do not install or upgrade shared-environment dependencies
without maintainer authorization, and do not create an unrequested virtual environment.

## Research integrity

- Use training data only for fitting.
- Use validation data for early stopping, model selection, and hyperparameter tuning.
- Evaluate test data only after the configuration is frozen.
- Never write test metrics into best-parameter configuration.
- Do not silently change formulas, losses, modalities, masks, split rules, sampling, or metrics.
- Claims that behavior is unchanged require fixed-config, fixed-seed regression evidence.
- Preserve every seed result and aggregate with mean, standard deviation, and valid-run count.

Any result affected by test-set selection must be marked contaminated and regenerated under a
valid protocol.

## Code and data rules

- New public Python code must be typed and remain compatible with the declared Python baseline.
- Use `mmctr...` absolute imports once the target package exists; do not add new `sys.path`
  manipulation.
- Library imports must not parse CLI arguments, launch work, read large files, or mutate global
  process state.
- Models must not create optimizers, select devices, write files, or mutate input dictionaries in
  `forward`.
- Use explicit tensor dimensions; avoid unconstrained `squeeze()` on batch tensors.
- Use `pathlib.Path` for new path logic and obtain output roots from config or the runner.
- Do not commit raw/processed data, checkpoints, logs, caches, local path configuration, secrets,
  private samples, or personal absolute paths.
- Do not use bare `except:` or silently skip malformed data.

## Validation

Run the smallest checks that prove the task, then the applicable stage gate. The intended Linux
gate is:

```bash
<SERVER_ENV>/bin/python -m ruff format --check .
<SERVER_ENV>/bin/python -m ruff check .
<SERVER_ENV>/bin/python -m pytest tests/unit tests/smoke
<SERVER_ENV>/bin/python -m mypy src/mmctr
<SERVER_ENV>/bin/python -m pytest --cov=mmctr --cov-report=term-missing
<SERVER_ENV>/bin/python -m build
```

If a tool or external dependency is unavailable, record the exact missing condition. Do not
report a skipped command as passing.

## Commit and review checklist

- The change matches one registered task and does not revert unrelated work.
- Generated artifacts and private paths are absent from both the diff and staged files.
- Tests cover new behavior and relevant boundary cases.
- Documentation and migration notes match the implementation.
- Public-interface changes include compatibility or migration guidance.
- Experimental changes include configuration, seed, data-version, and metric evidence.
- The issue or review description records final status, validation commands/results, date, and
  follow-up risks.

Use this handoff structure in the task notes or review description:

```text
Task ID:
Summary:
Changed files:
Behavior changes:
Compatibility/migration:
Validation commands:
Validation results:
Metric comparison (if applicable):
Remaining risks:
Follow-up tasks:
```

## License status

The repository-owned source code, configuration, and documentation are licensed under Apache-2.0.
By submitting a contribution, you agree that it may be distributed under that license. Do not add
third-party data, checkpoints, incompatible code, or a copyright/NOTICE statement whose ownership
has not been verified. Dataset and checkpoint terms remain separate from the software license.
