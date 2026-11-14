# Release checklist

Use this checklist for each Linux source release. Record completion, command output, artifact
hashes, and CI links in local release evidence. This public document defines the procedure.

## Source and privacy audit

- Review `git status --short --branch`, the diff, and remote synchronization.
- Confirm `git ls-files` excludes papers, real data, fitted vocabularies, checkpoints, raw local
  results, development progress, caches, credentials, and machine-specific configuration.
- Keep the README reference tables and public dataset/model protocols consistent.
- Check source/configuration text for private paths, tokens, keys, and conflict markers.
- Verify dataset terms and citations without redistributing third-party data.
- Confirm `LICENSE`, `CITATION.cff`, and package metadata agree on the version and license.
- Preserve excluded local files; remove tracked private reports from the index without deleting
  the local copies.

## Linux quality gates

Activate the existing `bm` experiment environment, resolve its absolute interpreter, and keep
caches and temporary files on the project filesystem:

```bash
<BM_PYTHON> -m pip check
<BM_PYTHON> -m ruff format --check src/mmctr tests
<BM_PYTHON> -m ruff check src/mmctr tests
<BM_PYTHON> -m mypy src/mmctr
<BM_PYTHON> -m pytest -q tests/unit tests/smoke
<BM_PYTHON> -m pytest --cov=mmctr --cov-report=term-missing --cov-fail-under=80
<BM_PYTHON> -m build --no-isolation
```

The final line is the absolute-interpreter equivalent of `python -m build --no-isolation`.
Real-data/GPU checks are separate from CPU tests; retain their actual device and dataset
fingerprints in local evidence. AntM2C, MicroLens, TikTok, and EB-NeRD validation must use their
configured canonical data versions.

## Distribution audit

- Record sdist/wheel sizes and SHA-256 hashes locally.
- Inspect archive members for papers, data payloads, fitted vocabularies, checkpoints, local
  configuration, run outputs, progress records, and temporary/build directories.
- Include source, public documentation, portable launchers, configuration templates, and tests.
- Verify each artifact includes exactly one `LICENSE`.
- Install the wheel with `--no-deps --target` into an ignored project-local temporary directory.
- From a repository-external working directory, import `mmctr` and run CLI help and all catalogs.
- Confirm 23 canonical CTR models, four datasets, and two quantizers are available.

## Publication

- Review the staged allowlist and commit the approved source changes.
- Push to the intended branch and verify the remote commit SHA.
- Check the Linux CI result for that exact commit.
- Keep tags, release pages, and package uploads subject to their own publication authorization.
- Verify public installation commands, documentation links, citation metadata, and the absence
  of bundled third-party data or local development artifacts.
