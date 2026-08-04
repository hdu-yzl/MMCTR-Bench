# Release checklist

This checklist is the Linux release gate for MMCTR-Bench. A successful package rehearsal is not a
release authorization: every blocking item must also be closed by the maintainer.

## Blocking decisions and inputs

- [x] `OSS-001`: the maintainer selected Apache-2.0 and the matching `LICENSE` file is
  present, reviewed, and included in both source and wheel distributions.
- [x] `OSS-002`: dataset acquisition/terms and model references are documented without distributing
  third-party data.
- [x] AntM2C source events, raw text/images, and approved encoder checkpoint have been restored so
  `ANT-002`, `ANT-003`, and `ANT-004` can be replayed from original inputs.
- [x] Maintainer approved version `0.1.0`, the consolidated public tree, and its changelog by
  explicitly requesting that the completed latest revision be uploaded to GitHub.

The local source/build gates below are complete. Creating a signed tag or publishing artifacts is a
separate maintainer action from pushing the requested source revision.

## Source and privacy audit

- [x] `git status --short --branch` has been reviewed; all cumulative refactor and directory
  consolidation changes are intentional.
- [x] `git ls-files` contains no raw/processed dataset arrays, checkpoints, credentials, local path
  configuration, caches, run outputs, or build products.
- [x] Release source/config text contains no personal absolute paths, access tokens, passwords,
  private keys, or
  conflict markers.
- [x] every dataset/provider link and paper mapping in `data/README.md` and `docs/references.md` has
  been reviewed at release time.
- [x] `CITATION.cff` and `pyproject.toml` agree on version `0.1.0`; the CFF release date is
  `2026-08-04`.

## Linux quality gates

Run every command with the absolute interpreter resolved from the maintained `bm` environment and
project-local cache/temp directories:

The final line is the absolute-interpreter equivalent of `python -m build --no-isolation`.

```bash
<BM_PYTHON> -m pip check
<BM_PYTHON> -m ruff format --check src/mmctr tests
<BM_PYTHON> -m ruff check src/mmctr tests
<BM_PYTHON> -m mypy src/mmctr
<BM_PYTHON> -m pytest -q tests/unit tests/smoke
<BM_PYTHON> -m pytest --cov=mmctr --cov-report=term-missing --cov-fail-under=80
<BM_PYTHON> -m build --no-isolation
```

- [x] all commands pass on Linux and the exact counts/tool versions are recorded in
  `REFACTORING_PLAN.md`.
- [x] required real-data CPU/CUDA smoke evidence still matches the canonical dataset fingerprints.

## Distribution rehearsal

- [x] Record the source archive and wheel byte sizes and SHA-256 digests.
- [x] Inspect both archive member lists; the wheel contains no data tree, while the sdist may
  contain only tracked data placement README files and small canonical manifests—never payload
  arrays. Neither artifact may contain outputs, checkpoints, local config, the deleted
  `models`/`scripts` packages, or `mmctr/models/compat.py`.
- [x] Install the wheel with `--no-deps --target` into an ignored project-local temporary directory.
- [x] From a repository-external working directory, import `mmctr`, print its installed path/version,
  run `python -m mmctr.cli --help`, `list-models`, `list-datasets`, and `list-quantizers`.
- [x] Confirm the registry exposes exactly 23 canonical CTR models, 3 datasets, and 2 quantizers.
- [x] Remove only the scoped temporary install/build directories after recording evidence.

### Rehearsal evidence: 2026-08-04

- interpreter: the absolute `bm` interpreter recorded in the private server evidence
  (Python 3.8.20);
- `pip check`, Ruff format (142 files), Ruff lint, and mypy (86 source files): passed;
- unit + smoke: 198 passed; full coverage gate: 213 passed, 84.90%;
- Apache-2.0 metadata contract: 5 passed; CFF YAML parsing passed;
- final wheel/sdist sizes, member counts, and SHA-256 values are recorded in the release-task
  evidence in `REFACTORING_PLAN.md`;
- each artifact contains exactly one `LICENSE`; installed metadata exposes the Apache classifier and
  license text (the plan remains excluded from the sdist to avoid a self-referential hash);
- archive forbidden-member audit: zero payload findings; sdist contains citation, contribution,
  data terms/placement README files, reference, release, config, and test material through
  `MANIFEST.in`;
- repository-external wheel import resolved below `.tmp/release-install`, reported version `0.1.0`,
  canonical Trainer/RQ/PSRQ help passed, and all CLI catalog commands returned 23/3/2 entries.

## Publication

- [ ] `REL-001`: close the public-source release task after the release commit is pushed and its CI
  result is recorded; artifact publication remains a separate maintainer action.
- [ ] Create a clean release commit and signed/annotated tag only after all blockers are closed.
- [ ] Re-run CI from the release commit and retain the run URL and commit SHA.
- [ ] Publish artifacts only after comparing their SHA-256 values with the locally audited files.
- [ ] Verify the public release page, installation command, citation metadata, documentation links,
  and absence of bundled third-party data.
