# Cross-dataset diagnostic studies

The diagnostic executor supports cold-start, alignment, modality robustness, and fusion studies
on canonical datasets. A frozen baseline manifest supplies dataset versions, model/training
configurations, seed-specific main-table metrics, and input fingerprints. Each task records its
intervention and implementation settings in a versioned matrix.

## Execution and comparisons

Diagnostic matrices use seed 2025 per configuration. Training restores the validation-selected
weights from memory without saving `.pt` checkpoints. Every completed task writes its metrics,
resolved settings, baseline identity, and comparison fields to an isolated result directory.
Single-seed comparisons do not establish statistical significance.

For alignment, robustness, and fusion, the comparator is the same dataset/model/seed main-table
result. Cold-start uses the full-test evaluation of the same selected weights and OOV policy.
Comparison fields include `test_auc`, `baseline_test_auc`, signed `auc_delta`,
`auc_absolute_difference`, and
`relative_change_percent = 100 * (test_auc - baseline_test_auc) / baseline_test_auc`.

## Cold start

Use [`paper-mixed-graph-oov-v2`](../benchmark-protocol.md#cold-start-evaluation): full training,
full-validation checkpoint selection, and final evaluation on mixed user/item test subsets.
Few-shot uses candidate support <5 including zero. Zero-shot uses absence from the training
graph, including item exposure in histories. Unseen identities map to distinct shared user/item
OOV rows after selection, without fine-tuning or changing non-ID inputs.

Retain population audits and the full-test comparator beside subset metrics. Empty or
single-class subsets are `not_evaluable`; report their counts explicitly.

## Alignment

The diagnostic path supports KL, symmetric InfoNCE, cosine, five-kernel MMD, and adversarial
alignment with coefficients 0.1, 0.3, 0.5, and 0.7. It aligns each non-ID modality with ID.
Ordinary adapters expose projected target features; QARM and PSRQ expose codebook-embedded,
projected history means with explicit padding masks.

Adversarial alignment uses discriminator and generator optimization phases while retaining
native backbone losses and alternating phases. KL is evaluated in log space as
`exp(log_p) * (log_p - log_q)` to support finite gradients at underflowed probabilities.
The generic `plan-alignment-study` hook-based API has its own `cosine`/`mse` method set;
use the diagnostic executor for the five-objective matrix.

## Modality robustness

The diagnostic executor jointly drops all non-ID target and history modalities for each sampled
row at rates 0.1, 0.3, 0.5, and 0.7. It perturbs training and test while keeping validation clean.
IDs and history validity masks remain present. Its deterministic mask generator uses seed 42
with epoch/batch offsets.

The [generic robustness planner](../experiments.md#modal-robustness) supports independent
per-modality masks on explicitly configured splits. These are distinct perturbation protocols;
report the mask coupling and affected splits with each result.

## Fusion

The diagnostic operators are CAT, MAF, LMF, MTFN, SRC, FQ-Former, SimCEN, DTA, and DMF.
Explicit model adapters define their substitution sites and preserve padding-mask semantics.

LMF, MTFN, GMMF, and SimCEN diagnostics use a shared sequence wrapper with
`structure_policy=legacy_forced_shared_wrapper`. These variants study the shared architecture;
they do not preserve native GMMF adversarial decomposition or the native SimCEN architecture.

The projected DMF diagnostic adapter uses discrete similarity buckets. The local content branch
does not receive click-loss gradients through bucket indices; an operator-specific auxiliary
loss can supply a separate gradient. Its representation settings must be recorded independently
of the [native DMF configuration](../benchmark-protocol.md#dmf-configuration).

MAKE retains similarity tiers and target-aware attention; NAML retains attention pooling and
dot-product prediction; EM3 retains its content/ID contrastive loss. MMMLP retains modality and
temporal mixers for local operators, while sequence operators bypass those mixers. M3SRec
retains shared attention and expert refinement. QARM retains codebook encoding and Cross+Deep;
the PSRQ fusion adapter uses history mean pooling and configurable fusion with frozen quantizers
and joint-code features.

MB retains modality scores and its balance objective. PAMD retains target-pair disentanglement;
sequence fusion substitutes its history branch. MARN retains specific/invariant splitting and
auxiliary classifiers. Diff-MSIN retains experts, two-level gates, and its cross network;
sequence operators receive a single-step sequence built from pooled features. Its MTFN
diagnostic predictor accumulates BatchNorm moments in float64 while keeping learned parameters
and outputs in float32.

SRC requires at least two inputs. A single content-modality pair, such as MicroLens PAMD,
cannot form a valid SRC combination and is rejected by the adapter.

## Artifacts

The executor consumes a matrix, a frozen baseline, a task ID, an output root, and a device:

```bash
<BM_PYTHON> -m mmctr.analysis.diagnostic_execution \
  --matrix outputs/analysis/tasks.json \
  --baseline outputs/analysis/main-table-baseline.json \
  --task-id <TASK_ID> --output-root outputs/analysis/results --cuda 0
```

Replace the placeholders with the absolute environment interpreter and a task ID from the
matrix. The scheduler writes consolidated JSON/CSV reports and per-task `result.json` artifacts.
Keep task fingerprints and source/configuration provenance when aggregating diagnostic results.
