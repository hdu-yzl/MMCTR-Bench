# Cross-dataset diagnostic studies

Run cold-start, modality robustness, alignment, and fusion experiments one setting at a time.
One invocation selects a dataset, a model, a seed, and a fixed analysis setting, such as one
masking rate or one fusion operator. Cold start and the native control are available on all
four datasets; the other analysis settings are available on MicroLens and TikTok.

## Tuning and comparisons

Before running the experiments, define the settings to report and the candidate
hyperparameters for each one. Fit every candidate on training data and use validation AUC
for both checkpoint and hyperparameter selection. Ties keep the first candidate. Freeze the
highest-scoring candidate and its validation-selected weights, then evaluate on test. The
final scores describe the best candidates **within the declared search**, and test scores
do not guide later changes to the search space.

For robustness, alignment, and fusion, run a matching native `control` with no intervention.
Tune it separately on validation, just as you tune each analysis setting. Use the same dataset
fingerprint, split protocol, model backbone, paired seeds, and comparable search spaces and
budgets. Do not copy the hyperparameters or scores from the README main tables into these
comparisons. Cold start instead compares two test populations using the same selected weights.

For each matched seed, report test AUC and LogLoss and the signed AUC difference
(`setting - control`). If useful, also report the relative change
(`100 * (setting - control) / control`), undefined when the comparator is zero. Record the
candidate budget, the selected configurations, and the dataset fingerprint. For multiple
seeds, report the mean and sample standard deviation of the paired differences; a single
seed does not establish statistical significance.

## Cold start

Use [`paper-mixed-graph-oov-v2`](../benchmark-protocol.md#cold-start-evaluation): full training,
full-validation checkpoint and hyperparameter selection, and final evaluation on mixed
user/item test subsets.
Few-shot uses candidate support <5 including zero. Zero-shot uses absence from the training
graph, including item exposure in histories. Unseen identities map to distinct shared user/item
OOV rows after selection, without fine-tuning or changing non-ID inputs.

The frozen weights and OOV policy are identical for the full-test comparator and the subset.
The executor audits the test subset after selection. Empty or single-class subsets are
`not_evaluable`; report their counts and omit their AUC and LogLoss.

## Alignment

The diagnostic path supports KL, symmetric InfoNCE, cosine, five-kernel MMD, and adversarial
alignment with coefficients 0.1, 0.3, 0.5, and 0.7. Use `contrastive` for InfoNCE and
keep Adam fixed across alignment candidates. Each non-ID modality is aligned with ID.
Ordinary adapters expose projected target features; QARM and PSRQ expose codebook-embedded,
projected history means with explicit padding masks.

For DMF alignment, set `model: {similarity_source: legacy_projected}` in **every** candidate
in both the alignment and control searches. This variant supplies same-width projections;
the frozen-feature DMF variant uses the original modality dimensions and does not support
this alignment objective.

Adversarial alignment uses discriminator and generator optimization phases while retaining
native backbone losses and alternating phases. KL is evaluated in log space as
`exp(log_p) * (log_p - log_q)` to support finite gradients at underflowed probabilities.
The generic `plan-alignment-study` hook-based API has its own `cosine`/`mse` method set;
use the diagnostic executor to run the objectives listed here.

## Modality robustness

The diagnostic executor jointly drops all non-ID target and history modalities for each sampled
row at rates 0.1, 0.3, 0.5, and 0.7. It perturbs training and test while keeping validation clean.
IDs and history validity masks remain present. Its deterministic mask generator uses seed 42
with epoch/batch offsets. Keep batch size fixed across masking rates so that the same
row-level random draws are used at test time.

The [generic robustness planner](../experiments.md#modal-robustness) supports independent
per-modality masks on explicitly configured splits. These are distinct perturbation protocols;
report the mask coupling and affected splits with each result.

## Fusion

The diagnostic operators are CAT, MAF, LMF, MTFN, SRC, FQ-Former, SimCEN, DTA, and DMF.
Explicit model adapters define their substitution sites and preserve padding-mask semantics.

LMF, MTFN, GMMF, and SimCEN diagnostics use a shared sequence wrapper with
`structure_policy=legacy_forced_shared_wrapper`. These variants study the shared architecture;
they do not preserve native GMMF adversarial decomposition or the native SimCEN architecture.
A difference against the native control can therefore also reflect the backbone change.
To isolate an operator where the wrapper and substitution site are the same, separately
tune a reference operator within that wrapper.

The projected DMF diagnostic adapter uses discrete similarity buckets. The local content branch
does not receive click-loss gradients through bucket indices; an operator-specific auxiliary
loss can supply a separate gradient. Set `model: {similarity_source: legacy_projected}` in
every DMF fusion candidate and in its matching control to compare projected variants. See the
[DMF configuration](../benchmark-protocol.md#dmf-configuration) for the similarity modes.

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

## Run one setting

Use `mmctr.analysis.diagnostic_execution` with one of `control`, `robustness`, `alignment`,
`fusion`, or `cold-start`. Run the commands from the repository root after activating the
experiment environment. This example uses two independent commands for a TikTok robustness
setting and its separately tuned clean control:

```bash
MMCTR_PYTHON="$(command -v python)"

"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution robustness \
  --dataset-name tiktok --model-name dmf --probability 0.3 --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-drop03-s2025 --output-root outputs/analysis --cuda 0

"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution control \
  --dataset-name tiktok --model-name dmf --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-control-s2025 --output-root outputs/analysis --cuda 0
```

Pass `--dataset-config` if the dataset entries are in a local catalog, and see each subcommand's
`--help` for its arguments. A cold-start example is in the
[benchmark protocol](../benchmark-protocol.md#cold-start-evaluation). The candidate YAML
provides an example search over learning rate and weight decay; adapt it before running a
study. The executor evaluates the supplied candidate list and does not implement an
automatic Optuna or time-budget policy. The generic `plan-*-study` commands generate task
descriptions only.

Each study writes `trial_history.json`, `frozen_selection.json`, and `final_test_result.json`
under `<output-root>/<study-id>/`, alongside per-run logs and resolved settings. The selected
weights are retained in memory, without `.pt` files. Use a unique study ID for every setting
and seed. Compare the matched final result files; trial history contains validation metrics
only. For cold start, one final result contains the full-test metrics (`test_full_auc`,
`test_full_loss`), subset metrics when available, and the subset audit and AUC comparison.
