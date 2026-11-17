# Benchmark protocol

MMCTR-Bench uses versioned canonical datasets, validation-based model selection, and isolated
run artifacts. The [README reference tables](../README.md#reference-results) provide reference
results after dataset restructuring, with five-seed mean and sample standard deviation.

## Model selection and final evaluation

Training fits model parameters on the training split. Validation AUC controls early stopping,
checkpoint selection, and hyperparameter selection. Test evaluation follows configuration and
checkpoint freezing.

`mmctr.cli tune` accepts a predeclared list of candidate configurations and runs each with the
production Trainer. Every candidate records full-validation AUC and LogLoss; the highest
validation AUC wins, with ties keeping the first candidate. Test evaluation is disabled during
candidate search. A fresh Trainer loads the exact selected weights, including buffers, and
evaluates test once without retraining.

The study stores candidate configurations and validation metrics, an immutable frozen selection,
and a separate final-test result. A study lock and provenance checks prevent selection artifacts
from being overwritten. See [experiment commands](experiments.md#validation-only-tuning).
The candidate-list interface runs the supplied configurations; it does not implement the
paper's Optuna 50-trial / 50-GPU-hour stopping rule.

## Dataset versions

AntM2C, MicroLens, and TikTok use canonical-v1. EB-NeRD reference results use canonical-v4 with
history length 5, ID embedding width 16, and 10% of local-test impressions. EB-NeRD uses the
provider's validation split as local test; this is not the official challenge test. Dataset
manifests and per-run metadata retain fingerprints and resolved configurations.

TikTok canonical-v1 contains 9,308 users, 6,710 target items, and 412,332 binary CTR instances:
357,246 training, 18,306 validation, and 36,780 test rows. The positive:negative ratio is 1:5.
See the [array-measured breakdown](data.md#tiktok-canonical-v1).

TikTok uses the supplied split files and list-prefix histories. Source files lack reliable
timestamps, so list order does not establish temporal causality. Negative sampling excludes
all supplied positives for each user across the three splits, including later-split positives.
This exclusion policy does not provide an online causal negative-sampling guarantee.

## Reference result aggregation

Each of the four README tables contains 21 models evaluated with seeds 2025–2029. AUC and
LogLoss are arithmetic mean ± sample standard deviation (`ddof=1`), computed from all five
per-seed scores. The table generator validates matrix completeness and agreement between raw
scores and report aggregates before writing any table. Per-seed results, configuration, data
fingerprints, and implementation provenance remain in the source report's job records.

## Cold-start evaluation

Protocol identity: `paper-mixed-graph-oov-v2`, based on benchmark Appendix C (pp. 11–12).

- Train on the complete training set and select the checkpoint using **full validation**.
- Few-shot: include a test row if its user **or** item has fewer than 5 training candidate
  interactions, including zero. Positive and sampled negative rows both count. History
  appearances do not add candidate-interaction counts.
- Zero-shot: include a test row if its user **or** item is absent from the training graph.
  Item exposure includes both training candidates and valid training-history positions.
- After freezing the selected weights, append distinct user/item OOV indices with deterministic
  zero vectors. Map unseen users, target items, and valid history IDs consistently. Preserve
  non-ID features, context, labels, padding, and history masks. No fine-tuning follows OOV
  reservation. AntM2C user ID zero remains valid.
- Evaluate the subset and the complete test set with the same selected weights and OOV policy.
  Report signed AUC change and `100 * (subset_auc - full_test_auc) / full_test_auc`. Subset
  composition contributes to this difference; it is not an isolated measure of model degradation.
- Empty or single-class test subsets are `not_evaluable`. Report a subset that covers the full
  test set as such, without changing its threshold or the dataset split.

The canonical-v1 population audit gives:

| Dataset | Zero-shot mixed test rows | Few-shot mixed test rows (<5) |
|---|---:|---:|
| AntM2C | 386,775 | 410,705 |
| MicroLens | 2,862 | 360,000 (full test) |
| TikTok | 30 | 30 |

TikTok's two subsets coincide and contain five positives and 25 negatives. Their size limits
the strength of cold-start conclusions. These are population counts, not model scores.

Run one cold-start setting with a predeclared candidate list. Set `MMCTR_PYTHON` to the
absolute interpreter path in your experiment environment:

```bash
"$MMCTR_PYTHON" -m mmctr.analysis.diagnostic_execution cold-start \
  --dataset-name tiktok --model-name dmf --regime zero --seed 2025 \
  --candidates configs/experiments/diagnostic-tuning.example.yaml \
  --study-id tiktok-dmf-cold-zero-s2025 --output-root outputs/analysis --cuda 0
```

Use `--regime low_lt5` for few-shot. Both hyperparameter and checkpoint selection use full
validation; the final result compares the selected weights on the full and subset tests.
See the [analysis experiment guide](analysis/cross-dataset-diagnostics.md) for candidate
files, other analysis settings, and reporting.

`ColdStartProtocol` is a separate entity-support audit API. Its v2 few-shot constraint is strict
`<N` including zero; item zero-shot audits accept training-history IDs. Its audit and matrix
loaders require v2 schemas. See [the audit workflow](experiments.md#cold-start-protocol).

## DMF configuration

The default `similarity_source: frozen_features` selects `dmf-frozen-features-v2`.
[DMF IV-A / Eq. 10](https://arxiv.org/html/2510.11066) computes cosine similarity from frozen
pretrained multimodal vectors; IV-C uses discrete similarity-bucket embedding lookups.

The native benchmark model computes cosine from concatenated upstream modality vectors.
ID projections, ID embeddings, similarity key/value embeddings, and prediction heads are
trainable. The DTA buckets and histograms remain discrete; no surrogate bucket gradient is used.
The benchmark consumes pre-extracted features and does not perform DMF's additional contrastive
encoder fine-tuning.

`similarity_source: legacy_projected` selects the compatibility implementation with projected
modality similarity. Checkpoints require matching implementation and pipeline settings for
strict loading. Alignment and fusion studies also record their own representation and adapter
settings; see [diagnostic protocols](analysis/cross-dataset-diagnostics.md).
