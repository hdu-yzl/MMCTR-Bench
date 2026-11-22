# Methods

MMCTR-Bench evaluates five ID-only baselines and 16 multimodal methods. Two additional DNN
variants are available for comparison. Use the table names as `--model` values in the launchers.

## ID-only baselines (5)

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `dnn` | An MLP scores user and target IDs with pooled history IDs. | — |
| `deepfm` | Combines factorization-machine pairwise terms with a deep prediction branch. | Huifeng Guo et al., [*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*](https://doi.org/10.24963/ijcai.2017/239), 2017. |
| `din` | Uses the target item to attend to relevant interactions in the user's history. | Guorui Zhou et al., [*Deep Interest Network for Click-Through Rate Prediction*](https://doi.org/10.1145/3219819.3219823), 2018. |
| `autoint` | Learns feature interactions with self-attention across fields. | Weiping Song et al., [*AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*](https://doi.org/10.1145/3357384.3357925), 2019. |
| `dcn` | Combines explicit feature crosses with an MLP. | Ruoxi Wang et al., [*Deep & Cross Network for Ad Click Predictions*](https://doi.org/10.1145/3124749.3124754), 2017. |

## TMIE — Target-aware Multimodal Interest Enhancement (3)

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `mmmlp` | Mixes modalities and history positions with MLP blocks. | Jiahao Liang et al., *MMMLP: Multi-Modal Multilayer Perceptron for Sequential Recommendations*, 2023. |
| `diff_msin` | Combines shared and modality-specific experts with stochastic reverse fusion and gated interactions. | Xiaoxi Cui et al., [*Diffusion-Based Multi-Modal Synergy Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/2508.21460), 2025. |
| `dmf` | Combines target-aware ID attention with bucketed similarity from frozen multimodal vectors. | Alin Fan et al., [*Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction*](https://arxiv.org/abs/2510.11066), 2025. |

## CSAQ — Content Semantic Alignment and Quantization (5)

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `make` | Combines target-aware history attention with counts in multimodal similarity tiers. | MAKE adaptation of Xiang-Rong Sheng et al., *Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights*, 2024. |
| `m3srec` | Uses shared self-attention and a mixture of experts to refine multimodal histories. | Shuqing Bian et al., *Multi-Modal Mixture of Experts Representation Learning for Sequential Recommendation*, 2023. |
| `em3` | Compresses content with learned queries and aligns content and item IDs through a contrastive loss. | Xiuqi Deng et al., [*End-to-End Training of Multimodal Model and Ranking Model*](https://arxiv.org/abs/2404.06078), 2024. |
| `psrq` | Pools history with attention over item representations from pretrained modality and joint codebooks. | Shijia Wang et al., [*Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation*](https://arxiv.org/abs/2508.20359), 2025. |
| `qarm` | Embeds residual modality codes, pools their histories, and scores with a cross-and-deep head. | Xinchen Luo et al., *QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou*, 2025. |

## GFFI — General Fusion and Feature Interaction (5)

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `naml` | A benchmark adaptation with masked view-level attention followed by history-level attention. Projected frozen content features and an ID view replace the original news view encoders. | Chuhan Wu et al., [*Neural News Recommendation with Attentive Multi-View Learning*](https://doi.org/10.24963/ijcai.2019/536), 2019. |
| `mb` | Learns modality weights and uses an auxiliary objective to balance modalities. | Yu Shang et al., *Enhancing Adversarial Robustness of Multi-Modal Recommendation via Modality Balancing*, 2023. |
| `lmf` | Uses low-rank tensor fusion on target and history modalities. | Zhun Liu et al., *Efficient Low-Rank Multimodal Fusion with Modality-Specific Factors*, 2018. |
| `simcen` | Uses segmented feature experts and a contrastive objective across two views. | Honghao Li et al., *SimCEN: Simple Contrast-Enhanced Network for CTR Prediction*, 2024. |
| `mtfn` | Models cross-modal interactions through factorized tensor fusion. | Tan Wang et al., *Matching Images and Text with Multi-Modal Tensor Fusion and Re-Ranking*, 2019. |

## RDRR — Representation Disentanglement and Redundancy Reduction (3)

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `pamd` | Disentangles modality pairs and uses ranking and orthogonality losses. | Tengyue Han et al., *Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation*, 2022. |
| `gmmf` | Uses modality decomposition and adversarial generation with alternating optimizer phases. | Fangxiong Xiao et al., *From Abstract to Details: A Generative Multimodal Fusion Framework for Recommendation*, 2022. |
| `marn` | Separates modality-specific information from adversarially learned modality-invariant interests. | Xiang Li et al., [*Adversarial Multimodal Representation Learning for Click-Through Rate Prediction*](https://arxiv.org/abs/2003.07162), WWW 2020. |

## Software-only reference variants (2)

These two software baselines are outside the paper's 21-model evaluation set.

| Registry name | Method in this benchmark | Reference |
|---|---|---|
| `dnn_mm` | An MLP reference that combines ID and multimodal target and pooled-history features. | — |
| `dnn_mm_seq` | A multimodal DNN reference with separate user, target, and history branches. | — |

## Normalization and NAML adaptation

DIN-style pooling (used by DIN, MAKE, EM3, Diff-MSIN, and MARN) computes Dice
statistics per hidden channel over valid history tokens during training. It stores
exponential moving averages of the population mean and variance (momentum 0.1) and
uses these fixed buffers during evaluation. Padded positions never enter the Dice
network; an all-padding batch leaves the buffers unchanged. A single valid token
uses zero population variance and remains finite.

NAML uses shared additive view-level attention for targets and history items, then
learned attention over valid history items. View weights are normalized over present
views only; missing content and padded items receive zero weight. Presence comes from
raw input vectors before projection. The ID view is present for every target and
valid history item. `target_view_weights` and `history_view_weights` in model output
representations follow the configured `use_mm_features` order.

This NAML implementation adapts the original news model to the common CTR protocol:
projected frozen text/image/audio features and item IDs replace its raw
title/abstract/category encoders, and the user context branch remains in place.
English comments in `naml.py` mark this adaptation explicitly. The executable NAML
pipeline preset uses the same view-attention fusion and history attention.

These fixes change learned behavior. Retrain the affected models and regenerate
benchmark metrics; historical result tables have not been recomputed. Old Dice
checkpoints lack running-statistic buffers, and old NAML checkpoints contain sum
fusion parameters instead of attention parameters. They cannot be loaded strictly
as checkpoints of the corrected models; using `strict=False` does not recover the
missing learned state.

## Quantizer requirements

QARM requires pretrained RQ artifacts; PSRQ requires pretrained PSRQ artifacts. Train them
for the same dataset used by the CTR run. See the [training commands](../README.md#one-click-training).
