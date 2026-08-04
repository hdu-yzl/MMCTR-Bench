# References and provenance

This page records the papers and upstream sources associated with the public dataset adapters and
model registry. A citation here identifies provenance; it does **not** grant this repository the
right to redistribute a third-party dataset, checkpoint, or feature artifact. Obtain those assets
from their official sources and follow their terms.

## Datasets

- **AntM2C:** Zhaoxin Huan et al. *Exploring Multi-Scenario Multi-Modal CTR Prediction with a
  Large-Scale Dataset*. SIGIR, 2024. Use the
  [official dataset portal](https://www.atecup.cn/OfficalDataSet) and follow its access terms.
- **MicroLens:** Yongxin Ni et al. *A Content-Driven Micro-Video Recommendation Dataset at Scale*.
  CIKM, 2025. See the [official repository](https://github.com/westlake-repl/MicroLens).
- **TikTok:** Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. *Multi-Modal Self-Supervised
  Learning for Recommendation*. WWW, 2023. The benchmark's TikTok source is the dataset layout
  released with the paper in the [official HKUDS/MMSSL repository](https://github.com/HKUDS/MMSSL).

These references do not imply that the datasets are covered by this repository's Apache-2.0
license. Dataset redistribution and derived-artifact permissions remain governed by the respective
upstream owners and terms.

## Paper evaluation set

The MMCTR-Bench paper evaluates **five ID-only baselines** and **16 multimodal models**. The 16
multimodal models follow the paper's four modeling paradigms. Registry names are shown in code font;
where the paper name and software entry differ, the mapping is stated explicitly.

### ID-only baselines (5)

| Registry name | Reference |
|---|---|
| `dnn` | Standard multilayer-perceptron CTR baseline; no separate originating paper is claimed here. |
| `deepfm` | Huifeng Guo et al., [*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*](https://doi.org/10.24963/ijcai.2017/239), 2017. |
| `din` | Guorui Zhou et al., [*Deep Interest Network for Click-Through Rate Prediction*](https://doi.org/10.1145/3219819.3219823), 2018. |
| `autoint` | Weiping Song et al., [*AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*](https://doi.org/10.1145/3357384.3357925), 2019. |
| `dcn` | Ruoxi Wang et al., [*Deep & Cross Network for Ad Click Predictions*](https://doi.org/10.1145/3124749.3124754), 2017. |

### TMIE — Target-aware Multimodal Interest Enhancement (3)

| Registry name | Reference |
|---|---|
| `mmmlp` | Jiahao Liang et al., *MMMLP: Multi-Modal Multilayer Perceptron for Sequential Recommendations*, 2023. |
| `diff_msin` | Xiaoxi Cui et al., [*Diffusion-Based Multi-Modal Synergy Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/2508.21460), 2025. |
| `dmf` | Alin Fan et al., [*Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction*](https://arxiv.org/abs/2510.11066), 2025. |

### CSAQ — Content Semantic Alignment and Quantization (5)

| Registry name | Paper model and reference |
|---|---|
| `make` | MAKE adaptation of Xiang-Rong Sheng et al., *Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights*, 2024. |
| `m3srec` | Shuqing Bian et al., *Multi-Modal Mixture of Experts Representation Learning for Sequential Recommendation*, 2023. |
| `em3` | Xiuqi Deng et al., [*End-to-End Training of Multimodal Model and Ranking Model*](https://arxiv.org/abs/2404.06078), 2024. |
| `psrq` | Shijia Wang et al., [*Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation*](https://arxiv.org/abs/2508.20359), 2025. The canonical model uses a PSRQ pretrainer whose artifacts are consumed by an internal downstream MCCA network; `mcca` is a compatibility alias only. |
| `qarm` | Xinchen Luo et al., *QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou*, 2025. |

### GFFI — General Fusion and Feature Interaction (5)

| Registry name | Reference |
|---|---|
| `naml` | Chuhan Wu et al., [*Neural News Recommendation with Attentive Multi-View Learning*](https://doi.org/10.24963/ijcai.2019/536), 2019. |
| `mb` | Yu Shang et al., *Enhancing Adversarial Robustness of Multi-Modal Recommendation via Modality Balancing*, 2023. |
| `lmf` | Zhun Liu et al., *Efficient Low-Rank Multimodal Fusion with Modality-Specific Factors*, 2018. |
| `simcen` | Honghao Li et al., *SimCEN: Simple Contrast-Enhanced Network for CTR Prediction*, 2024. |
| `mtfn` | Tan Wang et al., *Matching Images and Text with Multi-Modal Tensor Fusion and Re-Ranking*, 2019. |

### RDRR — Representation Disentanglement and Redundancy Reduction (3)

| Registry name | Reference |
|---|---|
| `pamd` | Tengyue Han et al., *Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation*, 2022. |
| `gmmf` | Fangxiong Xiao et al., *From Abstract to Details: A Generative Multimodal Fusion Framework for Recommendation*, 2022. |
| `marn` | Xiang Li et al., [*Adversarial Multimodal Representation Learning for Click-Through Rate Prediction*](https://arxiv.org/abs/2003.07162), WWW 2020. |

## Software-only reference variants (2)

These entries are useful controlled software baselines, but they are **not** additional models in the
paper's five-plus-16 evaluation set.

| Registry name | Provenance |
|---|---|
| `dnn_mm` | DNN with canonical target/history multimodal fusion; no separate paper is claimed. |
| `dnn_mm_seq` | Sequential DNN-MM variant (legacy alias `dnn_seq`); no separate paper is claimed. |

## Quantization pretraining components

RQ and PSRQ pretraining are not extra registry models. RQ produces artifacts for the downstream
`qarm` entry. PSRQ pretraining produces artifacts for the canonical downstream `psrq` entry, which
implements the paper model named **PSRQ**; `mcca` is only its backward-compatible registry alias.
