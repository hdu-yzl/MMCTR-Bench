# Dataset and model references

This is the citation and provenance index for the public registry. It distinguishes a paper's
algorithm from this repository's canonical implementation: a matching name does not by itself claim
bitwise reproduction, checkpoint compatibility, or permission to redistribute the paper's data.
Implementation behavior and numerical evidence are documented in [`models.md`](models.md).

## Dataset papers

- **AntM2C:** Zhaoxin Huan et al. *AntM²C: A Large Scale Dataset For Multi-Scenario Multi-Modal CTR
  Prediction*. arXiv:2308.16437, 2023. [Paper](https://arxiv.org/abs/2308.16437) and
  [provider homepage](https://www.atecup.com/home).
- **MicroLens:** Yongxin Ni et al. *A Content-Driven Micro-Video Recommendation Dataset at Scale*.
  arXiv:2309.15379, 2023. [Paper](https://arxiv.org/abs/2309.15379) and
  [official repository](https://github.com/westlake-repl/MicroLens).
- **TikTok processed layout:** the locally observed layout is traceable to the `tiktok/` directory
  in the [official InvRL implementation](https://github.com/nickwzk/InvRL). The corresponding paper
  is Xiaoyu Du et al., *Invariant Representation Learning for Multimedia Recommendation*, ACM
  Multimedia 2022. This is a provenance lead, not a verified data license; the maintainer must
  retain the actual acquisition record.

## Model registry

Status meanings:

- **paper adaptation:** the implementation is based on a named published method, adapted to the
  benchmark's `Batch -> ModelOutput` and CTR contracts;
- **fusion adaptation:** a published fusion operator is used inside a benchmark CTR body;
- **benchmark variant:** this repository does not claim a separate paper for that registry entry.

| Registry name | Status | Reference or provenance |
|---|---|---|
| `dnn` | benchmark variant | Generic multilayer-perceptron CTR baseline; no separate paper is claimed. |
| `dnn_mm` | benchmark variant | DNN with canonical target/history multimodal fusion; no separate paper is claimed. |
| `dnn_mm_seq` | benchmark variant | Sequential DNN-MM variant; legacy alias `dnn_seq`; no separate paper is claimed. |
| `dcn` | paper adaptation | Ruoxi Wang et al., [*Deep & Cross Network for Ad Click Predictions*](https://arxiv.org/abs/1708.05123), 2017. |
| `deepfm` | paper adaptation | Huifeng Guo et al., [*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*](https://arxiv.org/abs/1703.04247), 2017. |
| `din` | paper adaptation | Guorui Zhou et al., [*Deep Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1706.06978), 2018. |
| `autoint` | paper adaptation | Weiping Song et al., [*AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*](https://arxiv.org/abs/1810.11921), 2019. |
| `lmf` | fusion adaptation | Zhun Liu et al., [*Efficient Low-rank Multimodal Fusion with Modality-Specific Factors*](https://aclanthology.org/P18-1209/), 2018. |
| `mtfn` | fusion adaptation | Tan Wang et al., [*Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking*](https://arxiv.org/abs/1908.04011), 2019. |
| `naml` | paper adaptation | Chuhan Wu et al., [*Neural News Recommendation with Attentive Multi-View Learning*](https://arxiv.org/abs/1907.05576), 2019; adapted here to canonical item modalities. |
| `marn` | paper adaptation | Xiang Li et al., [*Adversarial Multimodal Representation Learning for Click-Through Rate Prediction*](https://arxiv.org/abs/2003.07162), 2020. |
| `gmmf` | paper adaptation | Fangxiong Xiao et al., [*From Abstract to Details: A Generative Multimodal Fusion Framework for Recommendation*](https://dblp.org/rec/conf/mm/XiaoDCJYDL22), ACM Multimedia 2022. |
| `pamd` | paper adaptation | Hongyu Han et al., [*Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation*](https://doi.org/10.1145/3485447.3512079), 2022; the canonical CTR consumer retains its common/specific decomposition objective. |
| `mmmlp` | paper adaptation | Jiahao Liang et al., [*MMMLP: Multi-modal Multilayer Perceptron for Sequential Recommendations*](https://doi.org/10.1145/3543507.3583378), 2023. |
| `m3srec` | paper adaptation | Shuqing Bian et al., [*Multi-modal Mixture of Experts Representation Learning for Sequential Recommendation*](https://doi.org/10.1145/3583780.3614978), 2023. |
| `mb` | paper adaptation | Yu Shang et al., [*Enhancing Adversarial Robustness of Multi-modal Recommendation via Modality Balancing*](https://doi.org/10.1145/3581783.3612337), 2023. |
| `em3` | paper adaptation | Nan Xu et al., [*End-to-end Training of Multimodal Model and Ranking Model*](https://arxiv.org/abs/2302.03497), 2023. |
| `make` | paper adaptation | Xinyang Chen et al., [*Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights*](https://arxiv.org/abs/2407.19467), 2024; MAKE is the paper's staged adaptation module. |
| `simcen` | paper adaptation | Honghao Li et al., [*SimCEN: Simple Contrast-enhanced Network for CTR Prediction*](https://openreview.net/forum?id=pJHu4hDlLX), ACM Multimedia 2024. |
| `qarm` | paper adaptation | Xinchen Luo et al., [*QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou*](https://arxiv.org/abs/2411.11739), 2024. |
| `diff_msin` | paper adaptation | Xiaoxi Cui et al., [*Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction*](https://arxiv.org/abs/2508.21460), 2025. |
| `mcca` | paper adaptation | Shijia Wang et al., [*Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation*](https://arxiv.org/abs/2508.20359), 2025; MCCA is the downstream cross-attention network and PSRQ is its pretraining component. |
| `dmf` | paper adaptation | Alin Fan et al., [*Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction*](https://arxiv.org/abs/2510.11066), 2025/2026 revision. |

RQ and PSRQ are quantization pretraining components rather than CTR registry models. Cite the QARM
paper for the RQ-based QARM path and the PSRQ/MCCA paper for the PSRQ-based MCCA path.

## Citation practice

When reporting results, cite MMCTR-Bench through [`CITATION.cff`](../CITATION.cff), the dataset paper,
and every evaluated model paper above. For benchmark variants, cite the benchmark software and state
the exact registry/config name. Do not cite a paper as evidence of data redistribution rights.
