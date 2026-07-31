"""
PAMD 模型 —— 融合分析版本（强制兼容所有融合方法）
原始模型对每对模态进行解耦（共有/独有），并以 mean 池化所有 pair 表示作为最终融合。
此版本保留 PAMD 解耦机制（模型贡献），并支持所有本地融合 + 序列感知融合。

模型核心操作（保持不变）:
  - 模态对解耦（_PAMDDisentangleBlock）：common_a / common_b / spec_a / spec_b
  - 解耦的辅助损失：align + orth + rank
  - 每对的 pair-attention 融合（基于 ID 作为 query）

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    - 将目标侧 / 序列侧 “mean over pairs” 替换为对 pair 表示的可配置融合
    - 序列侧仍先池化到 2D，再做 pair 解耦 + 融合
  序列感知融合（dta, gmmf, dmf）:
    - 目标侧仍走 pair 解耦 + 配置融合（保留 PAMD 贡献）
    - 序列侧跳过 pair 解耦，直接对原始 3D 序列特征应用序列感知融合
"""
from itertools import combinations
import math

from models.base_model import BaseModel
from models.layers.common import MultiLayerPerceptron
from models.layers import seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PAMDDisentangleBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, use_layer_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.common_a = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.common_b = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.a2b = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.b2a = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _mlp(in_dim, hidden_dim, out_dim, dropout, use_layer_norm):
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*layers)

    def decompose(self, a, b):
        common_a = self.common_a(a)
        common_b = self.common_b(b)
        return common_a, common_b, a - common_a, b - common_b

    def aux_loss(self, a, b, common_a, common_b, spec_a, spec_b):
        align_loss = F.mse_loss(common_a, common_b)
        orth_loss = torch.mean(torch.sum(F.normalize(spec_a, dim=-1) * F.normalize(spec_b, dim=-1), dim=-1).pow(2))

        loss_c = F.mse_loss(b, self.a2b(common_a)) + F.mse_loss(a, self.b2a(common_b))
        loss_p = F.mse_loss(b, self.a2b(a)) + F.mse_loss(a, self.b2a(b))
        loss_s = F.mse_loss(b, self.a2b(spec_a)) + F.mse_loss(a, self.b2a(spec_b))
        rank_loss = -F.logsigmoid(loss_p - loss_c) - F.logsigmoid(loss_s - loss_p)
        return align_loss + orth_loss + rank_loss

    def fuse(self, q, common_a, common_b, spec_a, spec_b):
        reps = torch.stack([common_a, common_b, spec_a, spec_b], dim=-2)
        score = torch.sum(self.attn_q(q).unsqueeze(-2) * self.attn_k(reps), dim=-1)
        weight = torch.softmax(score / math.sqrt(q.size(-1)), dim=-1).unsqueeze(-1)
        return q + torch.sum(weight * reps, dim=-2)

    def forward(self, a, b, q):
        common_a, common_b, spec_a, spec_b = self.decompose(a, b)
        fused = self.fuse(q, common_a, common_b, spec_a, spec_b)
        loss = self.aux_loss(a, b, common_a, common_b, spec_a, spec_b)
        return fused, loss


class PAMD(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.pamd_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_features]
        self.seq_pamd_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_seq_features]
        if len(self.pamd_features) < 2:
            raise ValueError('PAMD requires at least two modalities among image/text/audio.')

        # 序列感知融合所需的公共特征集
        self.seq_fusion_features = [k for k in self.mm_features if k in self.mm_seq_features]

        hidden_dim = model_config.get('pamd_hidden_dim', self.projection_dim)
        self.pamd_aux_weight = model_config.get('pamd_aux_weight', 0.1)
        self.pooling = seq_pooling.get_pooling(self.seq_pooling_method, dim=1)

        self.target_pamd = self._build_pamd_blocks(self.pamd_features, hidden_dim, model_config)

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'maf'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 目标侧 pair 表示融合（始终为本地融合，因为 pair 表示是 2D）
        # 当全局融合方法为序列感知时，目标侧 pair 融合退化为 'cat'
        target_pair_method = self._fusion_method if not self._is_seq_fusion else 'cat'
        self.target_pair_keys = [self._pair_key(a, b) for a, b in combinations(self.pamd_features, 2)]
        f_info_t = build_fusion(target_pair_method, self.projection_dim,
                                self.target_pair_keys, model_config)
        self.target_pair_fusion = f_info_t['layer']
        self._target_query_num = f_info_t['query_num']
        self._target_pair_method = target_pair_method
        target_pair_out = f_info_t['out_dim']
        self.target_pair_projector = (nn.Linear(target_pair_out, self.projection_dim)
                                      if target_pair_out != self.projection_dim
                                      else nn.Identity())

        if self._is_seq_fusion:
            # 序列感知融合：跳过 seq pair 解耦，直接对原始 3D 序列融合
            f_info_s = build_fusion(self._fusion_method, self.projection_dim,
                                    self.seq_fusion_features, model_config)
            self.seq_full_fusion = f_info_s['layer']
            self._seq_full_query_num = f_info_s['query_num']
            self._sim_fusion = f_info_s.get('sim_fusion')
            self._gmmf_pooling = f_info_s.get('seq_pooling')
            seq_full_out = f_info_s['out_dim']
            self.seq_full_projector = (nn.Linear(seq_full_out, self.projection_dim)
                                       if seq_full_out != self.projection_dim
                                       else nn.Identity())
            self.seq_pamd = None
            self.seq_pair_fusion = None
            self._seq_query_num = None
            self.seq_pair_projector = nn.Identity()
        else:
            # 本地融合：保留 seq pair 解耦
            if len(self.seq_pamd_features) < 2:
                raise ValueError('PAMD (local fusion) requires at least two seq modalities.')
            self.seq_pamd = self._build_pamd_blocks(self.seq_pamd_features, hidden_dim, model_config)
            self.seq_pair_keys = [self._pair_key(a, b) for a, b in combinations(self.seq_pamd_features, 2)]
            f_info_s = build_fusion(self._fusion_method, self.projection_dim,
                                    self.seq_pair_keys, model_config)
            self.seq_pair_fusion = f_info_s['layer']
            self._seq_query_num = f_info_s['query_num']
            seq_pair_out = f_info_s['out_dim']
            self.seq_pair_projector = (nn.Linear(seq_pair_out, self.projection_dim)
                                       if seq_pair_out != self.projection_dim
                                       else nn.Identity())

        self.dnn = MultiLayerPerceptron(
            self.projection_dim * 2,
            self.mlp_dims,
            self.dropout,
            use_bn=self.bn,
            activation='relu',
        )
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            use_bn=self.bn,
            activation=None,
        )

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def _build_pamd_blocks(self, features, hidden_dim, model_config):
        return nn.ModuleDict({
            self._pair_key(a, b): _PAMDDisentangleBlock(
                self.projection_dim,
                hidden_dim=hidden_dim,
                dropout=self.dropout,
                use_layer_norm=model_config.get('pamd_layer_norm', True),
            )
            for a, b in combinations(features, 2)
        })

    @staticmethod
    def _pair_key(a, b):
        return f'{a}__{b}'

    def _run_pamd_local(self, blocks, features, feats_p, q, fusion_layer, projector,
                        pair_keys, query_num, fusion_method):
        """对每个 pair 进行解耦融合，并将 pair 表示用配置融合层聚合。"""
        pair_dict = {}
        aux_loss = []
        for a, b in combinations(features, 2):
            key = self._pair_key(a, b)
            fused, loss = blocks[key](feats_p[a], feats_p[b], q)
            pair_dict[key] = fused
            aux_loss.append(loss)

        fused, cl = fuse_local(fusion_layer, pair_dict, pair_keys, fusion_method, query_num)
        fused = projector(fused)
        aux = torch.stack(aux_loss).mean()
        if isinstance(cl, torch.Tensor):
            aux = aux + cl
        return fused, aux

    def forward(self, feats, feats_seq):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_seq_projector[k](feats_seq[k]) for k in self.mm_seq_features}

        # 目标侧（始终走 pair 解耦 + 本地融合）
        feats_fusion, target_aux = self._run_pamd_local(
            self.target_pamd, self.pamd_features, feats_p, feats_p['id'],
            self.target_pair_fusion, self.target_pair_projector,
            self.target_pair_keys, self._target_query_num, self._target_pair_method,
        )

        au_loss = feats_fusion.new_tensor(0.0)
        au_loss = au_loss + target_aux

        if self._is_seq_fusion:
            # 跳过 pair 解耦，直接对原始 3D 序列融合
            target_dict = {k: feats_p[k] for k in self.seq_fusion_features}
            seq_dict = {k: feats_seq_p[k] for k in self.seq_fusion_features}

            if self._fusion_method == 'dta':
                fused_seq, _ = fuse_seq_dta(self.seq_full_fusion, self._sim_fusion,
                                            target_dict, seq_dict,
                                            self.seq_fusion_features, self.seq_len)
            elif self._fusion_method == 'gmmf':
                user_vec_tmp = torch.cat([target_dict[k] for k in self.seq_fusion_features], dim=-1)
                fused_seq, gmmf_loss = fuse_seq_gmmf(self.seq_full_fusion, self._gmmf_pooling,
                                                    target_dict, seq_dict, user_vec_tmp)
                au_loss = au_loss + gmmf_loss
            else:  # dmf
                fused_seq, _ = fuse_seq_dmf(self.seq_full_fusion,
                                            target_dict, seq_dict, self.seq_len)

            feats_seq_pool_fusion = self.seq_full_projector(fused_seq)
        else:
            # 本地融合：池化后的 pair 解耦 + 融合
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}
            feats_seq_pool_fusion, seq_aux = self._run_pamd_local(
                self.seq_pamd, self.seq_pamd_features, feats_seq_pool, feats_seq_pool['id'],
                self.seq_pair_fusion, self.seq_pair_projector,
                self.seq_pair_keys, self._seq_query_num, self._fusion_method,
            )
            au_loss = au_loss + seq_aux

        x_dnn = torch.cat([feats_fusion, feats_seq_pool_fusion], dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit, 'au_loss': self.pamd_aux_weight * au_loss}
        return out
