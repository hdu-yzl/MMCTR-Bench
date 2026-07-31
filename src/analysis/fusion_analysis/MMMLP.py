"""
MMMLP 模型 —— 融合分析版本
原始模型硬编码使用 “按通道维度 cat 模态后 + Fusion Mixer” 的融合策略，
此版本支持所有融合方法。

模型核心操作（保持不变）:
  - 每模态 MLP-Mixer（feature_mixer）：先 token-mixing 再 channel-mixing
  - 跨模态融合后再做一轮 Fusion Mixer
  - 用户特征拼接

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    - 每模态先经过 feature_mixer 精炼
    - 在每个时间步对模态进行可配置融合 → 投影对齐回 projection_dim
    - 沿时间维再做一轮 Fusion Mixer（保留模型贡献）
    - 目标侧同样使用本地融合 + 投影
  序列感知融合（dta, gmmf, dmf）:
    - 跳过 Fusion Mixer，直接用序列感知融合的输出作为序列表示
    - 目标侧使用 ID 嵌入（保持几何一致性）
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)

import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(self, seq_len, dim, token_hidden_dim=None, channel_hidden_dim=None, dropout=0.0):
        super().__init__()
        token_hidden_dim = token_hidden_dim or max(seq_len * 2, 1)
        channel_hidden_dim = channel_hidden_dim or dim * 2

        self.norm_token = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.norm_channel = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.token_mlp(self.norm_token(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.norm_channel(x))


class _MixerModule(nn.Module):
    def __init__(self, seq_len, dim, num_layers=2, token_hidden_dim=None, channel_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _MixerBlock(seq_len, dim, token_hidden_dim, channel_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MMMLP(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        if not self.mm_features:
            raise ValueError('MMMLP requires at least one modality.')

        self.feature_mixer_layers = model_config.get('feature_mixer_layers', model_config.get('mixer_layers', 2))
        self.fusion_mixer_layers = model_config.get('fusion_mixer_layers', model_config.get('mixer_layers', 2))
        self.token_hidden_dim = model_config.get('token_hidden_dim', max(self.seq_len * 2, 1))
        self.channel_hidden_dim = model_config.get('channel_hidden_dim', self.projection_dim * 2)
        self.user_dim = self.projection_dim * len(self.user_features)

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'cat'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 序列侧融合（用于每时间步本地融合 / 序列感知融合）
        f_info_seq = build_fusion(self._fusion_method, self.projection_dim,
                                  self.mm_features, model_config)
        self.seq_fusion_layer = f_info_seq['layer']
        self._seq_query_num = f_info_seq['query_num']
        self._seq_sim_fusion = f_info_seq.get('sim_fusion')
        self._seq_gmmf_pooling = f_info_seq.get('seq_pooling')
        seq_fusion_out = f_info_seq['out_dim']
        self.seq_fusion_projector = (nn.Linear(seq_fusion_out, self.projection_dim)
                                     if seq_fusion_out != self.projection_dim
                                     else nn.Identity())

        # 目标侧融合（仅本地融合时使用；序列感知融合下目标用 ID 嵌入）
        if not self._is_seq_fusion:
            f_info_tar = build_fusion(self._fusion_method, self.projection_dim,
                                      self.mm_features, model_config)
            self.target_fusion_layer = f_info_tar['layer']
            self._target_query_num = f_info_tar['query_num']
            target_fusion_out = f_info_tar['out_dim']
            self.target_fusion_projector = (nn.Linear(target_fusion_out, self.projection_dim)
                                            if target_fusion_out != self.projection_dim
                                            else nn.Identity())
        else:
            self.target_fusion_layer = None
            self._target_query_num = None
            self.target_fusion_projector = nn.Identity()

        # 每模态 feature_mixer（仅本地融合时使用）
        if not self._is_seq_fusion:
            self.feature_mixers = nn.ModuleDict({
                k: _MixerModule(
                    self.seq_len,
                    self.projection_dim,
                    num_layers=self.feature_mixer_layers,
                    token_hidden_dim=self.token_hidden_dim,
                    channel_hidden_dim=self.channel_hidden_dim,
                    dropout=self.dropout,
                )
                for k in self.mm_features
            })
            self.fusion_mixer = _MixerModule(
                self.seq_len,
                self.projection_dim,
                num_layers=self.fusion_mixer_layers,
                token_hidden_dim=self.token_hidden_dim,
                channel_hidden_dim=model_config.get('fusion_channel_hidden_dim', self.projection_dim * 2),
                dropout=self.dropout,
            )
            self.target_projector_post = nn.Sequential(
                nn.LayerNorm(self.projection_dim),
                nn.Linear(self.projection_dim, self.projection_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
        else:
            self.feature_mixers = None
            self.fusion_mixer = None
            self.target_projector_post = nn.Identity()

        self.dnn = MultiLayerPerceptron(
            self.projection_dim * 3 + self.user_dim,
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

    @staticmethod
    def _squeeze_item(x):
        return x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x

    def _last_hidden(self, seq_repr, seq_id_raw=None):
        if seq_id_raw is None:
            return seq_repr[:, -1]
        mask = seq_id_raw.ne(0).squeeze(-1) if seq_id_raw.dim() > 2 else seq_id_raw.ne(0)
        idx = mask.long().sum(dim=1).clamp(min=1) - 1
        return seq_repr[torch.arange(seq_repr.size(0), device=seq_repr.device), idx]

    def _encode_user_features(self, user_feats):
        if not self.user_features:
            return None
        user_outputs = []
        for k in self.user_features:
            x = self.embedding(user_feats[k]) if k == 'id' else user_feats[k].float()
            user_outputs.append(self.user_projector[k](self._squeeze_item(x)))
        return torch.cat(user_outputs, dim=-1)

    def forward(self, user_feats, feats, feats_seq):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        user_feats = dict(user_feats)
        seq_id_raw = feats_seq.get('id')

        if 'id' in feats:
            feats['id'] = self._squeeze_item(self.embedding(feats['id']))
        if 'id' in feats_seq:
            feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k] if k == 'id' else feats[k].float())
                   for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k] if k == 'id' else feats_seq[k].float())
                       for k in self.mm_features}

        au_loss = 0.0

        if self._fusion_method == 'dta':
            seq_fused, _ = fuse_seq_dta(
                self.seq_fusion_layer, self._seq_sim_fusion,
                feats_p, feats_seq_p, self.mm_features, self.seq_len)
            feats_seq_pool_fusion = self.seq_fusion_projector(seq_fused)
            feats_fusion = feats_p['id'] if 'id' in feats_p else next(iter(feats_p.values()))

        elif self._fusion_method == 'gmmf':
            user_vec_tmp = torch.cat([feats_p[k] for k in self.mm_features], dim=-1)
            seq_fused, gmmf_loss = fuse_seq_gmmf(
                self.seq_fusion_layer, self._seq_gmmf_pooling,
                feats_p, feats_seq_p, user_vec_tmp)
            feats_seq_pool_fusion = self.seq_fusion_projector(seq_fused)
            au_loss = au_loss + gmmf_loss
            feats_fusion = feats_p['id'] if 'id' in feats_p else next(iter(feats_p.values()))

        elif self._fusion_method == 'dmf':
            seq_fused, _ = fuse_seq_dmf(
                self.seq_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            feats_seq_pool_fusion = self.seq_fusion_projector(seq_fused)
            feats_fusion = feats_p['id'] if 'id' in feats_p else next(iter(feats_p.values()))

        else:
            # 本地融合：每模态先经 feature_mixer 精炼，再每时间步融合，再过 fusion_mixer
            mixed_per_modal = {
                k: self.feature_mixers[k](feats_seq_p[k])
                for k in self.mm_features
            }  # 每个为 (B, L, D)

            seq_fused_steps = []
            cl_total = 0.0
            for i in range(self.seq_len):
                step_dict = {k: mixed_per_modal[k][:, i] for k in self.mm_features}
                fused_i, cl_i = fuse_local(self.seq_fusion_layer, step_dict,
                                           self.mm_features, self._fusion_method,
                                           self._seq_query_num)
                fused_i = self.seq_fusion_projector(fused_i)
                seq_fused_steps.append(fused_i)
                if isinstance(cl_i, torch.Tensor):
                    cl_total = cl_total + cl_i
            seq_fused_seq = torch.stack(seq_fused_steps, dim=1)  # (B, L, projection_dim)

            # 跨时间 Fusion Mixer
            seq_mixed = self.fusion_mixer(seq_fused_seq)
            feats_seq_pool_fusion = self._last_hidden(seq_mixed, seq_id_raw)

            # 目标侧融合
            target_dict = {k: self._squeeze_item(feats_p[k]) for k in self.mm_features}
            target_fused, cl_t = fuse_local(self.target_fusion_layer, target_dict,
                                            self.mm_features, self._fusion_method,
                                            self._target_query_num)
            target_fused = self.target_fusion_projector(target_fused)
            feats_fusion = self.target_projector_post(target_fused)
            if isinstance(cl_t, torch.Tensor):
                cl_total = cl_total + cl_t
            if isinstance(cl_total, torch.Tensor):
                au_loss = au_loss + cl_total

        pred_inputs = [
            feats_seq_pool_fusion,
            feats_fusion,
            feats_seq_pool_fusion * feats_fusion,
        ]

        user_vec = self._encode_user_features(user_feats)
        if user_vec is not None:
            pred_inputs.append(user_vec)

        x_dnn = torch.cat(pred_inputs, dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
