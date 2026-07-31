"""
M3SRec 模型 —— 融合分析版本
原始模型硬编码使用 _ModalAttentionFusion（MAF 变体）做模态融合，
此版本支持所有融合方法。

模型核心操作（保持不变）:
  - 位置嵌入 + 模态嵌入
  - 输入共享自注意力（input_shared_attn）
  - 模态专有 MoE 层（specific_moe）
  - 跨模态自注意力 + MoE 层（cross_attn / cross_moe）
  - 用户特征拼接（user features）

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    - 用户侧：对每个模态最后有效时间步嵌入应用本地融合
    - 目标侧：对目标模态投影特征应用本地融合
  序列感知融合（dta, gmmf, dmf）:
    - 替代“最后有效时间步 + 模态注意力融合”步骤
    - 直接融合目标特征 + 精炼后的序列特征 → 用户兴趣
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)

import torch
import torch.nn as nn


class _FeedForwardExpert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class _MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, dropout=0.0):
        super().__init__()
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            _FeedForwardExpert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        gate = torch.softmax(self.router(x), dim=-1)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        out = torch.sum(gate.unsqueeze(-1) * expert_outs, dim=-2)
        return self.norm(x + self.dropout(out))


class _SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_dim=None, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim or dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim or dim * 4, dim),
        )

    def forward(self, x, key_padding_mask=None):
        attn_mask = key_padding_mask
        all_padding = None
        if key_padding_mask is not None:
            all_padding = key_padding_mask.all(dim=1)
            if all_padding.any():
                attn_mask = key_padding_mask.clone()
                attn_mask[all_padding] = False

        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask, need_weights=False)
        if all_padding is not None and all_padding.any():
            attn_out = attn_out.masked_fill(all_padding.view(-1, 1, 1), 0.0)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ffn(x)))


class M3SRec(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.dim = self.projection_dim
        self.num_heads = model_config.get('num_heads', 4)
        self.num_experts = model_config.get('num_experts', 4)
        self.moe_hidden_dim = model_config.get('moe_hidden_dim', self.dim * 4)
        self.attn_ffn_dim = model_config.get('attn_ffn_dim', self.dim * 4)
        self.num_specific_layers = model_config.get('num_specific_layers', 1)
        self.num_cross_layers = model_config.get('num_cross_layers', 1)
        self.max_seq_len = model_config.get('max_seq_len', self.seq_len)
        self.m3_features = self._resolve_m3_features()
        self.m3_nums = len(self.m3_features)

        self.position_embedding = nn.Embedding(self.max_seq_len, self.dim)
        self.modality_embedding = nn.Embedding(self.m3_nums, self.dim)
        self.input_shared_attn = _SelfAttentionBlock(self.dim, self.num_heads, self.attn_ffn_dim, self.dropout)
        self.specific_moe = nn.ModuleDict({
            m: nn.ModuleList([
                _MoELayer(self.dim, self.moe_hidden_dim, self.num_experts, self.dropout)
                for _ in range(self.num_specific_layers)
            ])
            for m in self.m3_features
        })
        self.cross_attn_layers = nn.ModuleList([
            _SelfAttentionBlock(self.dim, self.num_heads, self.attn_ffn_dim, self.dropout)
            for _ in range(self.num_cross_layers)
        ])
        self.cross_moe_layers = nn.ModuleList([
            _MoELayer(self.dim, self.moe_hidden_dim, self.num_experts, self.dropout)
            for _ in range(self.num_cross_layers)
        ])

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'maf'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 用户侧融合（对最后有效时间步聚合后的模态向量进行融合）
        f_info_user = build_fusion(self._fusion_method, self.dim,
                                   self.m3_features, model_config)
        self.user_fusion_layer = f_info_user['layer']
        self._user_query_num = f_info_user['query_num']
        self._user_sim_fusion = f_info_user.get('sim_fusion')
        self._user_gmmf_pooling = f_info_user.get('seq_pooling')
        user_fusion_out = f_info_user['out_dim']
        self.user_fusion_projector = (nn.Linear(user_fusion_out, self.dim)
                                      if user_fusion_out != self.dim
                                      else nn.Identity())

        # 目标侧融合（仅本地融合：序列感知融合不会用到目标融合层，目标向量由 ID 嵌入代替）
        if not self._is_seq_fusion:
            f_info_target = build_fusion(self._fusion_method, self.dim,
                                         self.m3_features, model_config)
            self.target_fusion_layer = f_info_target['layer']
            self._target_query_num = f_info_target['query_num']
            target_fusion_out = f_info_target['out_dim']
            self.target_fusion_projector = (nn.Linear(target_fusion_out, self.dim)
                                            if target_fusion_out != self.dim
                                            else nn.Identity())
        else:
            self.target_fusion_layer = None
            self._target_query_num = None
            self.target_fusion_projector = nn.Identity()

        self.dnn = MultiLayerPerceptron(
            self.dim * 2 + self.dim * len(self.user_features),
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

    def _resolve_m3_features(self):
        m3_features = self.model_config.get('m3_modalities')
        m3_features = list(m3_features) if m3_features is not None else list(self.mm_features)
        m3_features = list(dict.fromkeys(m3_features))
        missing = [k for k in m3_features if k not in self.mm_features]
        if not m3_features or missing:
            raise ValueError(f'M3SRec invalid features: {m3_features}, missing={missing}.')
        return m3_features

    @staticmethod
    def _squeeze_item(x):
        return x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x

    def _get_sequence_padding_mask(self, seq_id):
        if seq_id is None:
            return None
        ids = seq_id
        ids = ids.squeeze(-1) if ids.dim() > 2 else ids
        mask = ids.eq(0) if not torch.is_floating_point(ids) else ids.abs().sum(dim=-1).eq(0)
        return mask if mask.dim() == 2 else None

    def _add_position_and_modality_embedding(self, seq_list):
        out = []
        for idx, seq in enumerate(seq_list):
            bsz, seq_len, _ = seq.shape
            if seq_len > self.max_seq_len:
                raise ValueError(f'Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.')
            pos_ids = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(bsz, seq_len)
            modal_ids = torch.full((bsz, seq_len), idx, dtype=torch.long, device=seq.device)
            out.append(seq + self.position_embedding(pos_ids) + self.modality_embedding(modal_ids))
        return out

    def _split_modal_sequence(self, x, seq_len):
        return [x[:, i * seq_len:(i + 1) * seq_len] for i in range(self.m3_nums)]

    @staticmethod
    def _last_valid_token(seq, padding_mask):
        if padding_mask is None:
            return seq[:, -1]
        valid_counts = (~padding_mask).long().sum(dim=1)
        idx = (valid_counts.clamp(min=1) - 1).view(-1, 1, 1)
        out = seq.gather(1, idx.expand(-1, 1, seq.size(-1))).squeeze(1)
        return out.masked_fill(valid_counts.eq(0).view(-1, 1), 0.0)

    def _refine_user_sequence(self, feats_seq_p, seq_id):
        """运行 M3SRec 的核心序列建模流程，返回精炼后的每模态序列字典与 padding mask。"""
        seq_list = [feats_seq_p[k] for k in self.m3_features]
        seq_list = self._add_position_and_modality_embedding(seq_list)
        seq_len = seq_list[0].size(1)
        base_mask = self._get_sequence_padding_mask(seq_id)
        full_mask = torch.cat([base_mask] * self.m3_nums, dim=1) if base_mask is not None else None

        x = self.input_shared_attn(torch.cat(seq_list, dim=1), key_padding_mask=full_mask)
        chunks = self._split_modal_sequence(x, seq_len)
        refined = []
        for name, chunk in zip(self.m3_features, chunks):
            for moe in self.specific_moe[name]:
                chunk = moe(chunk)
            refined.append(chunk)
        x = torch.cat(refined, dim=1)

        for attn, moe in zip(self.cross_attn_layers, self.cross_moe_layers):
            x = moe(attn(x, key_padding_mask=full_mask))

        refined_per_modal = self._split_modal_sequence(x, seq_len)
        refined_dict = {k: chunk for k, chunk in zip(self.m3_features, refined_per_modal)}
        return refined_dict, base_mask

    def _encode_user_features(self, user_feats):
        if not self.user_features:
            return None

        user_vecs = []
        for k in self.user_features:
            raw = self.embedding(user_feats[k].long()) if k == 'id' else user_feats[k]
            user_vecs.append(self.user_projector[k](self._squeeze_item(raw)))
        return torch.cat(user_vecs, dim=-1)

    def forward(self, user_feats, feats, feats_seq):
        user_feats = dict(user_feats)
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        seq_id = feats_seq.get('id')

        feats['id'] = self._squeeze_item(self.embedding(feats['id']))
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # 精炼后的每模态序列字典 (B, L, D)
        refined_seq, base_mask = self._refine_user_sequence(feats_seq_p, seq_id)

        # 仅取目标侧需要的 m3 模态
        target_feats = {k: self._squeeze_item(feats_p[k]) for k in self.m3_features}

        au_loss = 0.0

        # ===== 用户侧融合 =====
        if self._fusion_method == 'dta':
            user_fused, _ = fuse_seq_dta(
                self.user_fusion_layer, self._user_sim_fusion,
                target_feats, refined_seq, self.m3_features, refined_seq[self.m3_features[0]].size(1))
            user_fused = self.user_fusion_projector(user_fused)
            target_fused = target_feats['id'] if 'id' in target_feats else next(iter(target_feats.values()))
        elif self._fusion_method == 'gmmf':
            # 用占位 user_vec：使用模态 id 嵌入或第一个模态向量（user 信息后面拼接）
            user_vec_tmp = torch.cat([target_feats[k] for k in self.m3_features], dim=-1)
            user_fused, gmmf_loss = fuse_seq_gmmf(
                self.user_fusion_layer, self._user_gmmf_pooling,
                target_feats, refined_seq, user_vec_tmp)
            user_fused = self.user_fusion_projector(user_fused)
            au_loss = au_loss + gmmf_loss
            target_fused = target_feats['id'] if 'id' in target_feats else next(iter(target_feats.values()))
        elif self._fusion_method == 'dmf':
            seq_len_ = refined_seq[self.m3_features[0]].size(1)
            user_fused, _ = fuse_seq_dmf(
                self.user_fusion_layer, target_feats, refined_seq, seq_len_)
            user_fused = self.user_fusion_projector(user_fused)
            target_fused = target_feats['id'] if 'id' in target_feats else next(iter(target_feats.values()))
        else:
            # 本地融合：先取每个模态的最后有效时间步，再融合
            modal_user_dict = {
                k: self._last_valid_token(refined_seq[k], base_mask)
                for k in self.m3_features
            }
            user_fused, cl1 = fuse_local(
                self.user_fusion_layer, modal_user_dict, self.m3_features,
                self._fusion_method, self._user_query_num)
            user_fused = self.user_fusion_projector(user_fused)
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1

            # 目标侧本地融合
            target_fused, cl2 = fuse_local(
                self.target_fusion_layer, target_feats, self.m3_features,
                self._fusion_method, self._target_query_num)
            target_fused = self.target_fusion_projector(target_fused)
            if isinstance(cl2, torch.Tensor):
                au_loss = au_loss + cl2

        user_vec = self._encode_user_features(user_feats)
        x_dnn = torch.cat([user_fused, target_fused], dim=-1)
        if user_vec is not None:
            x_dnn = torch.cat([x_dnn, user_vec], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
