"""
NAML 模型 —— 融合分析版本
原始模型硬编码使用 MAF 融合，此版本支持所有融合方法。

模型核心操作（保持不变）:
  - UserEncoder（用户注意力编码器）
  - 目标物品-用户兴趣点积预测

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    通过 model_config['modal_fusion_method'] 指定，逐时间步融合序列特征。
  序列感知融合（dta, gmmf, dmf）:
    替代融合+序列池化步骤，输出经投影对齐后直接作为用户兴趣。
"""
from models.base_seq_model import BaseSeqModel
from models.layers import modal_fusion
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_fusion_kwargs(method, model_config, projection_dim):
    """根据融合方法构建额外参数（仅用于不走 _fusion_helper 的旧路径，保留兼容）"""
    kwargs = {}
    if method == 'lmf':
        kwargs['rank'] = model_config.get('rank', 5)
        kwargs['output_dim'] = model_config.get('fusion_dim', projection_dim)
    elif method == 'mtfn':
        kwargs['rank'] = model_config.get('rank', 20)
    elif method == 'src':
        kwargs['T'] = model_config.get('T', 10)
    return kwargs


class UserEncoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.Wn = torch.nn.Linear(dim, dim, bias=True)
        self.bn = torch.nn.Parameter(torch.randn(dim))
        self.qn = torch.nn.Parameter(torch.randn(dim))

    def forward(self, modal_seq):
        transformed = torch.tanh(self.Wn(modal_seq) + self.bn)
        attention_scores = torch.einsum('d,bnd->bn', self.qn, transformed)
        attention_weights = F.softmax(attention_scores, dim=-1)
        user_embedding = torch.sum(attention_weights.unsqueeze(-1) * modal_seq, dim=1)
        return user_embedding


class NAML(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'maf'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        fusion_out_dim = f_info['out_dim']

        # 融合输出 → projection_dim，保证与 UserEncoder 和点积兼容
        self.fusion_projector = (nn.Linear(fusion_out_dim, self.projection_dim)
                                 if fusion_out_dim != self.projection_dim
                                 else nn.Identity())

        self.user_linear = torch.nn.Linear(self.projection_dim * self.user_features_num, self.projection_dim)

        if not self._is_seq_fusion:
            self.user_encoder = UserEncoder(self.projection_dim)

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec_raw = torch.cat(user_tensors, dim=1)
        user_vec = self.user_linear(user_vec_raw)

        au_loss = 0.0

        if self._fusion_method == 'dta':
            # DTA：序列感知融合，输出直接作为用户兴趣
            user_interest, _ = fuse_seq_dta(
                self.modal_fusion_layer, self._sim_fusion,
                feats_p, feats_seq_p, self.mm_features, self.seq_len)
            user_interest = self.fusion_projector(user_interest)
            feats_fusion = feats_p['id']  # 用 ID 嵌入作为目标表示
            user_vec = torch.sum(torch.stack([user_vec, user_interest], dim=1), dim=1)
            logit = torch.einsum("bd,bd->b", feats_fusion, user_vec).unsqueeze(-1)

        elif self._fusion_method == 'gmmf':
            fused, gmmf_loss = fuse_seq_gmmf(
                self.modal_fusion_layer, self._gmmf_pooling,
                feats_p, feats_seq_p, user_vec_raw)
            fused = self.fusion_projector(fused)
            au_loss = gmmf_loss
            feats_fusion = feats_p['id']
            user_vec = torch.sum(torch.stack([user_vec, fused], dim=1), dim=1)
            logit = torch.einsum("bd,bd->b", feats_fusion, user_vec).unsqueeze(-1)

        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(
                self.modal_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            fused = self.fusion_projector(fused)
            feats_fusion = feats_p['id']
            user_vec = torch.sum(torch.stack([user_vec, fused], dim=1), dim=1)
            logit = torch.einsum("bd,bd->b", feats_fusion, user_vec).unsqueeze(-1)

        else:
            # 本地融合
            feats_fusion, cl1 = fuse_local(
                self.modal_fusion_layer, feats_p, self.mm_features,
                self._fusion_method, self._query_num)
            feats_fusion = self.fusion_projector(feats_fusion)

            feats_seq_fusion = torch.stack(
                [self.fusion_projector(fuse_local(
                    self.modal_fusion_layer,
                    {k: feats_seq_p[k][:, i] for k in self.mm_features},
                    self.mm_features, self._fusion_method, self._query_num)[0])
                 for i in range(self.seq_len)],
                dim=1,
            )  # (B, seq_len, projection_dim)

            user_interest = self.user_encoder(feats_seq_fusion)
            au_loss = cl1
            user_vec = torch.sum(torch.stack([user_vec, user_interest], dim=1), dim=1)
            logit = torch.einsum("bd,bd->b", feats_fusion, user_vec).unsqueeze(-1)

        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
