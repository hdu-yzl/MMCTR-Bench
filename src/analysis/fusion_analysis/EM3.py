"""
EM3 模型 —— 融合分析版本
原始模型硬编码使用 FQ-Former 融合，此版本支持所有融合方法。

模型核心操作（保持不变）:
  - DIN 序列池化
  - CIC_Loss（内容-ID 对比学习）
  - content_map 映射

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    FQ-Former 用堆叠输入，其他用 dict/stack 接口，逐时间步融合序列。
  序列感知融合（dta, gmmf, dmf）:
    替代融合+DIN池化步骤。
"""
from models.base_seq_model import BaseSeqModel
from models.layers import modal_fusion, seq_pooling
from models.layers.common import MultiLayerPerceptron
from models.layers.losses import CIC_Loss
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn


class EM3(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.query_num = model_config.get('query_num', 5)
        self.cic_tau = model_config.get('cic_tau', 0.1)
        self.cic_weight = model_config.get('cic_weight', 0.1)

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'fq-former'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        self.fusion_dim = f_info['out_dim']

        if not self._is_seq_fusion:
            self.DIN = seq_pooling.get_pooling('din', dim=self.fusion_dim, dropout=self.dropout,
                                               mlp_dims=self.mlp_dims)

        if self._is_seq_fusion:
            # 序列感知融合需要辅助本地融合层来计算 content_vec
            self._local_fusion_for_content = modal_fusion.get_fusion_layer(
                'maf', self.projection_dim, self.mm_features)
            content_fusion_dim = self._local_fusion_for_content.getDim()
        else:
            content_fusion_dim = self.fusion_dim

        self.content_map = MultiLayerPerceptron(content_fusion_dim, [self.projection_dim], self.dropout,
                                                use_bn=self.bn)
        self.cic = CIC_Loss(self.cic_tau)

        if self._is_seq_fusion:
            # 序列感知融合：输出维度可能不同
            fusion_proj_dim = self.projection_dim
            self.fusion_projector = (nn.Linear(self.fusion_dim, fusion_proj_dim)
                                     if self.fusion_dim != fusion_proj_dim
                                     else nn.Identity())
            dnn_input = fusion_proj_dim + self.user_features_num * self.projection_dim + self.projection_dim
        else:
            dnn_input = self.fusion_dim + self.user_features_num * self.projection_dim + self.projection_dim

        self.dnn = MultiLayerPerceptron(dnn_input, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1], [1], self.dropout,
                                            use_bn=self.bn, activation=None)

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def _fuse_features(self, feats_dict):
        """统一融合接口：根据融合方法调用不同的前向逻辑"""
        return fuse_local(self.modal_fusion_layer, feats_dict, self.mm_features,
                          self._fusion_method, self._query_num)

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_vec = torch.cat([user_feats[k] for k in user_feats], dim=1)

        au_loss = 0.0

        if self._fusion_method == 'dta':
            fused, _ = fuse_seq_dta(
                self.modal_fusion_layer, self._sim_fusion,
                feats_p, feats_seq_p, self.mm_features, self.seq_len)
            fused = self.fusion_projector(fused)
            # 用辅助本地融合层计算 content_vec（DTA 层不支持 dict 输入）
            local_fused = self._local_fusion_for_content(feats_p)
            content_vec = self.content_map(local_fused)
            au_loss = self.cic(content_vec, feats_p['id']) * self.cic_weight
            vec = torch.cat([user_vec, fused, content_vec], dim=1)

        elif self._fusion_method == 'gmmf':
            fused, gmmf_loss = fuse_seq_gmmf(
                self.modal_fusion_layer, self._gmmf_pooling,
                feats_p, feats_seq_p, user_vec)
            fused = self.fusion_projector(fused)
            au_loss = gmmf_loss
            # 用辅助本地融合层计算 content_vec
            local_fused = self._local_fusion_for_content(feats_p)
            content_vec = self.content_map(local_fused)
            au_loss = au_loss + self.cic(content_vec, feats_p['id']) * self.cic_weight
            vec = torch.cat([user_vec, fused, content_vec], dim=1)

        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(
                self.modal_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            fused = self.fusion_projector(fused)
            # 用辅助本地融合层计算 content_vec
            local_fused = self._local_fusion_for_content(feats_p)
            content_vec = self.content_map(local_fused)
            au_loss = self.cic(content_vec, feats_p['id']) * self.cic_weight
            vec = torch.cat([user_vec, fused, content_vec], dim=1)

        else:
            # 本地融合（含 fq-former 和 simcen）
            target_fusion, cl1 = self._fuse_features(feats_p)
            seq_fusion_list = []
            for i in range(self.seq_len):
                step_feats = {k: feats_seq_p[k][:, i, :] for k in self.mm_features}
                sf, _ = self._fuse_features(step_feats)
                seq_fusion_list.append(sf)
            seq_fusion = torch.stack(seq_fusion_list, dim=1)

            feats_seq_fusion_pool = self.DIN(target_fusion, seq_fusion)
            content_vec = self.content_map(target_fusion)
            cic_loss = self.cic(content_vec, feats_p['id'])
            au_loss = cic_loss * self.cic_weight
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1

            vec = torch.cat([user_vec, feats_seq_fusion_pool, content_vec], dim=1)

        dnn_out = self.dnn(vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit, 'au_loss': au_loss}
        return out
