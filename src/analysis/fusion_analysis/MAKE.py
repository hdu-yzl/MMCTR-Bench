"""
MAKE 模型 —— 融合分析版本
已支持 modal_fusion_method 配置。
新增 fq-former、simcen 本地融合及 dta、gmmf、dmf 序列感知融合支持。
"""
from models.base_seq_model import BaseSeqModel
from models.layers import modal_fusion, seq_pooling
from models.layers.common import SimTier, MultiLayerPerceptron
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn


class MAKE(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.tier_num = model_config.get('tier_num', 10)
        self.simtier_modules = SimTier(self.tier_num)

        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'cat'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        fusion_out_dim = f_info['out_dim']

        if self._is_seq_fusion:
            # 序列感知融合：输出经投影对齐
            self.fusion_projector = (nn.Linear(fusion_out_dim, self.projection_dim)
                                     if fusion_out_dim != self.projection_dim
                                     else nn.Identity())
            # 需要辅助融合层计算 SimTier 的相似度 + 目标物品融合表示
            self._local_fusion_for_sim = modal_fusion.get_fusion_layer(
                'cat', self.projection_dim, self.mm_features)
            sim_fusion_dim = self._local_fusion_for_sim.getDim()
            # fused(projection_dim) + local_f(sim_fusion_dim) + user_vec + tier_num
            dnn_input = (self.projection_dim
                         + sim_fusion_dim
                         + self.user_features_num * self.projection_dim
                         + self.tier_num)
        else:
            dnn_input = (fusion_out_dim * 2
                         + self.user_features_num * self.projection_dim
                         + self.tier_num)

        self.DIN = seq_pooling.get_pooling('din', dim=fusion_out_dim if not self._is_seq_fusion else self.projection_dim,
                                           dropout=self.dropout, mlp_dims=self.mlp_dims)
        self.dnn = MultiLayerPerceptron(dnn_input, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1], [1], self.dropout, use_bn=self.bn, activation=None)
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
        user_vec = torch.cat([user_feats[k] for k in user_feats], dim=1)

        au_loss = 0.0

        if self._fusion_method == 'dta':
            fused, _ = fuse_seq_dta(
                self.modal_fusion_layer, self._sim_fusion,
                feats_p, feats_seq_p, self.mm_features, self.seq_len)
            fused = self.fusion_projector(fused)
            # 用辅助融合计算 SimTier + 目标物品表示
            local_f = self._local_fusion_for_sim(feats_p)
            local_seq = torch.stack(
                [self._local_fusion_for_sim({k: feats_seq_p[k][:, i] for k in self.mm_features})
                 for i in range(self.seq_len)], dim=1)
            sims = torch.cosine_similarity(local_f.unsqueeze(1), local_seq, dim=2)
            mm_simtier = self.simtier_modules(sims)
            vec = torch.cat([user_vec, fused, local_f, mm_simtier], dim=1)

        elif self._fusion_method == 'gmmf':
            fused, gmmf_loss = fuse_seq_gmmf(
                self.modal_fusion_layer, self._gmmf_pooling,
                feats_p, feats_seq_p, user_vec)
            fused = self.fusion_projector(fused)
            au_loss = gmmf_loss
            local_f = self._local_fusion_for_sim(feats_p)
            local_seq = torch.stack(
                [self._local_fusion_for_sim({k: feats_seq_p[k][:, i] for k in self.mm_features})
                 for i in range(self.seq_len)], dim=1)
            sims = torch.cosine_similarity(local_f.unsqueeze(1), local_seq, dim=2)
            mm_simtier = self.simtier_modules(sims)
            vec = torch.cat([user_vec, fused, local_f, mm_simtier], dim=1)

        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(
                self.modal_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            fused = self.fusion_projector(fused)
            local_f = self._local_fusion_for_sim(feats_p)
            local_seq = torch.stack(
                [self._local_fusion_for_sim({k: feats_seq_p[k][:, i] for k in self.mm_features})
                 for i in range(self.seq_len)], dim=1)
            sims = torch.cosine_similarity(local_f.unsqueeze(1), local_seq, dim=2)
            mm_simtier = self.simtier_modules(sims)
            vec = torch.cat([user_vec, fused, local_f, mm_simtier], dim=1)

        else:
            # 本地融合
            feats_fusion, cl1 = fuse_local(
                self.modal_fusion_layer, feats_p, self.mm_features,
                self._fusion_method, self._query_num)
            feats_seq_fusion = torch.stack(
                [fuse_local(self.modal_fusion_layer,
                            {k: feats_seq_p[k][:, i] for k in self.mm_features},
                            self.mm_features, self._fusion_method, self._query_num)[0]
                 for i in range(self.seq_len)],
                dim=1,
            )
            combined_sims = torch.cosine_similarity(
                feats_fusion.unsqueeze(1), feats_seq_fusion, dim=2)
            feats_seq_fusion_pool = self.DIN(feats_fusion, feats_seq_fusion)
            mm_simtier = self.simtier_modules(combined_sims)
            if isinstance(cl1, torch.Tensor):
                au_loss = cl1
            vec = torch.cat([user_vec, feats_seq_fusion_pool, feats_fusion, mm_simtier], dim=1)

        dnn_out = self.dnn(vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
