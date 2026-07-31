"""
DMF 模型 —— 融合分析版本（强制兼容所有融合方法）
原始 DMF 内部已使用 DTA 序列感知机制 + SimTier 双路径，本版本：
  - 本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）：替换非ID模态融合
  - 序列感知融合（dta, gmmf, dmf）：用配置的序列感知融合替代内部 DTA + me_mlp 路径，
    保留 SimTier 路径（基于固定 cat 融合的相似度）作为模型核心贡献
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron, SimTier
from models.layers import modal_fusion, seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn


class DMF(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.tier_num = model_config.get('tier_num', 10)
        self.attention_dim = model_config.get('attention_dim', 128)
        self.num_buckets = model_config.get('num_buckets', 35)
        self.alpha = model_config.get('alpha', 0.5)

        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'cat'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        non_id = [k for k in self.mm_features if k != 'id']
        self._non_id_features = non_id

        # SimTier 路径所需的固定 cat 融合（用于计算 sim 序列）
        self._sim_local_fusion = modal_fusion.get_fusion_layer('cat', self.projection_dim, non_id)

        if self._is_seq_fusion:
            # 序列感知融合：用配置的 seq fusion 替代 DTA + me_mlp 路径
            f_info = build_fusion(self._fusion_method, self.projection_dim,
                                  self.mm_features, model_config)
            self.seq_fusion_layer = f_info['layer']
            self._seq_query_num = f_info['query_num']
            self._sim_fusion_aux = f_info.get('sim_fusion')  # for DTA inside fuse_seq_dta
            self._gmmf_pooling = f_info.get('seq_pooling')
            seq_fusion_out = f_info['out_dim']
            self.r_me_projector = (nn.Linear(seq_fusion_out, self.mlp_dims[-1])
                                   if seq_fusion_out != self.mlp_dims[-1]
                                   else nn.Identity())
            # 本地融合层不再使用，但保留接口
            self.modal_fusion_layer = None
            self._query_num = None
        else:
            # 本地融合
            f_info = build_fusion(self._fusion_method, self.projection_dim, non_id, model_config)
            self.modal_fusion_layer = f_info['layer']
            self._query_num = f_info['query_num']

            self.modal_poolings = seq_pooling.get_pooling('din', dim=self.projection_dim,
                                                          mlp_dims=self.mlp_dims)
            self.DTA = modal_fusion.get_fusion_layer('dta', dim=self.projection_dim,
                                                     attention_dim=self.attention_dim,
                                                     num_buckets=self.num_buckets,
                                                     dropout=self.dropout)
            self.me_mlp = MultiLayerPerceptron(
                self.attention_dim, self.mlp_dims, dropout=self.dropout, use_bn=self.bn)

        self.simtier_module = SimTier(self.tier_num)
        self.mc_mlp = MultiLayerPerceptron(
            self.tier_num, self.mlp_dims, dropout=self.dropout, use_bn=self.bn)

        final_input_dim = self.projection_dim * self.user_features_num + self.mlp_dims[-1]
        self.dnn = MultiLayerPerceptron(final_input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def _fuse_non_id_local(self, feats_dict):
        non_id = self._non_id_features
        non_id_feats = {k: feats_dict[k] for k in non_id}
        fused, cl_loss = fuse_local(self.modal_fusion_layer, non_id_feats,
                                    non_id, self._fusion_method, self._query_num)
        return fused, cl_loss

    def _compute_combined_sim(self, feats_p, feats_seq_p):
        """用固定 cat 融合计算目标 vs 序列的余弦相似度（SimTier 路径）。"""
        non_id = self._non_id_features
        target_mm = self._sim_local_fusion({k: feats_p[k] for k in non_id})  # (B, cat_dim)
        seq_mm = torch.stack(
            [self._sim_local_fusion({k: feats_seq_p[k][:, i] for k in non_id})
             for i in range(self.seq_len)],
            dim=1,
        )  # (B, L, cat_dim)
        return torch.cosine_similarity(target_mm.unsqueeze(1), seq_mm, dim=2)  # (B, L)

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        au_loss = feats_p['id'].new_tensor(0.0)

        # SimTier 路径（始终基于固定 cat 融合）
        combined_sim = self._compute_combined_sim(feats_p, feats_seq_p)
        tier_output = self.simtier_module(combined_sim)
        r_mc = self.mc_mlp(tier_output)

        # ME 路径（取决于融合方法）
        if self._fusion_method == 'dta':
            fused, _ = fuse_seq_dta(self.seq_fusion_layer, self._sim_fusion_aux,
                                    feats_p, feats_seq_p, self.mm_features, self.seq_len)
            r_me = self.r_me_projector(fused)
        elif self._fusion_method == 'gmmf':
            user_vec_tmp = torch.cat([feats_p[k] for k in self.mm_features], dim=-1)
            fused, gmmf_loss = fuse_seq_gmmf(self.seq_fusion_layer, self._gmmf_pooling,
                                             feats_p, feats_seq_p, user_vec_tmp)
            r_me = self.r_me_projector(fused)
            au_loss = au_loss + gmmf_loss
        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(self.seq_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            r_me = self.r_me_projector(fused)
        else:
            # 本地融合：原始 DMF 路径
            feats_fusion, cl1 = self._fuse_non_id_local(feats_p)
            feats_seq_fusion = torch.stack(
                [self._fuse_non_id_local({k: feats_seq_p[k][:, i] for k in self.mm_features})[0]
                 for i in range(self.seq_len)],
                dim=1,
            )
            local_sim = torch.cosine_similarity(feats_fusion.unsqueeze(1), feats_seq_fusion, dim=2)
            r_me_raw = self.DTA(feats_p['id'], feats_seq_p['id'], local_sim)
            r_me = self.me_mlp(r_me_raw)
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1

        user_interest = self.alpha * r_me + (1 - self.alpha) * r_mc

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        final_vec = torch.cat([user_vec, user_interest], dim=1)
        dnn_out_put = self.dnn(final_vec)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit, 'au_loss': au_loss}
        return out
