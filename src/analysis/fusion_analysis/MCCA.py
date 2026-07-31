"""
MCCA 模型 —— 融合分析版本

原始 MCCA 依赖预训练 PSRQ pre-model 将多模态特征量化为离散码字（含每模态码本
与联合 joint 码本），查表得到码本嵌入后使用跨模态注意力池化 + cat 拼接各模态。
本版本支持所有融合方法：

适配方式（按用户需求）:
  1. 获取码本中最近的向量：PSRQ 编码 + 多层码本嵌入查表，得到每模态码本表征及 joint 表征
  2. mean pooling：序列侧码本表征经均值池化得到每模态多模态表征
  3. 可配置融合：对目标侧 / 序列侧的每模态表征应用可配置融合方法
       本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
         - 目标侧：对目标物品各模态码本表征融合
         - 序列侧：序列码本表征 mean pooling 后融合
       序列感知融合（dta, gmmf, dmf）:
         - 序列侧：直接对目标 + 序列码本表征做序列感知融合
         - 目标侧：使用目标 ID 码本嵌入
  4. 与原结构融合：融合后的表征与原始 joint 联合码本表征拼接后送入 DNN 输出
"""
from pathlib import Path

import torch
import torch.nn as nn

from models.base_seq_model import BaseSeqModel
from models.layers import seq_pooling
from models.layers.common import MultiLayerPerceptron, FeatureEmbedding
from models.pre_models.PSRQ import PSRQ_Premodel as PSRQ
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    SEQ_FUSIONS, normalize_fusion_method,
)


class MCCA(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        # 融合方法需在数据集 narrowing 前从顶层配置读取
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'cat'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 优先使用数据集特定配置，缺省时回退到顶层配置（与原模型保持一致）
        model_config = model_config.get(data_config['name'], model_config)

        self.ckpt_path = Path(self.train_config['checkpoint_dir']) / \
            f"{self.data_config['name']}_{self.model_config.get('model_name', 'mcca')}.pt"
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.n_levels = model_config.get('n_levels', 3)
        self.codebook_size = model_config.get('codebook_size', 1024)
        self.num_emb_list = [self.codebook_size for _ in range(self.n_levels)]
        self.rq_features = [k for k in self.mm_features if k != 'id']

        # ── PSRQ 预训练量化器 ────────────────────────────────────
        self.PSRQ = PSRQ(model_config, train_config, data_config)
        self.PSRQ.load()
        self.PSRQ.to(device=self.device)
        self.PSRQ.train(False)

        # ── PSRQ 码本嵌入（获取码本中最近的向量后查表）────────────
        self.modal_embeddings = nn.ModuleDict({
            m: nn.ModuleList([
                FeatureEmbedding(self.num_emb_list[0], self.latent_dim)
                for _ in range(self.n_levels)
            ])
            for m in self.rq_features
        })
        self.joint_embedding = nn.ModuleList([
            FeatureEmbedding(self.num_emb_list[0], self.latent_dim)
            for _ in range(self.n_levels)
        ])

        # 重写模态维度（码本表征 = n_levels × latent_dim）
        for k in self.rq_features:
            self.mm_dims[k] = int(self.latent_dim * self.n_levels)

        self.mm_projector = nn.ModuleDict({
            k: nn.Linear(self.mm_dims[k], self.projection_dim)
            for k in self.mm_features
        })
        self.joint_projector = nn.Linear(int(self.latent_dim * self.n_levels), self.projection_dim)

        # 序列侧均值池化
        self.pooling = seq_pooling.get_pooling('mean')

        # ── 可配置融合层 ──────────────────────────────────────────
        # GMMF 门控的用户向量维度需与实际拼接的 user_vec 维度对齐
        if 'gmmf_user_dim' not in model_config:
            model_config['gmmf_user_dim'] = self.projection_dim * self.user_features_num

        f_user = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.user_fusion_layer = f_user['layer']
        self._user_query_num = f_user['query_num']
        self._user_sim_fusion = f_user.get('sim_fusion')
        self._user_gmmf_pooling = f_user.get('seq_pooling')
        user_fusion_out = f_user['out_dim']
        self.user_fusion_projector = (nn.Linear(user_fusion_out, self.projection_dim)
                                      if user_fusion_out != self.projection_dim
                                      else nn.Identity())

        if not self._is_seq_fusion:
            f_target = build_fusion(self._fusion_method, self.projection_dim,
                                    self.mm_features, model_config)
            self.target_fusion_layer = f_target['layer']
            self._target_query_num = f_target['query_num']
            target_fusion_out = f_target['out_dim']
            self.target_fusion_projector = (nn.Linear(target_fusion_out, self.projection_dim)
                                            if target_fusion_out != self.projection_dim
                                            else nn.Identity())
        else:
            self.target_fusion_layer = None
            self._target_query_num = None
            self.target_fusion_projector = nn.Identity()

        # ── DNN（输入 = [序列融合, 目标融合, joint 联合码本, 用户特征]）──
        # joint 联合码本表征即原始 MCCA 的核心结构，与融合结果一同送入 DNN
        self.input_dim = self.projection_dim * (self.user_features_num + 3)
        self.dnn = MultiLayerPerceptron(
            self.input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1], [1], self.dropout, use_bn=self.bn, activation=None)

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def model_to_device(self):
        """重写 model_to_device 以同时移动 PSRQ"""
        super().model_to_device()
        self.PSRQ.to(device=self.device)

    def psrq_encode(self, feats, feats_seq):
        feats_id, joint_id, feats_seq_id = self.PSRQ.get_encode(feats, feats_seq)
        return feats_id, joint_id, feats_seq_id

    def forward(self, user_feats, feats, feats_seq):
        user_feats = dict(user_feats)
        feats = dict(feats)
        feats_seq = dict(feats_seq)

        feats['id'] = self.embedding(feats['id']).squeeze()        # (B, latent_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])          # (B, seq_num, latent_dim)

        # 1) 获取码本中最近的向量（PSRQ 编码索引）
        feats_id, joint_id, feats_seq_id = self.psrq_encode(feats, feats_seq)

        # 多层码本嵌入查表
        for m in self.rq_features:
            emb, seq_emb = [], []
            for i in range(self.n_levels):
                seq_emb.append(self.modal_embeddings[m][i](feats_seq_id[m][:, :, i]))
                emb.append(self.modal_embeddings[m][i](feats_id[m][:, i]))
            feats[m] = torch.cat(emb, dim=-1)                      # (B, n_levels*latent_dim)
            feats_seq[m] = torch.cat(seq_emb, dim=-1)              # (B, seq_num, n_levels*latent_dim)

        emb = []
        for i in range(self.n_levels):
            emb.append(self.joint_embedding[i](joint_id[:, i]))
        joint_id = torch.cat(emb, dim=-1)                          # (B, n_levels*latent_dim)

        # 2) 投影到 projection_dim
        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}
        joint_id_p = self.joint_projector(joint_id)                # (B, projection_dim)

        # 用户侧特征
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_vec = torch.cat([user_feats[k] for k in user_feats], dim=-1)

        seq_len = feats_seq_p[self.mm_features[0]].size(1)
        au_loss = 0.0

        # 3) 可配置融合
        if self._fusion_method == 'dta':
            user_fused, _ = fuse_seq_dta(
                self.user_fusion_layer, self._user_sim_fusion,
                feats_p, feats_seq_p, self.mm_features, seq_len)
            target_fused = feats_p['id']
        elif self._fusion_method == 'gmmf':
            user_fused, gmmf_loss = fuse_seq_gmmf(
                self.user_fusion_layer, self._user_gmmf_pooling,
                feats_p, feats_seq_p, user_vec)
            au_loss = au_loss + gmmf_loss
            target_fused = feats_p['id']
        elif self._fusion_method == 'dmf':
            user_fused, _ = fuse_seq_dmf(
                self.user_fusion_layer, feats_p, feats_seq_p, seq_len)
            target_fused = feats_p['id']
        else:
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_features}
            user_fused, cl1 = fuse_local(
                self.user_fusion_layer, feats_seq_pool, self.mm_features,
                self._fusion_method, self._user_query_num)
            target_fused, cl2 = fuse_local(
                self.target_fusion_layer, feats_p, self.mm_features,
                self._fusion_method, self._target_query_num)
            target_fused = self.target_fusion_projector(target_fused)
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1
            if isinstance(cl2, torch.Tensor):
                au_loss = au_loss + cl2

        user_fused = self.user_fusion_projector(user_fused)

        # 4) 与原始 joint 联合码本表征融合
        x_dnn = torch.cat([user_fused, target_fused, joint_id_p, user_vec], dim=-1)
        dnn_out = self.dnn(x_dnn)
        logit = self.out_put(dnn_out)

        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
