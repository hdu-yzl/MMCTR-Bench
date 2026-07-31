"""
通用融合包装基类 —— 用于将原本"融合即模型"的 LMF / MTFN / GMMF / SimCEN
强制适配为可配置融合的标准 BaseSeqModel 模型。

设计：
  - 不再保留各模型原有的特殊结构（如 GMMF 的 GAN/DSN、SimCEN 的 MLEP）
  - 统一为：模态投影 → 可配置融合（target & seq）→ MLP → 输出
  - 通过子类指定默认融合方法，使得在分析框架中各"模型名"具有不同默认行为
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron
from models.layers import seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn


class FusionWrapperBase(BaseSeqModel):
    """通用融合包装基类，子类需指定 _DEFAULT_FUSION 类变量。"""

    _DEFAULT_FUSION = 'cat'  # 子类覆盖

    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        # 优先使用 model_config 指定的融合方法，否则使用类默认值
        method = model_config.get('modal_fusion_method', None) or self._DEFAULT_FUSION
        self._fusion_method = normalize_fusion_method(method)
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 目标侧（或序列感知融合时唯一）融合层
        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        fusion_out_dim = f_info['out_dim']

        if self._is_seq_fusion:
            # 序列感知融合：替代 fusion + 序列池化，单独投影到 projection_dim
            self.fusion_projector = (
                nn.Linear(fusion_out_dim, self.projection_dim)
                if fusion_out_dim != self.projection_dim
                else nn.Identity()
            )
            dnn_input_dim = self.projection_dim + self.projection_dim * len(self.user_features)
        else:
            # 本地融合：分别融合目标 + 序列池化
            sf_info = build_fusion(self._fusion_method, self.projection_dim,
                                   self.mm_features, model_config)
            self.seq_modal_fusion_layer = sf_info['layer']
            self._seq_query_num = sf_info['query_num']
            self.pooling = seq_pooling.get_pooling('mean', dim=1)
            dnn_input_dim = (fusion_out_dim * 2
                             + self.projection_dim * len(self.user_features))

        self.dnn = MultiLayerPerceptron(dnn_input_dim, self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1], [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        user_feats = dict(user_feats)

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
            fused, _ = fuse_seq_dta(self.modal_fusion_layer, self._sim_fusion,
                                    feats_p, feats_seq_p, self.mm_features, self.seq_len)
            fused = self.fusion_projector(fused)
            x_dnn = torch.cat([fused, user_vec], dim=-1)

        elif self._fusion_method == 'gmmf':
            fused, gmmf_loss = fuse_seq_gmmf(self.modal_fusion_layer, self._gmmf_pooling,
                                             feats_p, feats_seq_p, user_vec)
            fused = self.fusion_projector(fused)
            au_loss = gmmf_loss
            x_dnn = torch.cat([fused, user_vec], dim=-1)

        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(self.modal_fusion_layer, feats_p, feats_seq_p, self.seq_len)
            fused = self.fusion_projector(fused)
            x_dnn = torch.cat([fused, user_vec], dim=-1)

        else:
            # 本地融合
            feats_fusion, cl1 = fuse_local(self.modal_fusion_layer, feats_p,
                                           self.mm_features, self._fusion_method,
                                           self._query_num)
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_features}
            feats_seq_fusion, cl2 = fuse_local(self.seq_modal_fusion_layer, feats_seq_pool,
                                               self.mm_features, self._fusion_method,
                                               self._seq_query_num)
            cl_total = 0.0
            if isinstance(cl1, torch.Tensor):
                cl_total = cl_total + cl1
            if isinstance(cl2, torch.Tensor):
                cl_total = cl_total + cl2
            if isinstance(cl_total, torch.Tensor):
                au_loss = cl_total
            x_dnn = torch.cat([feats_fusion, feats_seq_fusion, user_vec], dim=-1)

        dnn_out = self.dnn(x_dnn)
        logit = self.out_put(dnn_out)
        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
