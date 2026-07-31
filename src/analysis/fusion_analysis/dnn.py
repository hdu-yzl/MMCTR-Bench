"""
DNN 模型 —— 融合分析版本
DNN_mm 支持本地融合方法（maf, cat, lmf, src, mtfn, fq-former, simcen）。
DNN_mm_seq 支持所有融合方法（含 dta, gmmf, dmf）。
"""
from models.base_model import BaseModel
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron
from models.layers import modal_fusion, seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn


class DNN(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.projection_dim * 2,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()
        self.model_to_device()
        self.log_model_params()

    def forward(self, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])

        id_p = self.mm_projector['id'](feats['id'])
        id_seq_p = self.mm_seq_projector['id'](feats_seq['id'])
        id_seq_pool = self.pooling(id_seq_p)

        x_dnn = torch.cat([id_p, id_seq_pool], dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out


class DNN_mm(BaseModel):
    """支持所有融合方法（含 dta, gmmf, dmf）。
    BaseModel 默认会先池化序列；序列感知融合时跳过池化、直接对原始 3D 序列融合。
    """

    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self._fusion_method = normalize_fusion_method(self.modal_fusion_method)
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        # 序列感知融合所需的公共特征集（target & seq 都必须有）
        self.seq_fusion_features = [k for k in self.mm_features if k in self.mm_seq_features]

        if self._is_seq_fusion:
            f_info = build_fusion(self._fusion_method, self.projection_dim,
                                  self.seq_fusion_features, model_config)
            self.seq_fusion_layer_full = f_info['layer']
            self._seq_query_num_full = f_info['query_num']
            self._sim_fusion = f_info.get('sim_fusion')
            self._gmmf_pooling = f_info.get('seq_pooling')
            seq_full_out = f_info['out_dim']
            self.seq_full_projector = (
                nn.Linear(seq_full_out, self.projection_dim)
                if seq_full_out != self.projection_dim
                else nn.Identity())

            # 目标侧仍使用本地融合（target 是 2D），用 cat 简单对齐
            t_info = build_fusion('cat', self.projection_dim, self.mm_features, model_config)
            self.modal_fusion_layer = t_info['layer']
            self._query_num = t_info['query_num']
            target_out = t_info['out_dim']
            self.target_projector = (nn.Linear(target_out, self.projection_dim)
                                     if target_out != self.projection_dim
                                     else nn.Identity())
            dnn_input_dim = self.projection_dim * 2
        else:
            f_info = build_fusion(self._fusion_method, self.projection_dim,
                                  self.mm_features, model_config)
            self.modal_fusion_layer = f_info['layer']
            self._query_num = f_info['query_num']

            sf_info = build_fusion(self._fusion_method, self.projection_dim,
                                   self.mm_seq_features, model_config)
            self.seq_modal_fusion_layer = sf_info['layer']
            self._seq_query_num = sf_info['query_num']
            dnn_input_dim = f_info['out_dim'] + sf_info['out_dim']

        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(dnn_input_dim,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()
        self.model_to_device()
        self.log_model_params()

    def forward(self, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_seq_projector[k](feats_seq[k]) for k in self.mm_seq_features}

        au_loss = feats_p['id'].new_tensor(0.0)

        if self._is_seq_fusion:
            target_dict = {k: feats_p[k] for k in self.seq_fusion_features}
            seq_dict = {k: feats_seq_p[k] for k in self.seq_fusion_features}

            if self._fusion_method == 'dta':
                fused_seq, _ = fuse_seq_dta(self.seq_fusion_layer_full, self._sim_fusion,
                                            target_dict, seq_dict,
                                            self.seq_fusion_features, self.seq_len)
            elif self._fusion_method == 'gmmf':
                user_vec_tmp = torch.cat([target_dict[k] for k in self.seq_fusion_features], dim=-1)
                fused_seq, gmmf_loss = fuse_seq_gmmf(self.seq_fusion_layer_full, self._gmmf_pooling,
                                                    target_dict, seq_dict, user_vec_tmp)
                au_loss = au_loss + gmmf_loss
            else:  # dmf
                fused_seq, _ = fuse_seq_dmf(self.seq_fusion_layer_full,
                                            target_dict, seq_dict, self.seq_len)

            feats_seq_fusion = self.seq_full_projector(fused_seq)
            target_fused = self.modal_fusion_layer(feats_p)
            target_fused = self.target_projector(target_fused)
            x_dnn = torch.cat([target_fused, feats_seq_fusion], dim=-1)
        else:
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}

            feats_fusion, cl1 = fuse_local(self.modal_fusion_layer, feats_p,
                                           self.mm_features, self._fusion_method, self._query_num)
            feats_seq_fusion, cl2 = fuse_local(self.seq_modal_fusion_layer, feats_seq_pool,
                                               self.mm_seq_features, self._fusion_method, self._seq_query_num)
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1
            if isinstance(cl2, torch.Tensor):
                au_loss = au_loss + cl2
            x_dnn = torch.cat([feats_fusion, feats_seq_fusion], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit, 'au_loss': au_loss}
        return out


class DNN_mm_seq(BaseSeqModel):
    """支持所有融合方法（含 dta, gmmf, dmf）。"""

    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self._fusion_method = normalize_fusion_method(self.modal_fusion_method)
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        fusion_out_dim = f_info['out_dim']

        if self._is_seq_fusion:
            # 序列感知融合替代了 fusion+pooling，只需一个融合层
            self.fusion_projector = (
                nn.Linear(fusion_out_dim, self.projection_dim)
                if fusion_out_dim != self.projection_dim
                else nn.Identity()
            )
            dnn_input_dim = self.projection_dim + self.projection_dim * len(self.user_features)
        else:
            # 本地融合：目标物品 + 序列池化
            sf_info = build_fusion(self._fusion_method, self.projection_dim,
                                   self.mm_features, model_config)
            self.seq_modal_fusion_layer = sf_info['layer']
            self._seq_query_num = sf_info['query_num']
            self.pooling = seq_pooling.get_pooling('mean', dim=1)
            dnn_input_dim = fusion_out_dim * 2 + self.projection_dim * len(self.user_features)

        self.dnn = MultiLayerPerceptron(dnn_input_dim,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
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
                                           self.mm_features, self._fusion_method, self._query_num)
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_features}
            feats_seq_fusion, cl2 = fuse_local(self.seq_modal_fusion_layer, feats_seq_pool,
                                               self.mm_features, self._fusion_method, self._seq_query_num)
            au_loss = cl1 + cl2
            x_dnn = torch.cat([feats_fusion, feats_seq_fusion, user_vec], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        if isinstance(au_loss, torch.Tensor):
            out['au_loss'] = au_loss
        return out
