from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import modal_fusion, seq_pooling
import torch

class DNN(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.projection_dim*2,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)  # (B,2*latent_dim) userid,itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        id_p = self.mm_projector['id'](feats['id'])

        id_seq_p = self.mm_seq_projector['id'](feats_seq['id'])

        id_seq_pool = self.pooling(id_seq_p)

        x_dnn = torch.cat([id_p,id_seq_pool], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out

class DNN_mm(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                          self.mm_features)
        self.seq_modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                              self.mm_seq_features)
        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.modal_fusion.getDim() + self.seq_modal_fusion.getDim(),
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)  # (B,2*latent_dim) userid,itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_seq_projector[k](feats_seq[k]) for k in self.mm_seq_features}

        feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}

        feats_fusion = self.modal_fusion(feats_p)

        feats_seq_pool_fusion = self.seq_modal_fusion(feats_seq_pool)

        x_dnn = torch.cat([feats_fusion, feats_seq_pool_fusion], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out


class DNN_mm_seq(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                          self.mm_features)
        self.seq_modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                              self.mm_features)
        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.modal_fusion.getDim() * 2 + self.projection_dim * len(self.user_features),
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}

        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_features}

        feats_fusion = self.modal_fusion(feats_p)

        feats_seq_pool_fusion = self.seq_modal_fusion(feats_seq_pool)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        x_dnn = torch.cat([feats_fusion, feats_seq_pool_fusion, user_vec], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out
