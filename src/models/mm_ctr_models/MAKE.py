from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers import modal_fusion, seq_pooling
import torch
from ..layers.common import SimTier, MultiLayerPerceptron


class MAKE(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.tier_num = model_config.get('tier_num', 10)
        self.simtier_modules = SimTier(self.tier_num)
        self.modal_fusion_method = model_config.get('modal_fusion_method', 'cat')
        self.modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim, self.mm_features)
        self.DIN = seq_pooling.get_pooling('din', dim=self.modal_fusion.getDim(), dropout=self.dropout
                                           , mlp_dims=self.mlp_dims)
        self.dnn = MultiLayerPerceptron(
            self.modal_fusion.getDim() * 2 + self.user_features_num * self.projection_dim + self.tier_num,
            self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1], [1], self.dropout, use_bn=self.bn,activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        feats_fusion = self.modal_fusion(feats_p)
        feats_seq_fusion = torch.stack(
            [self.modal_fusion({k: feats_seq_p[k][:, i] for k in self.mm_features})
             for i in range(self.seq_len)],
            dim=1
        )
        combined_sims = torch.cosine_similarity(
            feats_fusion.unsqueeze(1),  # (B, 1, projection_dim)
            feats_seq_fusion,  # (B, seq_num, projection_dim)
            dim=2
        )
        #print(f"feats_fusion_shape: {feats_fusion.shape},feats_seq_fusion_shape: {feats_seq_fusion.shape}")
        feats_seq_fusion_pool = self.DIN(feats_fusion, feats_seq_fusion)

        mm_simtier = self.simtier_modules(combined_sims)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        vec = torch.cat([user_vec, feats_seq_fusion_pool, feats_fusion, mm_simtier], dim=1)  # [B, dim]
        dnn_out = self.dnn(vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit}
        return out
