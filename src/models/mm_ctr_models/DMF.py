from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron, SimTier
from ..layers import modal_fusion, seq_pooling
import torch
class DMF(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.tier_num = model_config.get('tier_num', 10)
        self.attention_dim = model_config.get('attention_dim', 128)
        self.num_buckets = model_config.get('num_buckets', 35)
        self.alpha = model_config.get('alpha', 0.5)
        self.modal_fusion_method = model_config.get('modal_fusion_method', 'cat')
        self.modal_poolings = seq_pooling.get_pooling('din', dim=self.projection_dim, mlp_dims=self.mlp_dims)
        self.modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                          [k for k in self.mm_features if k != 'id'])
        self.DTA = modal_fusion.get_fusion_layer('dta', dim=self.projection_dim, attention_dim=self.attention_dim,
                                                 num_buckets=self.num_buckets, dropout=self.dropout)

        self.simtier_module = SimTier(self.tier_num)

        # MLP处理DTA输出
        self.me_mlp = MultiLayerPerceptron(
            self.attention_dim,
            self.mlp_dims,
            dropout=self.dropout,
            use_bn=self.bn
        )

        # MLP用于模态中心表示
        self.mc_mlp = MultiLayerPerceptron(
            self.tier_num,
            self.mlp_dims,
            dropout=self.dropout,
            use_bn=self.bn
        )

        final_input_dim = self.projection_dim * self.user_features_num + self.mlp_dims[-1]
        self.dnn = MultiLayerPerceptron(final_input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
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

        feats_fusion = self.modal_fusion({k: feats_p[k] for k in self.mm_features if k != 'id'})
        feats_seq_fusion = torch.stack(
            [self.modal_fusion({k: feats_seq_p[k][:, i] for k in self.mm_features if k != 'id'})
             for i in range(self.seq_len)],
            dim=1
        )  # (B,seq_num,fusion_dim)

        # 计算相似度
        combined_sim = torch.cosine_similarity(
            feats_fusion.unsqueeze(1),  # (B, 1, projection_dim)
            feats_seq_fusion,  # (B, seq_num, projection_dim)
            dim=2
        )  # (B, seq_num)

        # DTA 模态增强分支
        r_me = self.DTA(feats_p['id'], feats_seq_p['id'], combined_sim)  # (B, attention_dim)
        r_me = self.me_mlp(r_me)  # (B, emb_dims[-1])

        # SimTier 模态中心分支
        tier_output = self.simtier_module(combined_sim)  # (B, tier_num)

        r_mc = self.mc_mlp(tier_output)  # (B, emb_dims[-1])
        # 互补模态建模
        user_interest = self.alpha * r_me + (1 - self.alpha) * r_mc  # (B, emb_dims[-1])

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        final_vec = torch.cat([user_vec, user_interest], dim=1)

        dnn_out_put = self.dnn(final_vec)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out

