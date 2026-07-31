from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import seq_pooling
import torch


class DIN(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.attention_mlp_dims = model_config.get('attention_mlp_dims', [64, 32])

        # DIN 注意力池化：以目标 item 为 query 对用户行为序列加权
        self.attention_pooling = seq_pooling.get_pooling('din', self.projection_dim,
                                                         dropout=self.dropout,
                                                         mlp_dims=self.attention_mlp_dims)

        # 输入：用户向量 + 目标 item 向量 + 兴趣向量
        self.dnn = MultiLayerPerceptron(self.projection_dim * 3,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        item_emb = self.embedding(feats['id']).squeeze(1)  # (B,latent_dim) 目标item
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)
        user_emb = self.embedding(user_feats['id']).squeeze(1)  # (B,latent_dim)

        item_p = self.mm_projector['id'](item_emb)          # (B,projection_dim) 作为 query
        seq_p = self.mm_projector['id'](feats_seq['id'])    # (B,seq_num,projection_dim)
        user_p = self.user_projector['id'](user_emb)        # (B,projection_dim)

        # 兴趣表征：目标 item 对历史行为序列的注意力池化
        interest = self.attention_pooling(item_p, seq_p)    # (B,projection_dim)

        x_dnn = torch.cat([user_p, item_p, interest], dim=-1)  # (B,3*projection_dim)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out
