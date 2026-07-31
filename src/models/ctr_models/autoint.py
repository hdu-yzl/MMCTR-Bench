from ..base_model import BaseModel
from ..layers.common import MultiLayerPerceptron, MultiHeadSelfAttention
from ..layers import seq_pooling
import torch


class AutoInt(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.attn_layers_num = model_config.get('attn_layers', 3)
        self.attn_heads = model_config.get('attn_heads', 2)
        self.attn_dim = model_config.get('attn_size', 64)
        self.attn_use_residual = model_config.get('attn_use_residual', True)

        self.pooling = seq_pooling.get_pooling('mean', dim=1)

        # 特征域：id_p（user+item）与序列池化向量
        self.num_fields = 2

        layers = []
        embed_dim = self.projection_dim
        for _ in range(self.attn_layers_num):
            layers.append(MultiHeadSelfAttention(embed_dim, self.attn_dim,
                                                 num_heads=self.attn_heads,
                                                 use_residual=self.attn_use_residual))
            embed_dim = self.attn_dim * self.attn_heads
        self.attn = torch.nn.ModuleList(layers)
        self.attn_output_dim = self.num_fields * embed_dim

        self.dnn = MultiLayerPerceptron(self.projection_dim * self.num_fields,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1] + self.attn_output_dim,
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)  # (B,2*latent_dim) userid,itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        id_p = self.mm_projector['id'](feats['id'])  # (B,projection_dim)

        id_seq_p = self.mm_seq_projector['id'](feats_seq['id'])  # (B,seq_num,projection_dim)
        id_seq_pool = self.pooling(id_seq_p)  # (B,projection_dim)

        # 显式特征交互：多层多头自注意力
        fields = torch.stack([id_p, id_seq_pool], dim=1)  # (B,num_fields,projection_dim)
        attn_out = fields
        for layer in self.attn:
            attn_out = layer(attn_out)
        attn_out = attn_out.flatten(start_dim=1)  # (B,num_fields*embed_dim)

        # 隐式特征交互：DNN
        x_dnn = torch.cat([id_p, id_seq_pool], dim=-1)  # (B,2*projection_dim)
        dnn_out_put = self.dnn(x_dnn)

        comb = torch.cat([dnn_out_put, attn_out], dim=-1)
        logit = self.out_put(comb)
        out = {'pred': logit}
        return out
