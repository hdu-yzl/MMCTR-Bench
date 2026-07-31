from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron,CrossNetwork
from ..layers import  seq_pooling
import torch


class DCN(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.cross_num = model_config.get('cross_num', 3)

        self.input_dim = self.projection_dim*2
        self.cross = CrossNetwork(self.input_dim, self.cross_num)

        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.input_dim,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')

        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1] + self.input_dim,
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

        x_dnn = torch.cat([id_p, id_seq_pool], dim=-1)
        cross_out = self.cross(x_dnn)
        dnn_out_put = self.dnn(x_dnn)
        comb = torch.cat([cross_out, dnn_out_put], dim=-1)

        logit = self.out_put(comb)
        out = {'pred': logit}
        return out
