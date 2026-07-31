from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron, FactorizationMachine
from ..layers import seq_pooling
import torch


class DeepFM(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.fm = FactorizationMachine()

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

        x_dnn = torch.cat([id_p, id_seq_pool], dim=-1)
        x_input = torch.stack([id_p, id_seq_pool], dim=1)
        fm_output = self.fm(x_input)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put) + fm_output
        out = {'pred': logit}
        return out
