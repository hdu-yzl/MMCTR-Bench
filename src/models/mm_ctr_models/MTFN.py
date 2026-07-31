from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import modal_fusion, seq_pooling
import torch


class MTFN(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.rank = model_config.get('rank', 20)
        self.modal_fusion = modal_fusion.get_fusion_layer('mtfn', self.projection_dim,
                                                          self.mm_features, rank=self.rank)
        self.seq_model_fusion = modal_fusion.get_fusion_layer('mtfn', self.projection_dim,
                                                              self.mm_seq_features, rank=self.rank)
        self.pooling = seq_pooling.get_pooling('mean', dim=1)
        self.dnn = MultiLayerPerceptron(self.modal_fusion.getDim() + self.seq_model_fusion.getDim(),
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

        feats_seq_fusion = torch.stack(
            [self.seq_model_fusion({k: feats_seq_p[k][:, i] for k in self.mm_seq_features})
             for i in range(self.seq_len)],
            dim=1
        )  # (B,seq_num,fusion_dim)

        feats_seq_fusion_pool = self.pooling(feats_seq_fusion)  # (B,fusion_dim)

        feats_fusion = self.modal_fusion(feats_p)

        x_dnn = torch.cat([feats_fusion, feats_seq_fusion_pool], dim=-1)

        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out
