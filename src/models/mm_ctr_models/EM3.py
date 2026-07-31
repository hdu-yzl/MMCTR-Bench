from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers import modal_fusion, seq_pooling
import torch
from ..layers.common import MultiLayerPerceptron
from ..layers.losses import CIC_Loss


class EM3(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.query_num = model_config.get('query_num', 5)
        self.cic_tau = model_config.get('cic_tau', 0.1)
        self.cic_weight = model_config.get('cic_weight', 0.1)
        self.modal_fusion = modal_fusion.get_fusion_layer('fq-former', self.projection_dim, query_num=self.query_num)
        self.DIN = seq_pooling.get_pooling('din', dim=self.modal_fusion.getDim(), dropout=self.dropout
                                           , mlp_dims=self.mlp_dims)
        self.content_map = MultiLayerPerceptron(self.modal_fusion.getDim(), [self.projection_dim], self.dropout,
                                                use_bn=self.bn)
        self.cic = CIC_Loss(self.cic_tau)
        self.dnn = MultiLayerPerceptron(
            self.modal_fusion.getDim() + self.user_features_num * self.projection_dim + self.projection_dim,
            self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1], [1], self.dropout, use_bn=self.bn, activation=None)

        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        target_emb = torch.cat([feats_p[k].unsqueeze(1) for k in feats_p], dim=1)  # (B,mm_num,projection_dim)

        seq_fusion = []
        for i in range(self.seq_len):
            seq_emb = torch.cat([feats_seq_p[k][:, i, :].unsqueeze(1) for k in self.mm_features],
                                dim=1)  # (B,mm_num,1,projection_dim)
            fusion = self.modal_fusion(seq_emb)[:, :self.query_num].view((seq_emb.size(0), -1))
            seq_fusion.append(fusion)
        seq_fusion = torch.stack(seq_fusion, dim=1)  # (B,seq_num,projection_dim*query_num)

        target_fusion = self.modal_fusion(target_emb)[:, :self.query_num].view(
            (target_emb.size(0), -1))  # (B,projection_dim*query_num)
        # print(f"feats_fusion_shape: {feats_fusion.shape},feats_seq_fusion_shape: {feats_seq_fusion.shape}")
        feats_seq_fusion_pool = self.DIN(target_fusion, seq_fusion)  # (B,projection_dim*query_num)

        content_vec = self.content_map(target_fusion)  # (B,projection_dim)
        au_loss = self.cic(content_vec, feats_p['id'])

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        vec = torch.cat([user_vec, feats_seq_fusion_pool, content_vec], dim=1)  # [B, dim]
        dnn_out = self.dnn(vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit, 'au_loss': au_loss * self.cic_weight}
        return out
