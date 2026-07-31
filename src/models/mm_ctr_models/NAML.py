from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers import modal_fusion
import torch
import torch.nn.functional as F


class UserEncoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.Wn = torch.nn.Linear(dim, dim, bias=True)
        self.bn = torch.nn.Parameter(torch.randn(dim))
        self.qn = torch.nn.Parameter(torch.randn(dim))

    def forward(self, modal_seq):
        transformed = torch.tanh(self.Wn(modal_seq) + self.bn)  # [B, N, dim]

        attention_scores = torch.einsum('d,bnd->bn', self.qn, transformed)  # [B, N]

        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, N]

        user_embedding = torch.sum(attention_weights.unsqueeze(-1) * modal_seq, dim=1)  # [B, dim]

        return user_embedding


class NAML(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.modal_fusion = modal_fusion.get_fusion_layer('maf', self.projection_dim, self.mm_features)

        self.user_linear = torch.nn.Linear(self.projection_dim*self.user_features_num, self.projection_dim)

        self.user_encoder = UserEncoder(self.projection_dim)

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
        user_interest = self.user_encoder(feats_seq_fusion)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)
        user_vec = self.user_linear(user_vec)

        user_vec = torch.sum(torch.stack([user_vec, user_interest], dim=1), dim=1) # [B, dim]

        logit = torch.einsum("bd,bd->b", feats_fusion, user_vec).unsqueeze(-1)
        out = {'pred': logit}
        return out

