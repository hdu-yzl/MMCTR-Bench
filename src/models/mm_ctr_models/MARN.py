from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import modal_fusion, seq_pooling
import torch.nn as nn
import torch

torch.autograd.set_detect_anomaly(True)
class GradientReversalLayer(torch.autograd.Function):
    """
    前向传播：x -> x
    反向传播：dy/dx -> -lambda * dy/dx
    """

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalLayer.apply(x, lambd)


class ModalitySplit(nn.Module):
    """
    把每个模态 embedding 拆成 specific & invariant 两部分
    """

    def __init__(self, dim, mm_features=['id', 'text', 'image', 'audio']):
        super().__init__()
        self.mm_features = mm_features
        self.specifics = nn.ModuleDict({i: nn.Linear(dim, dim) for i in mm_features})  # Sm
        self.invariant = nn.Linear(dim, dim)  # I (共享)

    def forward(self, feats):
        feats_s = {k: self.specifics[k](feats[k]) for k in self.mm_features}
        feats_c = {k: self.invariant(feats[k]) for k in self.mm_features}

        return feats_s, feats_c


class DDMA(nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image', 'audio']):
        super().__init__()
        self.mm_nums = len(mm_features)
        self.mm_features = mm_features
        self.D0 = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                nn.Linear(64, self.mm_nums))
        self.D1 = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                nn.Linear(64, self.mm_nums))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feats, lambd=1.0):
        feats_list = [feats[k] for k in self.mm_features]

        cim = torch.stack(feats_list, dim=1)  # (B, n_mod, proj_dim)
        B, n_mod, D = cim.shape

        # ---------- 1. D0 ----------
        # 把每个模态看作一个独立样本，D0 输出 logits (B*n_mod, n_mod)
        logits_d0 = self.D0(cim.detach().view(B * n_mod, D))
        label = torch.arange(n_mod, device=cim.device).repeat(B)  # (B*n_mod,)
        loss_d0 = self.ce(logits_d0, label)

        # 权重：1 - softmax 概率，取对应模态的列
        w = 1. - torch.softmax(logits_d0, dim=-1)  # (B*n_mod, n_mod)
        w = w[torch.arange(B * n_mod), label].unsqueeze(-1)  # (B*n_mod, 1)

        # reshape 回 (B, n_mod, 1)
        w = w.view(B, n_mod, 1)

        # ---------- 2. 加权特征 ----------
        inv_w = w * cim  # (B, n_mod, D)

        # ---------- 3. D1 对抗 ----------
        inv_grl = grad_reverse(inv_w.view(B * n_mod, D), lambd)

        logits_d1 = self.D1(inv_grl)  # (B*n_mod, n_mod)
        loss_d1 = self.ce(logits_d1, label)

        # ---------- 4. 融合 ----------
        inv_out = inv_w.max(dim=1)[0]  # (B, D)
        return loss_d0, loss_d1, inv_out


# Ds损失计算
class ModalitySpecificLoss(nn.Module):
    def __init__(self, proj_dim,  mm_features):
        super().__init__()
        self.mm_features = mm_features
        self.mm_nums = len(mm_features)
        self.Ds = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.ReLU(),
            nn.Linear(64, self.mm_nums)
        )
        self.ce = nn.CrossEntropyLoss()


    def forward(self, feats):
        feats = [feats[k] for k in self.mm_features]
        # 原文切断传播
        s = torch.stack([f.detach() for f in feats], dim=1)  # (B, n_mod, proj_dim)
        logits = self.Ds(s)  # (B, n_mod, n_mod)
        label = torch.arange(self.mm_nums, device=s.device).repeat(s.size(0))
        return self.ce(logits.view(-1, s.size(1)), label.reshape(-1))


class MARN(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.lambda0 = model_config.get('lambda0', 0.05)

        self.modal_poolings = seq_pooling.get_pooling('din', dim=self.projection_dim, mlp_dims=self.mlp_dims)

        self.split_module = ModalitySplit(self.projection_dim, self.mm_features)

        # MAF
        self.maf = modal_fusion.get_fusion_layer('maf', self.projection_dim, self.mm_features)

        # DDMA
        self.ddma = DDMA(self.projection_dim, self.mm_features)

        # GRU
        # self.gru = nn.GRU(self.projection_dim, self.projection_dim, batch_first=True)

        self.Ds_loss = ModalitySpecificLoss(self.projection_dim, self.mm_features)

        final_input_dim = self.projection_dim * (2 + self.user_features_num)  # user_rep + item_rep + attention_rep
        self.dnn = MultiLayerPerceptron(final_input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)

        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        au_loss = 0.0
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        feats_s, feats_c = self.split_module(feats_p)
        feats_seq_s, feats_seq_c = self.split_module(feats_seq_p)

        target_maf = self.maf(feats_s)
        seq_maf = self.maf(feats_seq_s)  # (B,seq_num,projection_dim)

        loss_d0, loss_d1, inv_w = self.ddma(feats_c)

        loss_ds = self.Ds_loss(feats_s)
        au_loss += loss_d0 + loss_d1 * self.lambda0 + loss_ds * self.lambda0

        w_seq_list = []
        for i in range(self.seq_len):
            feats_seq_input = {k: feats_seq_c[k][:, i, :] for k in self.mm_features}
            loss_d0, loss_d1, inv_w = self.ddma(feats_seq_input)

            w_seq_list.append(inv_w)

        inv_w_seq = torch.stack(w_seq_list, dim=1)  # (B,seq_num,projection_dim)

        rep_x = inv_w + target_maf
        rep_seq = inv_w_seq + seq_maf

        # rep_seq_h, hidden = self.gru(rep_seq)  # (B,seq_num,projection_dim)

        attention_pool = self.modal_poolings(rep_x, rep_seq)  # (B,projection_dim)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        final_vec = torch.cat([user_vec, attention_pool, rep_x], dim=1)

        dnn_out = self.dnn(final_vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit, 'au_loss': au_loss}
        return out

