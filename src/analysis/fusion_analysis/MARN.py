"""
MARN 模型 —— 融合分析版本
原始模型硬编码使用 MAF 融合，此版本支持所有融合方法。

模型核心操作（保持不变）:
  - ModalitySplit（模态拆分为 specific + invariant）
  - DDMA（双判别器模态对齐）+ GRL（梯度反转层）
  - ModalitySpecificLoss (Ds)
  - DIN 序列池化

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    逐时间步融合 specific 特征。
  序列感知融合（dta, gmmf, dmf）:
    替代 specific 特征融合 + DIN 池化步骤。
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron
from models.layers import modal_fusion, seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch.nn as nn
import torch

torch.autograd.set_detect_anomaly(True)


class GradientReversalLayer(torch.autograd.Function):
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
    def __init__(self, dim, mm_features=['id', 'text', 'image', 'audio']):
        super().__init__()
        self.mm_features = mm_features
        self.specifics = nn.ModuleDict({i: nn.Linear(dim, dim) for i in mm_features})
        self.invariant = nn.Linear(dim, dim)

    def forward(self, feats):
        feats_s = {k: self.specifics[k](feats[k]) for k in self.mm_features}
        feats_c = {k: self.invariant(feats[k]) for k in self.mm_features}
        return feats_s, feats_c


class DDMA(nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image', 'audio']):
        super().__init__()
        self.mm_nums = len(mm_features)
        self.mm_features = mm_features
        self.D0 = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, self.mm_nums))
        self.D1 = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, self.mm_nums))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feats, lambd=1.0):
        feats_list = [feats[k] for k in self.mm_features]
        cim = torch.stack(feats_list, dim=1)
        B, n_mod, D = cim.shape

        logits_d0 = self.D0(cim.detach().view(B * n_mod, D))
        label = torch.arange(n_mod, device=cim.device).repeat(B)
        loss_d0 = self.ce(logits_d0, label)

        w = 1. - torch.softmax(logits_d0, dim=-1)
        w = w[torch.arange(B * n_mod), label].unsqueeze(-1)
        w = w.view(B, n_mod, 1)

        inv_w = w * cim

        inv_grl = grad_reverse(inv_w.view(B * n_mod, D), lambd)
        logits_d1 = self.D1(inv_grl)
        loss_d1 = self.ce(logits_d1, label)

        inv_out = inv_w.max(dim=1)[0]
        return loss_d0, loss_d1, inv_out


class ModalitySpecificLoss(nn.Module):
    def __init__(self, proj_dim, mm_features):
        super().__init__()
        self.mm_features = mm_features
        self.mm_nums = len(mm_features)
        self.Ds = nn.Sequential(nn.Linear(proj_dim, 64), nn.ReLU(), nn.Linear(64, self.mm_nums))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feats):
        feats_list = [feats[k] for k in self.mm_features]
        s = torch.stack([f.detach() for f in feats_list], dim=1)
        logits = self.Ds(s)
        label = torch.arange(self.mm_nums, device=s.device).repeat(s.size(0))
        return self.ce(logits.view(-1, s.size(1)), label.reshape(-1))


def _build_fusion_kwargs(method, model_config, projection_dim):
    """根据融合方法构建额外参数（保留兼容）"""
    kwargs = {}
    if method == 'lmf':
        kwargs['rank'] = model_config.get('rank', 5)
        kwargs['output_dim'] = model_config.get('fusion_dim', projection_dim)
    elif method == 'mtfn':
        kwargs['rank'] = model_config.get('rank', 20)
    elif method == 'src':
        kwargs['T'] = model_config.get('T', 10)
    return kwargs


class MARN(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.lambda0 = model_config.get('lambda0', 0.05)

        self.split_module = ModalitySplit(self.projection_dim, self.mm_features)

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'maf'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        f_info = build_fusion(self._fusion_method, self.projection_dim,
                              self.mm_features, model_config)
        self.modal_fusion_layer = f_info['layer']
        self._query_num = f_info['query_num']
        self._sim_fusion = f_info.get('sim_fusion')
        self._gmmf_pooling = f_info.get('seq_pooling')
        fusion_out_dim = f_info['out_dim']

        # 当融合输出维度 ≠ projection_dim 时，添加投影层对齐
        self.fusion_projector = (nn.Linear(fusion_out_dim, self.projection_dim)
                                 if fusion_out_dim != self.projection_dim
                                 else nn.Identity())

        # DDMA
        self.ddma = DDMA(self.projection_dim, self.mm_features)
        self.Ds_loss = ModalitySpecificLoss(self.projection_dim, self.mm_features)

        if not self._is_seq_fusion:
            self.modal_poolings = seq_pooling.get_pooling('din', dim=self.projection_dim, mlp_dims=self.mlp_dims)

        final_input_dim = self.projection_dim * (2 + self.user_features_num)
        self.dnn = MultiLayerPerceptron(final_input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        au_loss = 0.0
        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        feats_s, feats_c = self.split_module(feats_p)
        feats_seq_s, feats_seq_c = self.split_module(feats_seq_p)

        # DDMA + Ds loss（所有融合方法共用）
        loss_d0, loss_d1, inv_w = self.ddma(feats_c)
        loss_ds = self.Ds_loss(feats_s)
        au_loss += loss_d0 + loss_d1 * self.lambda0 + loss_ds * self.lambda0

        w_seq_list = []
        for i in range(self.seq_len):
            feats_seq_input = {k: feats_seq_c[k][:, i, :] for k in self.mm_features}
            loss_d0_i, loss_d1_i, inv_w_i = self.ddma(feats_seq_input)
            w_seq_list.append(inv_w_i)
        inv_w_seq = torch.stack(w_seq_list, dim=1)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        if self._fusion_method == 'dta':
            target_fused, _ = fuse_seq_dta(
                self.modal_fusion_layer, self._sim_fusion,
                feats_s, feats_seq_s, self.mm_features, self.seq_len)
            target_fused = self.fusion_projector(target_fused)
            rep_x = inv_w + target_fused
            # DTA 已包含序列池化
            attention_pool = self.fusion_projector(fuse_seq_dta(
                self.modal_fusion_layer, self._sim_fusion,
                feats_s, feats_seq_s, self.mm_features, self.seq_len)[0])

        elif self._fusion_method == 'gmmf':
            fused, gmmf_loss = fuse_seq_gmmf(
                self.modal_fusion_layer, self._gmmf_pooling,
                feats_s, feats_seq_s, user_vec)
            fused = self.fusion_projector(fused)
            au_loss += gmmf_loss
            rep_x = inv_w + feats_s['id']
            attention_pool = fused

        elif self._fusion_method == 'dmf':
            fused, _ = fuse_seq_dmf(
                self.modal_fusion_layer, feats_s, feats_seq_s, self.seq_len)
            fused = self.fusion_projector(fused)
            rep_x = inv_w + feats_s['id']
            attention_pool = fused

        else:
            # 本地融合
            target_fused, cl1 = fuse_local(
                self.modal_fusion_layer, feats_s, self.mm_features,
                self._fusion_method, self._query_num)
            target_fused = self.fusion_projector(target_fused)
            seq_fused = torch.stack(
                [self.fusion_projector(fuse_local(
                    self.modal_fusion_layer,
                    {k: feats_seq_s[k][:, i, :] for k in self.mm_features},
                    self.mm_features, self._fusion_method, self._query_num)[0])
                 for i in range(self.seq_len)],
                dim=1,
            )
            if isinstance(cl1, torch.Tensor):
                au_loss = au_loss + cl1

            rep_x = inv_w + target_fused
            rep_seq = inv_w_seq + seq_fused
            attention_pool = self.modal_poolings(rep_x, rep_seq)

        final_vec = torch.cat([user_vec, attention_pool, rep_x], dim=1)
        dnn_out = self.dnn(final_vec)
        logit = self.out_put(dnn_out)
        out = {'pred': logit, 'au_loss': au_loss}
        return out
