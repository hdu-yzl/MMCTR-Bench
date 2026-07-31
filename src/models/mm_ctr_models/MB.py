from ..base_model import BaseModel
from ..layers.common import MultiLayerPerceptron
from ..layers import seq_pooling

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MBModalityFusion(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, modal_embs):
        x = torch.stack(modal_embs, dim=1)
        weight = torch.softmax(self.scorer(self.proj(x)).squeeze(-1), dim=1)
        return torch.sum(x * weight.unsqueeze(-1), dim=1), weight


class MB(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.modal_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_features]
        self.seq_modal_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_seq_features]
        self.sensitive_modal = model_config.get('sensitive_modal', 'image')
        self.insensitive_modal = model_config.get('insensitive_modal', 'text')

        self.balance_weight = model_config.get('mb_balance_weight', model_config.get('balance_weight', 0.1))
        self.balance_sample_num = model_config.get('mb_sample_num', model_config.get('balance_sample_num', 30))
        self.adv_eps = model_config.get('mb_adv_eps', model_config.get('adv_eps', 1.0))
        self.pgd_steps = model_config.get('mb_pgd_steps', model_config.get('pgd_steps', 3))
        self.pgd_step_size = model_config.get(
            'mb_pgd_step_size',
            model_config.get('pgd_step_size', self.adv_eps / max(self.pgd_steps, 1)),
        )

        self.id_pair_encoder = nn.Linear(self.id_dim, self.projection_dim)
        self.user_id_encoder = nn.Linear(self.latent_dim, self.projection_dim)
        self.item_id_encoder = nn.Linear(self.latent_dim, self.projection_dim)

        self.user_modal_encoder = nn.ModuleDict({
            k: nn.Linear(self.latent_dim, self.projection_dim)
            for k in self.modal_features
        })
        self.item_modal_encoder = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(self.projection_dim, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.projection_dim, self.projection_dim),
            )
            for k in dict.fromkeys(self.modal_features + self.seq_modal_features)
        })
        self.fusion = _MBModalityFusion(
            self.projection_dim,
            hidden_dim=model_config.get('mb_fusion_hidden_dim', self.projection_dim),
            dropout=self.dropout,
        )
        self.pooling = seq_pooling.get_pooling(self.seq_pooling_method, dim=1)

        self.dnn = MultiLayerPerceptron(
            self.projection_dim * 6,
            self.mlp_dims,
            self.dropout,
            use_bn=self.bn,
            activation='relu',
        )
        self.out_put = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            use_bn=self.bn,
            activation=None,
        )

        self.compile()
        self.model_to_device()
        self.log_model_params()

    def _encode_id(self, feats):
        bsz = feats['id'].size(0)
        id_emb = feats['id'].view(bsz, self.id_fields_num, self.latent_dim)
        user_raw, item_raw = id_emb[:, 0], id_emb[:, -1]
        return (
            user_raw,
            self.user_id_encoder(user_raw),
            self.item_id_encoder(item_raw),
            self.id_pair_encoder(id_emb.reshape(bsz, -1)),
        )

    def _encode_item_modalities(self, feats_p):
        return {
            k: self.item_modal_encoder[k](feats_p[k])
            for k in self.modal_features if k in feats_p
        }

    def _encode_seq_modalities(self, feats_seq_pool):
        seq_modal_embs = [
            self.item_modal_encoder[k](feats_seq_pool[k])
            for k in self.seq_modal_features if k in feats_seq_pool
        ]
        return self.fusion(seq_modal_embs)[0] if seq_modal_embs else feats_seq_pool['id']

    def _get_modal_score(self, user_raw, modal_embs, modal_name):
        if modal_name not in modal_embs:
            return user_raw.new_zeros(user_raw.size(0), 1)
        user = F.normalize(self.user_modal_encoder[modal_name](user_raw), dim=-1)
        item = F.normalize(modal_embs[modal_name], dim=-1)
        return torch.sum(user * item, dim=-1, keepdim=True)

    def forward(self, feats, feats_seq=None, label=None):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_seq_projector[k](feats_seq[k]) for k in self.mm_seq_features}
        feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}

        user_raw, user_id, item_id, pair_id = self._encode_id(feats)
        modal_embs = self._encode_item_modalities(feats_p)
        fused_modal, modal_weight = self.fusion(list(modal_embs.values()) or [item_id])
        feats_seq_pool_fusion = self._encode_seq_modalities(feats_seq_pool)

        modal_score = {
            k: self._get_modal_score(user_raw, modal_embs, k)
            for k in self.modal_features
        }
        id_score = torch.sum(
            F.normalize(user_id, dim=-1) * F.normalize(item_id, dim=-1),
            dim=-1,
            keepdim=True,
        )

        x_dnn = torch.cat([
            pair_id,
            user_id,
            item_id,
            fused_modal,
            feats_seq_pool_fusion,
            user_id * fused_modal,
        ], dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put) + id_score + sum(modal_score.values(), id_score.new_tensor(0.0))
        au_loss = logit.new_tensor(0.0)
        if self.training and label is not None and self._can_balance(modal_embs):
            au_loss = self._modality_balance_loss(feats, label, user_raw, modal_embs)

        out = {
            'pred': logit,
            'au_loss': au_loss,
            'modal_weight': modal_weight,
            'id_score': id_score,
        }
        out.update({f'{k}_score': v for k, v in modal_score.items()})
        return out

    def _can_balance(self, modal_embs):
        return (
            self.balance_weight > 0
            and self.sensitive_modal != self.insensitive_modal
            and self.sensitive_modal in modal_embs
            and self.insensitive_modal in modal_embs
        )

    def _modality_balance_loss(self, feats, label, user_raw, modal_embs):
        y = label.view(-1)
        pos_idx = torch.nonzero(y > 0.5, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(y <= 0.5, as_tuple=False).view(-1)
        sample_num = min(pos_idx.numel(), neg_idx.numel(), int(self.balance_sample_num))
        if sample_num <= 0:
            return user_raw.new_tensor(0.0)

        pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:sample_num]]
        neg_idx = neg_idx[torch.randint(0, neg_idx.numel(), (sample_num,), device=neg_idx.device)]
        sen, ins = self.sensitive_modal, self.insensitive_modal

        user_sen = F.normalize(self.user_modal_encoder[sen](user_raw[pos_idx]), dim=-1)
        user_ins = F.normalize(self.user_modal_encoder[ins](user_raw[pos_idx]), dim=-1)
        sen_pos = F.normalize(modal_embs[sen][pos_idx], dim=-1)
        sen_adv = F.normalize(self._distill_sensitive_adv_embedding(feats, pos_idx, neg_idx), dim=-1)
        ins_pos = F.normalize(modal_embs[ins][pos_idx], dim=-1)
        ins_neg = F.normalize(modal_embs[ins][neg_idx], dim=-1)

        sen_margin = torch.sum(user_sen * sen_pos, dim=-1) - torch.sum(user_sen * sen_adv, dim=-1)
        ins_margin = torch.sum(user_ins * ins_pos, dim=-1) - torch.sum(user_ins * ins_neg, dim=-1)
        return self.balance_weight * F.relu(sen_margin - ins_margin).mean()

    def _distill_sensitive_adv_embedding(self, feats, pos_idx, neg_idx):
        sen = self.sensitive_modal
        x0 = feats[sen][pos_idx].detach()
        target = self.item_modal_encoder[sen](self.mm_projector[sen](feats[sen][neg_idx])).detach()
        z = x0.clone().requires_grad_(True)

        for _ in range(max(int(self.pgd_steps), 1)):
            z_emb = self.item_modal_encoder[sen](self.mm_projector[sen](z))
            grad = torch.autograd.grad(F.mse_loss(z_emb, target), z)[0]
            z = torch.clamp(z - float(self.pgd_step_size) * grad.sign(), x0 - self.adv_eps, x0 + self.adv_eps)
            z = z.detach().requires_grad_(True)

        return self.item_modal_encoder[sen](self.mm_projector[sen](z.detach()))

    def _predict_batch(self, batch):
        feats, feats_seq, label = batch
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}
        label = label.to(self.device)
        return self(feats, feats_seq, label=label), label
