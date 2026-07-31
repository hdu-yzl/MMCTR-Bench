from ..base_model import BaseModel
from ..layers.common import MultiLayerPerceptron
from ..layers import seq_pooling

import math
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PAMDDisentangleBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, use_layer_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.common_a = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.common_b = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.a2b = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.b2a = self._mlp(dim, hidden_dim, dim, dropout, use_layer_norm)
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _mlp(in_dim, hidden_dim, out_dim, dropout, use_layer_norm):
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*layers)

    def decompose(self, a, b):
        common_a = self.common_a(a)
        common_b = self.common_b(b)
        return common_a, common_b, a - common_a, b - common_b

    def aux_loss(self, a, b, common_a, common_b, spec_a, spec_b):
        align_loss = F.mse_loss(common_a, common_b)
        orth_loss = torch.mean(torch.sum(F.normalize(spec_a, dim=-1) * F.normalize(spec_b, dim=-1), dim=-1).pow(2))

        loss_c = F.mse_loss(b, self.a2b(common_a)) + F.mse_loss(a, self.b2a(common_b))
        loss_p = F.mse_loss(b, self.a2b(a)) + F.mse_loss(a, self.b2a(b))
        loss_s = F.mse_loss(b, self.a2b(spec_a)) + F.mse_loss(a, self.b2a(spec_b))
        rank_loss = -F.logsigmoid(loss_p - loss_c) - F.logsigmoid(loss_s - loss_p)
        return align_loss + orth_loss + rank_loss

    def fuse(self, q, common_a, common_b, spec_a, spec_b):
        reps = torch.stack([common_a, common_b, spec_a, spec_b], dim=-2)
        score = torch.sum(self.attn_q(q).unsqueeze(-2) * self.attn_k(reps), dim=-1)
        weight = torch.softmax(score / math.sqrt(q.size(-1)), dim=-1).unsqueeze(-1)
        return q + torch.sum(weight * reps, dim=-2)

    def forward(self, a, b, q):
        common_a, common_b, spec_a, spec_b = self.decompose(a, b)
        fused = self.fuse(q, common_a, common_b, spec_a, spec_b)
        loss = self.aux_loss(a, b, common_a, common_b, spec_a, spec_b)
        return fused, loss


class PAMD(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.pamd_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_features]
        self.seq_pamd_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_seq_features]
        if len(self.pamd_features) < 2 or len(self.seq_pamd_features) < 2:
            raise ValueError('PAMD requires at least two modalities among image/text/audio.')

        hidden_dim = model_config.get('pamd_hidden_dim', self.projection_dim)
        self.pamd_aux_weight = model_config.get('pamd_aux_weight', 0.1)
        self.pooling = seq_pooling.get_pooling(self.seq_pooling_method, dim=1)

        self.target_pamd = self._build_pamd_blocks(self.pamd_features, hidden_dim, model_config)
        self.seq_pamd = self._build_pamd_blocks(self.seq_pamd_features, hidden_dim, model_config)
        self.dnn = MultiLayerPerceptron(
            self.projection_dim * 2,
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

    def _build_pamd_blocks(self, features, hidden_dim, model_config):
        return nn.ModuleDict({
            self._pair_key(a, b): _PAMDDisentangleBlock(
                self.projection_dim,
                hidden_dim=hidden_dim,
                dropout=self.dropout,
                use_layer_norm=model_config.get('pamd_layer_norm', True),
            )
            for a, b in combinations(features, 2)
        })

    @staticmethod
    def _pair_key(a, b):
        return f'{a}__{b}'

    def _run_pamd(self, blocks, features, feats_p, q):
        fused_list, aux_loss = [], []
        for a, b in combinations(features, 2):
            fused, loss = blocks[self._pair_key(a, b)](feats_p[a], feats_p[b], q)
            fused_list.append(fused)
            aux_loss.append(loss)
        return torch.stack(fused_list, dim=1).mean(dim=1), torch.stack(aux_loss).mean()

    def forward(self, feats, feats_seq):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        feats['id'] = self.embedding(feats['id']).view(-1, self.id_dim)
        feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_seq_projector[k](feats_seq[k]) for k in self.mm_seq_features}
        feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}

        feats_fusion, target_aux = self._run_pamd(self.target_pamd, self.pamd_features, feats_p, feats_p['id'])
        feats_seq_pool_fusion, seq_aux = self._run_pamd(
            self.seq_pamd,
            self.seq_pamd_features,
            feats_seq_pool,
            feats_seq_pool['id'],
        )

        x_dnn = torch.cat([feats_fusion, feats_seq_pool_fusion], dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit, 'au_loss': self.pamd_aux_weight * (target_aux + seq_aux)}
        return out
