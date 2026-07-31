from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron

import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(self, seq_len, dim, token_hidden_dim=None, channel_hidden_dim=None, dropout=0.0):
        super().__init__()
        token_hidden_dim = token_hidden_dim or max(seq_len * 2, 1)
        channel_hidden_dim = channel_hidden_dim or dim * 2

        self.norm_token = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.norm_channel = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.token_mlp(self.norm_token(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.norm_channel(x))


class _MixerModule(nn.Module):
    def __init__(self, seq_len, dim, num_layers=2, token_hidden_dim=None, channel_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _MixerBlock(seq_len, dim, token_hidden_dim, channel_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MMMLP(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        if not self.mm_features:
            raise ValueError('MMMLP requires at least one modality.')

        self.feature_mixer_layers = model_config.get('feature_mixer_layers', model_config.get('mixer_layers', 2))
        self.fusion_mixer_layers = model_config.get('fusion_mixer_layers', model_config.get('mixer_layers', 2))
        self.token_hidden_dim = model_config.get('token_hidden_dim', max(self.seq_len * 2, 1))
        self.channel_hidden_dim = model_config.get('channel_hidden_dim', self.projection_dim * 2)
        self.fusion_dim = self.projection_dim * len(self.mm_features)
        self.user_dim = self.projection_dim * len(self.user_features)

        self.feature_mixers = nn.ModuleDict({
            k: _MixerModule(
                self.seq_len,
                self.projection_dim,
                num_layers=self.feature_mixer_layers,
                token_hidden_dim=self.token_hidden_dim,
                channel_hidden_dim=self.channel_hidden_dim,
                dropout=self.dropout,
            )
            for k in self.mm_features
        })
        self.fusion_mixer = _MixerModule(
            self.seq_len,
            self.fusion_dim,
            num_layers=self.fusion_mixer_layers,
            token_hidden_dim=self.token_hidden_dim,
            channel_hidden_dim=model_config.get('fusion_channel_hidden_dim', self.fusion_dim * 2),
            dropout=self.dropout,
        )
        self.target_projector = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.dnn = MultiLayerPerceptron(
            self.fusion_dim * 3 + self.user_dim,
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

    @staticmethod
    def _squeeze_item(x):
        return x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x

    def _last_hidden(self, seq_repr, seq_id_raw=None):
        if seq_id_raw is None:
            return seq_repr[:, -1]
        mask = seq_id_raw.ne(0).squeeze(-1) if seq_id_raw.dim() > 2 else seq_id_raw.ne(0)
        idx = mask.long().sum(dim=1).clamp(min=1) - 1
        return seq_repr[torch.arange(seq_repr.size(0), device=seq_repr.device), idx]

    def _encode_sequence_modalities(self, feats_seq):
        seq_outputs = []
        for k in self.mm_features:
            x = feats_seq[k] if k == 'id' else feats_seq[k].float()
            seq_outputs.append(self.feature_mixers[k](self.mm_projector[k](x)))
        return seq_outputs

    def _encode_target_modalities(self, feats):
        target_outputs = []
        for k in self.mm_features:
            x = feats[k] if k == 'id' else feats[k].float()
            target_outputs.append(self.mm_projector[k](self._squeeze_item(x)))
        return self.target_projector(torch.cat(target_outputs, dim=-1))

    def _encode_user_features(self, user_feats):
        if not self.user_features:
            return None

        user_outputs = []
        for k in self.user_features:
            x = self.embedding(user_feats[k]) if k == 'id' else user_feats[k].float()
            user_outputs.append(self.user_projector[k](self._squeeze_item(x)))
        return torch.cat(user_outputs, dim=-1)

    def forward(self, user_feats, feats, feats_seq):
        feats = dict(feats)
        feats_seq = dict(feats_seq)
        user_feats = dict(user_feats)
        seq_id_raw = feats_seq.get('id')

        if 'id' in feats:
            feats['id'] = self._squeeze_item(self.embedding(feats['id']))
        if 'id' in feats_seq:
            feats_seq['id'] = self.embedding(feats_seq['id'])

        feats_seq_fusion = self.fusion_mixer(torch.cat(self._encode_sequence_modalities(feats_seq), dim=-1))
        feats_seq_pool_fusion = self._last_hidden(feats_seq_fusion, seq_id_raw)
        feats_fusion = self._encode_target_modalities(feats)
        pred_inputs = [feats_seq_pool_fusion, feats_fusion, feats_seq_pool_fusion * feats_fusion]

        user_vec = self._encode_user_features(user_feats)
        if user_vec is not None:
            pred_inputs.append(user_vec)

        x_dnn = torch.cat(pred_inputs, dim=-1)
        dnn_out_put = self.dnn(x_dnn)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit}
        return out
