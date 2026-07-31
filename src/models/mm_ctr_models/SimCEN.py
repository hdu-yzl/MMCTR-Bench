from ..base_model import BaseModel
from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import modal_fusion, seq_pooling
import torch
import torch.nn as nn


class SimCEN(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        # 参数
        self.hidden_unit = model_config.get('hidden_unit', [480, 480, 480])
        self.cl_temperature = model_config.get('cl_temperature', 0.1)
        self.alpha = model_config.get('alpha', 0.5)
        self.ego_dropout = model_config.get('ego_dropout', 0.0)
        self.v1_dropout = model_config.get('v1_dropout', 0.0)
        self.v2_dropout = model_config.get('v2_dropout', 0.0)
        self.ego_batch_norm = model_config.get('ego_batch_norm', True)
        self.v1_batch_norm = model_config.get('v1_batch_norm', True)
        self.v2_batch_norm = model_config.get('v2_batch_norm', True)


        #self.modal_fusion_method = model_config.get('modal_fusion_method', 'lmf')
        self.seq_pooling_method = model_config.get('seq_pooling_method', 'mean')

        self.modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
                                                          self.mm_features)
        #self.seq_modal_fusion = modal_fusion.get_fusion_layer(self.modal_fusion_method, self.projection_dim,
        #                                                      self.mm_seq_features)
        self.pooling = seq_pooling.get_pooling(self.seq_pooling_method, dim=1)

        flatten_dim = self.projection_dim * (
                len(self.mm_features) + len(self.mm_seq_features))  # 所有字段 embedding 拼接后的总维度
        num_fields = len(self.mm_features) + len(self.mm_seq_features)

        self.mlep = MLEP(input_dim=flatten_dim * 3,
                         hidden_units=self.hidden_unit,
                         ego_dropout=self.ego_dropout,
                         v1_dropout=self.v1_dropout,
                         v2_dropout=self.v2_dropout,
                         ego_batch_norm=self.ego_batch_norm,
                         v1_batch_norm=self.v1_batch_norm,
                         v2_batch_norm=self.v2_batch_norm)

        self.segmentation = Segmentation(num_fields=num_fields,
                                         embedding_dim=self.projection_dim,
                                         flatten_dim=flatten_dim)

        self.out_put = MultiLayerPerceptron(self.hidden_unit[-1],
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

        feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}  # (B,fusion_dim)

        feats_p_flatten = torch.stack([feats_p[k] for k in self.mm_features], dim=1) # (B,num_fields,latent_dim)
        feats_seq_pool_flatten = torch.stack([feats_seq_pool[k] for k in self.mm_seq_features], dim=1)
        x_dnn = torch.cat([feats_p_flatten, feats_seq_pool_flatten], dim=1)  # (B,num_fields,latent_dim)

        ego, view1, view2 = self.segmentation(x_dnn)
        V = self.mlep(torch.cat([ego, view1, view2], dim=-1))
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        v1 = ego + v1
        v2 = ego + v2
        V = torch.cat([ego, v1, v2], dim=-1)
        logit = self.out_put(V)
        return_dict = {"y_pred": logit,
                       "ego": ego,
                       "v1": v1,
                       "v2": v2}

        ego = return_dict["ego"]
        v1 = return_dict["v1"]
        v2 = return_dict["v2"]
        cl_loss = self.InfoNCE(ego, v1, v2, cl_temperature=self.cl_temperature)

        out = {'pred': logit, 'au_loss': self.alpha * cl_loss}
        return out

    def InfoNCE(self, ego, embedding_1, embedding_2, cl_temperature):
        ego = torch.nn.functional.normalize(ego)
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        pos_score_e_1 = (ego * embedding_1).sum(dim=-1)
        pos_score_e_2 = (ego * embedding_2).sum(dim=-1)
        pos_score = (pos_score_e_1 + pos_score_e_2) * 0.5

        pos_score = torch.exp(pos_score / cl_temperature)

        ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / cl_temperature).sum(dim=-1)
        loss = - torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(loss)


class Segmentation(nn.Module):
    def __init__(self, num_fields, embedding_dim, flatten_dim):
        super(Segmentation, self).__init__()
        self.triu_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 0).bool(),
                                      requires_grad=False)
        self.tril_mask = nn.Parameter(torch.tril(torch.ones(num_fields, num_fields), 0).bool(),
                                      requires_grad=False)
        self.kp_dim = int(num_fields * (num_fields + 1) / 2)
        self.kp_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim), requires_grad=True)
        nn.init.xavier_normal_(self.kp_W)
        self.project_triu = nn.Linear(self.kp_dim, flatten_dim, bias=False)
        self.project_tril = nn.Linear(self.kp_dim, flatten_dim, bias=False)

    def forward(self, feature_emb):
        embs_kp = torch.matmul(torch.matmul(feature_emb, self.kp_W), feature_emb.transpose(1, 2))
        triu = torch.masked_select(embs_kp, self.triu_mask).view(-1, self.kp_dim)
        tril = torch.masked_select(embs_kp, self.tril_mask).view(-1, self.kp_dim)
        embs_flatten = feature_emb.flatten(start_dim=1)  # 原始特征 flatten 后的表示
        view_1 = self.project_triu(triu)
        view_2 = self.project_tril(tril)
        return embs_flatten, view_1, view_2


class MLEP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=None,
                 ego_dropout=0.0,
                 v1_dropout=0.0,
                 v2_dropout=0.0,
                 ego_batch_norm=True,
                 v1_batch_norm=True,
                 v2_batch_norm=True):
        super(MLEP, self).__init__()
        if hidden_units is None:
            hidden_units = [480, 480, 480]
        if type(ego_dropout) != list:
            ego_dropout = [ego_dropout] * len(hidden_units)
        if type(v1_dropout) != list:
            v1_dropout = [v1_dropout] * len(hidden_units)
        if type(v2_dropout) != list:
            v2_dropout = [v2_dropout] * len(hidden_units)

        self.layer = nn.ModuleList()
        self.norm_ego = nn.ModuleList()
        self.norm_view_1 = nn.ModuleList()
        self.norm_view_2 = nn.ModuleList()
        self.dropout_ego = nn.ModuleList()
        self.dropout_view_1 = nn.ModuleList()
        self.dropout_view_2 = nn.ModuleList()
        self.activation_ego = nn.ModuleList()
        self.activation_view_1 = nn.ModuleList()
        self.activation_view_2 = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            one_third = int(hidden_units[idx + 1] / 3)
            self.layer.append(Linear_unit(hidden_units[idx], hidden_units[idx + 1]))
            if ego_batch_norm:
                self.norm_ego.append(nn.BatchNorm1d(one_third))
            if v1_batch_norm:
                self.norm_view_1.append(nn.BatchNorm1d(one_third))
            if v2_batch_norm:
                self.norm_view_2.append(nn.BatchNorm1d(one_third))
            if ego_dropout[idx] > 0:
                self.dropout_ego.append(nn.Dropout(ego_dropout[idx]))
            if v1_dropout[idx] > 0:
                self.dropout_view_1.append(nn.Dropout(v1_dropout[idx]))
            if v2_dropout[idx] > 0:
                self.dropout_view_2.append(nn.Dropout(v2_dropout[idx]))
            self.activation_ego.append(nn.ReLU())
            self.activation_view_1.append(nn.ReLU())
            self.activation_view_2.append(nn.ReLU())

    def forward(self, X):
        V_i = X
        for i in range(len(self.layer)):
            if i == 0:
                ego, v1, v2 = self.layer[i].forward1(V_i)
            else:
                ego, v1, v2 = self.layer[i].forward2(V_i)
            if len(self.norm_ego) > i:
                ego = self.norm_ego[i](ego)
            if len(self.norm_view_1) > i:
                v1 = self.norm_view_1[i](v1)
            if len(self.norm_view_2) > i:
                v2 = self.norm_view_2[i](v2)
            if self.activation_ego[i] is not None:
                ego = self.activation_ego[i](ego)
            if self.activation_view_1[i] is not None:
                v1 = self.activation_view_1[i](v1)
            if self.activation_view_2[i] is not None:
                v2 = self.activation_view_2[i](v2)
            if len(self.dropout_ego) > i:
                ego = self.dropout_ego[i](ego)
            if len(self.dropout_view_1) > i:
                v1 = self.dropout_view_1[i](v1)
            if len(self.dropout_view_2) > i:
                v2 = self.dropout_view_2[i](v2)
            V_i = torch.cat([ego, v1, v2], dim=-1)
        return V_i


class Linear_unit(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(Linear_unit, self).__init__()
        assert output_dim % 3 == 0, "output_dim should be divisible by 3."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        one_third = int(output_dim / 3)
        self.gate = nn.Sequential(nn.Linear(one_third, one_third, bias=True),
                                  nn.Sigmoid())
        self.gate_tau = nn.Parameter(torch.ones(one_third), requires_grad=True)
        self.noise = nn.Parameter(torch.empty(2 * one_third), requires_grad=True)
        nn.init.uniform_(self.noise.data)

    def forward1(self, V):
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / (self.gate_tau.clamp(min=1e-3))
        v1 = ego_gate * v1 + v1
        v2 = ego_gate * v2 + v2
        return ego, v1, v2

    def forward2(self, V):
        ego_V = V
        _, ego_v1, ego_v2 = torch.chunk(ego_V, chunks=3, dim=-1)
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / (self.gate_tau.clamp(min=1e-3))
        noise_v1, noise_v2 = torch.chunk(self.noise, chunks=2, dim=-1)
        v1 = (ego_gate * v1 + v1 + noise_v1) + ego_v1
        v2 = (ego_gate * v2 + v2 + noise_v2) + ego_v2
        return ego, v1, v2
