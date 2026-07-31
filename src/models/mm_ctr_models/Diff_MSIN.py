from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron, CrossModalAttentionMH, CrossNetwork
from ..layers import modal_fusion, seq_pooling, losses
import torch
import torch.nn.functional as F


class Diff_MSIN(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.T = self.model_config.get('T', 10)
        self.num_cross_layers = self.model_config.get('num_cross_layers', 3)
        self.lambda1 = self.model_config.get('lambda1', 0.1)
        self.lambda2 = self.model_config.get('lambda2', 0.1)
        self.heads = self.model_config.get('heads', 4)
        self.mlp_dims = list(self.mlp_dims) + [self.projection_dim]  # 创建新列表，避免原地修改 config
        self.din_mlp_dims = [64, 32]  # DIN 注意力评分网络使用轻量 MLP，避免过度参数化
        self.modal_poolings = torch.nn.ModuleDict({i: seq_pooling.get_pooling('din', dim=self.projection_dim,
                                                                              mlp_dims=self.din_mlp_dims) for i in
                                                   self.mm_features})

        self.specific_experts = torch.nn.ModuleDict({i: MultiLayerPerceptron(self.projection_dim, self.mlp_dims,
                                                                             self.dropout,
                                                                             use_bn=self.bn, activation='relu') for i in
                                                     self.mm_features})
        self.share_experts = MultiLayerPerceptron(self.projection_dim, self.mlp_dims,
                                                  self.dropout, use_bn=self.bn, activation='relu')
        self.modal_gates = torch.nn.ModuleDict(
            {i: MultiLayerPerceptron(self.projection_dim, [self.projection_dim * 2, self.projection_dim],
                                     dropout=0, use_bn=False, activation='sigmoid') for i in self.mm_features})

        self.final_gate = MultiLayerPerceptron(self.projection_dim,
                                               [self.projection_dim * 2, self.projection_dim * (self.mm_nums + 1)],
                                               dropout=0, use_bn=False, activation='sigmoid')

        self.src_fusion = modal_fusion.get_fusion_layer('src', self.projection_dim, self.mm_features, T=self.T)
        self.L_syn_loss = losses.HingeCosineLoss()

        self.multi_head_attention = CrossModalAttentionMH(self.projection_dim, self.heads)
        self.num_gate_entries = self.mm_nums + 1  # 非id模态 + share + syn
        self.cross_network = CrossNetwork(self.num_gate_entries * self.projection_dim, self.num_cross_layers)

        self.dnn = MultiLayerPerceptron((self.mm_nums + 2 + self.user_features_num) * self.projection_dim,
                                        self.mlp_dims, self.dropout,
                                        use_bn=self.bn, activation='relu')
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq, label):
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # 1. 对序列进行DIN
        feats_seq_p_pooling = {k: self.modal_poolings[k](feats_p[k], feats_seq_p[k]) for k in self.mm_features}

        # 2. 过各自专家
        E_target_specific = {k: self.specific_experts[k](feats_p[k]) for k in self.mm_features}
        E_seq_specific = {k: self.specific_experts[k](feats_seq_p_pooling[k]) for k in self.mm_features}

        # 3. 过共享专家
        E_target_share = {k: self.share_experts(feats_p[k]) for k in self.mm_features}
        E_seq_share = {k: self.share_experts(feats_seq_p_pooling[k]) for k in self.mm_features}

        E_target_sh_share = sum(E_target_share[k] for k in self.mm_features) / len(self.mm_features)
        E_seq_sh_share = sum(E_seq_share[k] for k in self.mm_features) / len(self.mm_features)

        # 3.5 计算L_con
        Target_L_con = self.calculate_mean_cosine_distance(E_target_specific, self.mm_features) \
                       - self.calculate_mean_cosine_distance(E_target_share, self.mm_features)
        Seq_L_con = self.calculate_mean_cosine_distance(E_seq_specific, self.mm_features) \
                    - self.calculate_mean_cosine_distance(E_seq_share, self.mm_features)
        l_con = (Target_L_con + Seq_L_con) / 2

        # 4. 门控加权，得到加权表示
        Target_gates = {k: self.modal_gates[k](E_target_specific[k]) for k in self.mm_features}
        Seq_gates = {k: self.modal_gates[k](E_seq_specific[k]) for k in self.mm_features}

        E_target = {k: Target_gates[k] * E_target_specific[k] + (1 - Target_gates[k]) * E_target_sh_share for k in
                    self.mm_features}
        E_seq = {k: Seq_gates[k] * E_seq_specific[k] + (1 - Seq_gates[k]) * E_seq_sh_share for k in self.mm_features}

        # 5.src 融合得到表示
        E_target_src = self.src_fusion(E_target)  # (B,dim)
        E_seq_src = self.src_fusion(E_seq)  # (B,dim)

        l_syn = self.L_syn_loss(E_target_src, E_seq_src, label.squeeze(-1))

        # 6.二次门控
        final_gates = self.final_gate(E_target['id'])
        gates = final_gates.view(final_gates.shape[0], self.mm_nums + 1, self.projection_dim)  # (B,mm_nums+1,dim)
        gate_list = torch.unbind(gates, dim=1)
        gate_dict = {}
        # 按顺序为非id模态、share、syn 分配门控（修复索引碰撞 bug）
        gate_idx = 0
        for m in self.mm_features:
            if m == 'id':
                continue
            gate_dict[m] = gate_list[gate_idx]
            gate_idx += 1
        gate_dict['share'] = gate_list[gate_idx]
        gate_dict['syn'] = gate_list[gate_idx + 1]

        E_seq_final = {k: gate_dict[k] * E_seq[k] for k in self.mm_features if k != 'id'}
        E_seq_final['share'] = E_seq_sh_share * gate_dict['share']  # (B,dim)
        E_seq_final['syn'] = E_seq_src * gate_dict['syn']

        E = torch.stack(list(E_seq_final.values()), dim=1)  # (B, num_gate_entries, dim)

        # 7. CrossNetwork: 在展平特征上做跨特征交叉
        B = E.shape[0]
        E_flat = E.view(B, -1)  # (B, num_gate_entries * dim)
        cross_out = self.cross_network(E_flat)  # (B, num_gate_entries * dim)
        cross_mean = cross_out.view(B, self.num_gate_entries, self.projection_dim).mean(dim=1)  # (B, dim)
        atten_id = self.multi_head_attention(E_seq['id'], cross_mean)

        vec = torch.cat([cross_out, atten_id], dim=-1)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        final_vec = torch.cat([vec, user_vec], dim=1)
        dnn_out_put = self.dnn(final_vec)
        logit = self.out_put(dnn_out_put)
        out = {'pred': logit, 'au_loss': self.lambda1 * l_syn + self.lambda2 * l_con}
        return out

    def calculate_mean_cosine_distance(self, tensors_dic, mm_features):
        sim_sum = 0.0
        n = len(mm_features)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                key_i, key_j = mm_features[i], mm_features[j]
                cosine_sim = F.cosine_similarity(tensors_dic[key_i], tensors_dic[key_j], dim=1)  # (B,)
                sim_sum += cosine_sim.mean()  # 取 batch 平均
                count += 1
        return sim_sum / max(count, 1)  # 按模态对数归一化，避免模态数量影响损失尺度

    def _predict_batch(self, batch):
        user_feats, feats, feats_seq, label = batch
        user_feats = {k: v.to(self.device) for k, v in user_feats.items()}
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}
        label = label.to(self.device)
        out = self(user_feats, feats, feats_seq, label)
        return out, label
