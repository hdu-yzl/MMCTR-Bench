"""
Diff_MSIN 模型 —— 融合分析版本
原始模型硬编码使用 SRC 融合，此版本支持所有融合方法。

模型核心操作（保持不变）:
  - 模态专有专家 + 共享专家
  - 模态门控 + 二次门控
  - CrossNetwork + 多头交叉注意力
  - HingeCosineLoss (l_syn) + cosine distance (l_con)

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    替换 SRC 融合为可配置融合层 + 投影对齐。
  序列感知融合（dta, gmmf, dmf）:
    替代门控后特征融合步骤，输出经投影对齐。
"""
from models.base_seq_model import BaseSeqModel
from models.layers.common import MultiLayerPerceptron, CrossModalAttentionMH, CrossNetwork
from models.layers import modal_fusion, seq_pooling, losses
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_fusion_kwargs(method, model_config, projection_dim, mm_features):
    """根据融合方法构建额外参数（保留兼容）"""
    kwargs = {}
    if method == 'src':
        kwargs['T'] = model_config.get('T', 10)
    elif method == 'lmf':
        kwargs['rank'] = model_config.get('rank', 5)
        kwargs['output_dim'] = model_config.get('fusion_dim', projection_dim)
    elif method == 'mtfn':
        kwargs['rank'] = model_config.get('rank', 20)
    return kwargs


class Diff_MSIN(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.T = self.model_config.get('T', 10)
        self.num_cross_layers = self.model_config.get('num_cross_layers', 3)
        self.lambda1 = self.model_config.get('lambda1', 0.1)
        self.lambda2 = self.model_config.get('lambda2', 0.1)
        self.heads = self.model_config.get('heads', 4)
        self.mlp_dims = list(self.mlp_dims) + [self.projection_dim]
        self.din_mlp_dims = [64, 32]
        self.modal_poolings = torch.nn.ModuleDict({
            i: seq_pooling.get_pooling('din', dim=self.projection_dim,
                                       mlp_dims=self.din_mlp_dims)
            for i in self.mm_features
        })

        self.specific_experts = torch.nn.ModuleDict({
            i: MultiLayerPerceptron(self.projection_dim, self.mlp_dims,
                                    self.dropout, use_bn=self.bn, activation='relu')
            for i in self.mm_features
        })
        self.share_experts = MultiLayerPerceptron(self.projection_dim, self.mlp_dims,
                                                  self.dropout, use_bn=self.bn, activation='relu')
        self.modal_gates = torch.nn.ModuleDict({
            i: MultiLayerPerceptron(self.projection_dim, [self.projection_dim * 2, self.projection_dim],
                                    dropout=0, use_bn=False, activation='sigmoid')
            for i in self.mm_features
        })

        self.final_gate = MultiLayerPerceptron(self.projection_dim,
                                               [self.projection_dim * 2, self.projection_dim * (self.mm_nums + 1)],
                                               dropout=0, use_bn=False, activation='sigmoid')

        # ===== 融合方法适配 =====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'src'))
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
        self.compile()
        self.model_to_device()
        self.log_model_params()

    def _fuse_gated(self, E_dict):
        """对门控后的特征应用融合"""
        return fuse_local(self.modal_fusion_layer, E_dict, self.mm_features,
                          self._fusion_method, self._query_num)

    def forward(self, user_feats, feats, feats_seq, label):
        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

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

        # 4. 门控加权
        Target_gates = {k: self.modal_gates[k](E_target_specific[k]) for k in self.mm_features}
        Seq_gates = {k: self.modal_gates[k](E_seq_specific[k]) for k in self.mm_features}

        E_target = {k: Target_gates[k] * E_target_specific[k] + (1 - Target_gates[k]) * E_target_sh_share
                    for k in self.mm_features}
        E_seq = {k: Seq_gates[k] * E_seq_specific[k] + (1 - Seq_gates[k]) * E_seq_sh_share
                 for k in self.mm_features}

        # 5. 融合得到表示（适配不同融合方法）
        au_loss_extra = 0.0

        if self._is_seq_fusion:
            # 序列感知融合：此处E_target和E_seq已经是池化后的2D特征
            # 将E_seq重新构造为伪序列形式（仅1个时间步）
            E_seq_pseudo = {k: E_seq[k].unsqueeze(1) for k in self.mm_features}

            if self._fusion_method == 'dta':
                E_target_fused, _ = fuse_seq_dta(
                    self.modal_fusion_layer, self._sim_fusion,
                    E_target, E_seq_pseudo, self.mm_features, 1)
                E_seq_fused, _ = fuse_seq_dta(
                    self.modal_fusion_layer, self._sim_fusion,
                    E_seq, E_seq_pseudo, self.mm_features, 1)
            elif self._fusion_method == 'gmmf':
                # 用户侧（提前计算）
                user_feats_temp = {k: feats_p[k] for k in self.mm_features}
                user_vec_temp = torch.cat([E_target[k] for k in self.mm_features], dim=-1)
                E_target_fused, gmmf_loss1 = fuse_seq_gmmf(
                    self.modal_fusion_layer, self._gmmf_pooling,
                    E_target, E_seq_pseudo, user_vec_temp)
                E_seq_fused, gmmf_loss2 = fuse_seq_gmmf(
                    self.modal_fusion_layer, self._gmmf_pooling,
                    E_seq, E_seq_pseudo, user_vec_temp)
                au_loss_extra = gmmf_loss1 + gmmf_loss2
            elif self._fusion_method == 'dmf':
                E_target_fused, _ = fuse_seq_dmf(
                    self.modal_fusion_layer, E_target, E_seq_pseudo, 1)
                E_seq_fused, _ = fuse_seq_dmf(
                    self.modal_fusion_layer, E_seq, E_seq_pseudo, 1)

            E_target_fused = self.fusion_projector(E_target_fused)
            E_seq_fused = self.fusion_projector(E_seq_fused)
        else:
            # 本地融合
            E_target_fused_raw, cl1 = self._fuse_gated(E_target)
            E_seq_fused_raw, cl2 = self._fuse_gated(E_seq)
            E_target_fused = self.fusion_projector(E_target_fused_raw)
            E_seq_fused = self.fusion_projector(E_seq_fused_raw)
            if isinstance(cl1, torch.Tensor):
                au_loss_extra = cl1 + cl2

        l_syn = self.L_syn_loss(E_target_fused, E_seq_fused, label.squeeze(-1))

        # 6. 二次门控
        final_gates = self.final_gate(E_target['id'])
        gates = final_gates.view(final_gates.shape[0], self.mm_nums + 1, self.projection_dim)
        gate_list = torch.unbind(gates, dim=1)
        gate_dict = {}
        gate_idx = 0
        for m in self.mm_features:
            if m == 'id':
                continue
            gate_dict[m] = gate_list[gate_idx]
            gate_idx += 1
        gate_dict['share'] = gate_list[gate_idx]
        gate_dict['syn'] = gate_list[gate_idx + 1]

        E_seq_final = {k: gate_dict[k] * E_seq[k] for k in self.mm_features if k != 'id'}
        E_seq_final['share'] = E_seq_sh_share * gate_dict['share']
        E_seq_final['syn'] = E_seq_fused * gate_dict['syn']

        E = torch.stack(list(E_seq_final.values()), dim=1)

        # 7. CrossNetwork
        B = E.shape[0]
        E_flat = E.view(B, -1)
        cross_out = self.cross_network(E_flat)
        cross_mean = cross_out.view(B, self.num_gate_entries, self.projection_dim).mean(dim=1)
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
        total_au_loss = self.lambda1 * l_syn + self.lambda2 * l_con
        if isinstance(au_loss_extra, torch.Tensor):
            total_au_loss = total_au_loss + au_loss_extra
        out = {'pred': logit, 'au_loss': total_au_loss}
        return out

    def calculate_mean_cosine_distance(self, tensors_dic, mm_features):
        sim_sum = 0.0
        n = len(mm_features)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                key_i, key_j = mm_features[i], mm_features[j]
                cosine_sim = F.cosine_similarity(tensors_dic[key_i], tensors_dic[key_j], dim=1)
                sim_sum += cosine_sim.mean()
                count += 1
        return sim_sum / max(count, 1)

    def _predict_batch(self, batch):
        user_feats, feats, feats_seq, label = batch
        user_feats = {k: v.to(self.device) for k, v in user_feats.items()}
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}
        label = label.to(self.device)
        out = self.forward(user_feats, feats, feats_seq, label)
        return out, label
