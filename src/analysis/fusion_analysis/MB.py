"""
MB 模型 —— 融合分析版本（强制兼容所有融合方法）
原始模型硬编码使用 _MBModalityFusion（带打分器的注意力融合）。
此版本支持所有本地融合 + 序列感知融合。

模型核心操作（保持不变）:
  - 双 ID 编码（user_id_encoder / item_id_encoder / id_pair_encoder）
  - 模态特定 user encoder（user_modal_encoder）
  - 物品模态编码 head（item_modal_encoder）
  - 模态分数（id_score + 各模态 score）求和
  - 模态平衡对抗损失（PGD 蒸馏）

适配方式:
  本地融合（maf, cat, lmf, src, mtfn, fq-former, simcen）:
    - 替换 self.fusion 为可配置融合层
    - 序列侧池化后同样使用本地融合
  序列感知融合（dta, gmmf, dmf）:
    - 跳过序列池化，直接对原始 3D 序列特征应用序列感知融合
    - 输出投影对齐至 projection_dim
"""
from models.base_model import BaseModel
from models.layers.common import MultiLayerPerceptron
from models.layers import seq_pooling
from analysis.fusion_analysis._fusion_helper import (
    build_fusion, fuse_local, fuse_seq_dta, fuse_seq_gmmf, fuse_seq_dmf,
    LOCAL_FUSIONS, SEQ_FUSIONS, normalize_fusion_method,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MB(BaseModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)

        self.modal_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_features]
        self.seq_modal_features = [k for k in ['image', 'text', 'audio'] if k in self.mm_seq_features]
        if not self.modal_features:
            raise ValueError('MB requires at least one of image/text/audio modalities.')

        # 序列感知融合所需的公共特征集（target 与 seq 都必须有）
        self.seq_fusion_features = [k for k in self.mm_features if k in self.mm_seq_features]

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

        # ===== 融合方法适配（同时支持本地融合 + 序列感知融合）=====
        self._fusion_method = normalize_fusion_method(
            model_config.get('modal_fusion_method', 'maf'))
        self._is_seq_fusion = self._fusion_method in SEQ_FUSIONS

        if self._is_seq_fusion:
            # 序列感知融合：直接在原始 3D 序列特征上工作，覆盖目标侧 + 序列侧
            f_info = build_fusion(self._fusion_method, self.projection_dim,
                                  self.seq_fusion_features, model_config)
            self.seq_fusion_layer_full = f_info['layer']
            self._seq_query_num_full = f_info['query_num']
            self._sim_fusion = f_info.get('sim_fusion')
            self._gmmf_pooling = f_info.get('seq_pooling')
            seq_out = f_info['out_dim']
            self.seq_fusion_projector_full = (
                nn.Linear(seq_out, self.projection_dim)
                if seq_out != self.projection_dim
                else nn.Identity())
            # 目标模态融合（item_modal_encoder 输出 → 简单 mean fusion 用于 fused_modal）
            self.target_fusion_layer = None
            self._target_query_num = None
        else:
            # 目标侧物品模态融合（仅本地融合）
            f_info_target = build_fusion(self._fusion_method, self.projection_dim,
                                         self.modal_features, model_config)
            self.target_fusion_layer = f_info_target['layer']
            self._target_query_num = f_info_target['query_num']
            target_out = f_info_target['out_dim']
            self.target_fusion_projector = (nn.Linear(target_out, self.projection_dim)
                                            if target_out != self.projection_dim
                                            else nn.Identity())

            # 序列侧（池化后）模态融合
            if self.seq_modal_features:
                f_info_seq = build_fusion(self._fusion_method, self.projection_dim,
                                          self.seq_modal_features, model_config)
                self.seq_fusion_layer = f_info_seq['layer']
                self._seq_query_num = f_info_seq['query_num']
                seq_out = f_info_seq['out_dim']
                self.seq_fusion_projector = (nn.Linear(seq_out, self.projection_dim)
                                             if seq_out != self.projection_dim
                                             else nn.Identity())
            else:
                self.seq_fusion_layer = None
                self._seq_query_num = None
                self.seq_fusion_projector = nn.Identity()

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

    def _fuse_target_local(self, modal_embs, fallback_id):
        if not modal_embs:
            return fallback_id, fallback_id.new_tensor(0.0)
        fused, cl = fuse_local(self.target_fusion_layer, modal_embs,
                               self.modal_features, self._fusion_method,
                               self._target_query_num)
        fused = self.target_fusion_projector(fused)
        cl_loss = cl if isinstance(cl, torch.Tensor) else fused.new_tensor(0.0)
        return fused, cl_loss

    def _fuse_seq_local(self, feats_seq_pool, fallback_id):
        seq_modal_embs = {
            k: self.item_modal_encoder[k](feats_seq_pool[k])
            for k in self.seq_modal_features if k in feats_seq_pool
        }
        if not seq_modal_embs or self.seq_fusion_layer is None:
            return fallback_id, fallback_id.new_tensor(0.0)
        fused, cl = fuse_local(self.seq_fusion_layer, seq_modal_embs,
                               self.seq_modal_features, self._fusion_method,
                               self._seq_query_num)
        fused = self.seq_fusion_projector(fused)
        cl_loss = cl if isinstance(cl, torch.Tensor) else fused.new_tensor(0.0)
        return fused, cl_loss

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

        user_raw, user_id, item_id, pair_id = self._encode_id(feats)
        modal_embs = self._encode_item_modalities(feats_p)

        au_loss = item_id.new_tensor(0.0)

        if self._is_seq_fusion:
            # 在原始 3D 序列上应用序列感知融合
            target_dict = {k: feats_p[k] for k in self.seq_fusion_features}
            seq_dict = {k: feats_seq_p[k] for k in self.seq_fusion_features}

            if self._fusion_method == 'dta':
                fused_seq, _ = fuse_seq_dta(self.seq_fusion_layer_full, self._sim_fusion,
                                            target_dict, seq_dict,
                                            self.seq_fusion_features, self.seq_len)
            elif self._fusion_method == 'gmmf':
                user_vec_tmp = torch.cat([target_dict[k] for k in self.seq_fusion_features], dim=-1)
                fused_seq, gmmf_loss = fuse_seq_gmmf(self.seq_fusion_layer_full, self._gmmf_pooling,
                                                    target_dict, seq_dict, user_vec_tmp)
                au_loss = au_loss + gmmf_loss
            else:  # dmf
                fused_seq, _ = fuse_seq_dmf(self.seq_fusion_layer_full,
                                            target_dict, seq_dict, self.seq_len)

            feats_seq_pool_fusion = self.seq_fusion_projector_full(fused_seq)

            # 目标侧：序列感知融合下，对物品模态使用简单平均（或 item_id 退路）
            if modal_embs:
                fused_modal = torch.stack(list(modal_embs.values()), dim=1).mean(dim=1)
            else:
                fused_modal = item_id
        else:
            fused_modal, target_cl = self._fuse_target_local(modal_embs, item_id)
            feats_seq_pool = {k: self.pooling(feats_seq_p[k]) for k in self.mm_seq_features}
            feats_seq_pool_fusion, seq_cl = self._fuse_seq_local(feats_seq_pool, item_id)
            au_loss = au_loss + target_cl + seq_cl

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

        if self.training and label is not None and self._can_balance(modal_embs):
            au_loss = au_loss + self._modality_balance_loss(feats, label, user_raw, modal_embs)

        out = {
            'pred': logit,
            'au_loss': au_loss,
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
