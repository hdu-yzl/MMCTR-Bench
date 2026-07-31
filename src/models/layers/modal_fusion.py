import torch
import torch.nn.functional as F
import torch.nn as nn
from models.layers.activation import activation_layer
from torch.nn.init import xavier_normal, zeros_
from .common import CrossModalAttentionMH, Discretizer, FeatureEmbedding
import math


class LMF(nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image'], rank=5, output_dim=16):
        super(LMF, self).__init__()
        self.rank = rank
        self.output_dim = output_dim
        self.mm_features = mm_features

        self.factors = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.rank, dim + 1, self.output_dim))
            for i in mm_features
        ])

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        for factor in self.factors:
            xavier_normal(factor)
        xavier_normal(self.fusion_weights)
        zeros_(self.fusion_bias)

    def getDim(self):
        return self.output_dim

    def forward(self, mm_feats: dict):
        """
        Args:
            modalities: Dict of tensors, each [batch_size, dim]
        Returns:
            fused output: [batch_size, output_dim]
        """
        batch_size = mm_feats[self.mm_features[0]].size(0)
        device = mm_feats[self.mm_features[0]].device
        dtype = mm_feats[self.mm_features[0]].dtype

        z_with_bias = []
        for z in mm_feats.values():
            ones = torch.ones(batch_size, 1, device=device, dtype=dtype)
            z_with_bias.append(torch.cat([ones, z], dim=1))  # [B, dim+1]

        fusion_terms = []
        for i, z in enumerate(z_with_bias):
            term = torch.einsum('rdo,bd->bro', self.factors[i], z)  # [B, rank, O]
            fusion_terms.append(term)

        fused = torch.prod(torch.stack(fusion_terms, dim=0), dim=0)  # [B, rank, dim]
        # print(f"fused_shape:{fused.shape}")
        weights = self.fusion_weights.unsqueeze(-1)  # [1, rank, 1]
        output = (fused * weights).sum(dim=1) + self.fusion_bias  # [B, dim]

        return output


class MAF(torch.nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image'], activation='tanh'):
        super(MAF, self).__init__()
        self.dim = dim
        self.mm_features = mm_features
        self.W = nn.ParameterDict({
            m: nn.Parameter(torch.empty(self.dim, self.dim))
            for m in self.mm_features
        })
        self.b = nn.ParameterDict({
            m: nn.Parameter(torch.empty(self.dim))
            for m in self.mm_features
        })

        for m in self.mm_features:
            nn.init.xavier_uniform_(self.W[m])
            nn.init.zeros_(self.b[m])

        self.act = activation_layer(activation)

    def getDim(self):
        return self.dim

    def forward(self, mm_feats: dict):
        """
        Args:
            mm_feats: Dict of tensors, each [batch_size, dim] or [batch_size, seq_num, dim]
        Returns:
            fused output: [batch_size, output_dim] or [batch_size, seq_num, output_dim]
        """
        outputs = []
        for m in self.mm_features:
            if m not in mm_feats:
                raise ValueError(f"{m} modal is not in the input dict")
            transformed = self.act(mm_feats[m] @ self.W[m] + self.b[m])
            outputs.append(transformed)

        return torch.sum(torch.stack(outputs, dim=0), dim=0)


class CAT(torch.nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image']):
        super(CAT, self).__init__()
        self.dim = dim
        self.mm_features = mm_features

    def getDim(self):
        return self.dim * len(self.mm_features)

    def forward(self, mm_feats: dict):
        return torch.cat([mm_feats[m] for m in self.mm_features], dim=-1)


class ADD(torch.nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image']):
        super(ADD, self).__init__()
        self.dim = dim
        self.mm_features = mm_features

    def getDim(self):
        return self.dim

    def forward(self, mm_feats: dict):
        return sum([mm_feats[m] for m in self.mm_features])


class MEAN(torch.nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image']):
        super(MEAN, self).__init__()
        self.dim = dim
        self.mm_features = mm_features

    def getDim(self):
        return self.dim

    def forward(self, mm_feats: dict):
        return sum([mm_feats[m] for m in self.mm_features]) / len(self.mm_features)


class SRCModule(torch.nn.Module):
    """
    Stochastic Reverse Cross-modal fusion (SRC) — 改进版

    与原版的差异及原因
    ─────────────────────────────────────────────────────────
    原版：为每个有向模态对 (src→dst) 各建一个独立的 CrossModalAttentionMH，
          M 个模态共 M*(M-1) 个注意力模块；每个扩散步 T 内调用 M*(M-1) 次。

    改进：每个目标模态仅保留 1 个注意力模块，先通过 alpha 权重将所有源模态
          加权聚合为单一表示，再执行 1 次交叉注意力。

    ┌──────────┬──────────────────────────┬─────────────────────────┐
    │ 模态数 M │ 原版                     │ 改进版                  │
    ├──────────┼──────────────────────────┼─────────────────────────┤
    │ 3        │  6 个注意力, 每步 6 次   │  3 个注意力, 每步 3 次  │
    │ 4        │ 12 个注意力, 每步 12 次  │  4 个注意力, 每步 4 次  │
    └──────────┴──────────────────────────┴─────────────────────────┘

    改动原因：
    1. 原版 M*(M-1) 个独立注意力参数量随模态数平方增长，在模态较多的
       数据集（如 tiktok 4 模态）上容易过拟合、且计算开销显著；
    2. 反向扩散的核心语义是"目标模态从其他模态聚合信息来恢复自身"，
       关键在于目标模态如何接收信息，而非每个源模态用不同投影发送信息；
       每个 (src→dst) 方向独立建模引入了冗余；
    3. alpha_dict 的 softmax 归一化权重已足以对不同源模态做差异化加权，
       加权聚合后再做一次注意力即可捕获跨模态交互，无需逐源独立注意力。
    """

    def __init__(self, dim=64, mm_features=['id', 'text', 'image'], T=10):
        super().__init__()
        self.T = T
        self.dim = dim
        self.eps = 1e-8
        self.mm_features = mm_features
        angles = torch.linspace(0, 0.9 * (torch.pi / 2), T + 1)
        self.register_buffer('alpha_bars', torch.cos(angles) ** 2)
        self.register_buffer('alphas', self.alpha_bars[1:] / (self.alpha_bars[:-1] + self.eps))

        # 源模态权重：每个 (src, dst) 对一个可学习标量（与原版一致）
        alpha_dict = torch.nn.ParameterDict()
        for src in self.mm_features:
            for dst in self.mm_features:
                if src == dst:
                    continue
                name = f'alpha_{src}2{dst}'
                alpha_dict[name] = torch.nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_dict = alpha_dict

        # 【改动】交叉注意力：从 M*(M-1) 个缩减为 M 个（每个目标模态 1 个）
        # 原版每个 (src, dst) 方向各建一个独立注意力，改为每个 dst 共享一个，
        # 接收已通过 alpha 加权聚合的源模态表示
        self.attn_dict = torch.nn.ModuleDict({
            m: CrossModalAttentionMH(dim, heads=8) for m in self.mm_features
        })

        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(len(self.mm_features) * dim),
            torch.nn.Linear(len(self.mm_features) * dim, 2 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * dim, dim)
        )
        for layer in self.fusion_mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.1)

    def getDim(self):
        return self.dim

    def forward_diffusion(self, h, t):
        if not self.training:
            # 推理时不加噪声，保证确定性输出
            alpha_t = self.alphas[t]
            return h * torch.sqrt(alpha_t + self.eps), torch.zeros_like(h)
        noise = torch.randn_like(h)
        alpha_t = self.alphas[t]
        return (h * torch.sqrt(alpha_t + self.eps)
                + noise * torch.sqrt(torch.clamp(1 - alpha_t, min=self.eps))), noise

    def reverse_diffusion(self, h_m, h_others, t, dst_key):
        alpha_t = self.alphas[t]
        bar_alpha_t = self.alpha_bars[t + 1]  # 经过 t 步后的累积量应为 alpha_bars[t+1]

        # 收集所有源模态的 alpha 参数，用 softmax 归一化权重
        alpha_logits = []
        src_tensors = []
        for h_n, src_key in h_others:
            alpha_param = self.alpha_dict[f'alpha_{src_key}2{dst_key}']
            alpha_logits.append(alpha_param)
            src_tensors.append(h_n)
        alpha_weights = torch.softmax(torch.cat(alpha_logits), dim=0)  # (num_others,)

        # 【改动】先将源模态加权聚合为单一表示，再做 1 次交叉注意力
        # 原版：对每个源模态各做 1 次独立注意力（共 M-1 次），再加权求和
        # 改进：加权聚合后仅做 1 次注意力，复杂度从 O(M-1) 降为 O(1)
        h_agg = sum(w * h_n for w, h_n in zip(alpha_weights, src_tensors))
        eps_pred = self.attn_dict[dst_key](h_m, h_agg)

        denominator = torch.sqrt(torch.clamp(alpha_t, min=self.eps))
        factor = (1 - alpha_t) / torch.sqrt(torch.clamp(1 - bar_alpha_t, min=self.eps))

        return (h_m - factor * eps_pred) / denominator

    def forward(self, features):

        h_mm = {k: features[k] for k in self.mm_features}

        for t in range(self.T):
            h_mm_noisy = {}
            for m in self.mm_features:
                noisy, _ = self.forward_diffusion(h_mm[m], t)
                h_mm_noisy[m] = noisy

            next_x = {}
            for dst in self.mm_features:
                others = [(h_mm_noisy[src], src) for src in self.mm_features if src != dst]
                next_x[dst] = self.reverse_diffusion(h_mm_noisy[dst], others, t,
                                                     dst_key=dst)

            for m in self.mm_features:
                h_mm[m] = next_x[m]

        final_vec = torch.cat([h_mm[m] for m in self.mm_features], dim=-1)
        h_syn = self.fusion_mlp(final_vec)

        return h_syn


class MTFNFusion(torch.nn.Module):
    def __init__(self, dim, mm_features=['id', 'text', 'image'], rank=20, output_dim=None):
        super().__init__()

        self.output_dim = dim if output_dim is None else output_dim

        self.rank = rank
        self.mm_features = mm_features
        self.mm_heads = nn.ModuleDict({
            k: nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(rank)])
            for k in mm_features
        })

        self.compress = torch.nn.Linear(dim, self.output_dim)

    def getDim(self):
        return self.output_dim

    def forward(self, features: dict):
        """
        features: dict{modality: tensor[B, D]}
        """
        fs = []
        for r in range(self.rank):
            mm_h = [self.mm_heads[k][r](features[k]) for k in self.mm_features]
            fused = mm_h[0]
            for h in mm_h[1:]:
                fused = fused * h
            fs.append(fused)
        f = torch.stack(fs, dim=0).sum(0)
        return self.compress(f)


class DecoupledTargetAttention(nn.Module):
    """DTA: 解耦基于ID的路径和相似度路径"""

    def __init__(self, dim, mm_features=['id', 'text', 'image'], attention_dim=128, num_buckets=35, dropout=0):
        super().__init__()
        self.attention_dim = attention_dim
        self.mm_features = mm_features
        # 用于ID特征的目标无关线性投影
        self.W_q = nn.Linear(dim, attention_dim)
        self.W_k = nn.Linear(dim, attention_dim)
        self.W_v = nn.Linear(dim, attention_dim)

        # 目标感知的相似度嵌入
        self.discretizer = Discretizer(num_buckets=num_buckets)
        self.sim_embedding_k = FeatureEmbedding(num_buckets, attention_dim)
        self.sim_embedding_v = FeatureEmbedding(num_buckets, attention_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = attention_dim ** 0.5

    def getDim(self):
        return self.attention_dim

    def forward(self, target_id_emb, hist_id_emb, similarity_scores):
        """
        参数:
            target_id_emb: (B, embedding_dim) 目标物品ID嵌入
            hist_id_emb: (B, L, embedding_dim) 历史物品ID嵌入
            similarity_scores: (B, L) 融合的多模态相似度分数
        返回:
            user_interest: (B, attention_dim) 模态增强的表示
        """
        # 从目标ID生成Query
        Q = self.W_q(target_id_emb)  # (B, attention_dim)

        # 从ID特征生成Key/Value
        K_id = self.W_k(hist_id_emb)  # (B, L, attention_dim)
        V_id = self.W_v(hist_id_emb)  # (B, L, attention_dim)

        # 相似度嵌入（目标感知但通过查找实现高效计算）
        bucket_indices = self.discretizer(similarity_scores)  # (B, L)
        K_sim = self.sim_embedding_k(bucket_indices)  # (B, L, attention_dim)
        V_sim = self.sim_embedding_v(bucket_indices)  # (B, L, attention_dim)

        # 解耦融合：逐元素相加
        K = K_id + K_sim
        V = V_id + V_sim

        # 缩放点积注意力
        attention_scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        user_interest = torch.matmul(attention_weights, V).squeeze(1)  # (B, attention_dim)
        return user_interest


class FQFormer(nn.Module):
    class _BertAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, attn_drop, name="default"):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.name = name

            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.drop = nn.Dropout(attn_drop)

        def transpose_for_scores(self, x):
            new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
            return x.view(*new_shape).permute(0, 2, 1, 3)

        def forward(self, hidden, context, mask=None):
            q = self.transpose_for_scores(self.query(hidden))
            k = self.transpose_for_scores(self.key(context))
            v = self.transpose_for_scores(self.value(context))

            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size + 1e-5)
            if mask is not None:
                scores += mask
            probs = torch.softmax(scores, dim=-1)
            probs = self.drop(probs)

            out = torch.matmul(probs, v)
            out = out.permute(0, 2, 1, 3).contiguous()
            return out.view(*out.size()[:-2], self.hidden_size)

    class _BertAttOutput(nn.Module):
        def __init__(self, hidden_size, hidden_drop):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
            self.drop = nn.Dropout(hidden_drop)

        def forward(self, hidden, inp):
            hidden = self.dense(hidden)
            hidden = self.drop(hidden)
            return self.norm(hidden + inp)

    class _SelfattLayer(nn.Module):
        def __init__(self, hidden_size, num_heads, attn_drop, hidden_drop, name="default"):
            super().__init__()
            self.self_att = FQFormer._BertAttention(hidden_size, num_heads, attn_drop, name)
            self.out = FQFormer._BertAttOutput(hidden_size, hidden_drop)

        def forward(self, inp, mask):
            self_out = self.self_att(inp, inp, mask)
            return self.out(self_out, inp)

    def __init__(self, dim=64,
                 query_num=3,
                 layer_num=2,
                 num_heads=8,
                 attn_drop=0.,
                 hidden_drop=0.):
        super().__init__()
        self.query_num = query_num
        self.layer_num = layer_num
        self.hidden_size = dim
        self.num_heads = num_heads

        self.queries = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, query_num, self.hidden_size)),
            requires_grad=True
        )
        self.layers = nn.ModuleList([
            FQFormer._SelfattLayer(self.hidden_size, self.num_heads, attn_drop, hidden_drop, f'layer{i}')
            for i in range(self.layer_num)
        ])

    def getDim(self):
        return self.hidden_size * self.query_num

    def forward(self, x, attention_mask=None):
        B, L, D = x.size()
        queries = self.queries.expand(B, -1, -1)
        x = torch.cat([queries, x], dim=1)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GMMFFusion：DSN（解耦共享-私有网络）+ 模态兴趣门控
# 来源：GMMF 模型
# ─────────────────────────────────────────────────────────────────────────────

class _AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.relu(encoded), self.relu(decoded)


class _CGANGenerator(nn.Module):
    def __init__(self, id_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(id_dim, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, id_embedding):
        return self.net(id_embedding)


class _CGANDiscriminator(nn.Module):
    def __init__(self, modality_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(modality_dim * 4, modality_dim * 2), nn.ReLU(),
            nn.Linear(modality_dim * 2, 1), nn.Sigmoid(),
        )

    def forward(self, id_feat, modality_feat):
        x = torch.cat([id_feat, id_feat - modality_feat,
                        id_feat * modality_feat, modality_feat], dim=-1)
        return self.net(x)


class _AutoDifference(nn.Module):
    def __init__(self, modality_dim):
        super().__init__()
        self.weight_net = nn.Linear(modality_dim * 4, modality_dim)

    def forward(self, Hm, Hm_hat):
        concat_feat = torch.cat([Hm, Hm_hat, Hm * Hm_hat, Hm - Hm_hat], dim=-1)
        return Hm - torch.sigmoid(self.weight_net(concat_feat)) * Hm_hat


class _ModalInterestGate(nn.Module):
    def __init__(self, user_dim, modal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_dim, modal_dim), nn.ReLU(),
            nn.Linear(modal_dim, modal_dim),
        )

    def forward(self, user_embedding):
        return torch.softmax(self.net(user_embedding), dim=-1)


class GMMFFusion(nn.Module):
    """
    GMMF 融合模块（DSN + 模态兴趣门控），可复用。

    参数:
        dim           : 投影后的模态特征维度
        mm_features   : 模态名称列表（含 'id'）
        user_dim      : 用户侧拼接向量维度（用于门控，默认=dim）

    forward(feats_p, feats_seq_p, user_vec, seq_pooling_fn):
        feats_p       : dict  投影后目标物品特征           {k: (B, dim)}
        feats_seq_p   : dict  投影后序列特征               {k: (B, L, dim)}
        user_vec      : Tensor(B, user_dim)               用户向量
        seq_pooling_fn: callable (query, seq) -> pooled    序列池化函数

    返回 dict:
        'fused'           : Tensor(B, getDim())  门控后拼接向量
        'recon_loss_pairs': list[(recon, target)]  重建 MSE 对
        'H_m', 'H_m_hat', 'H_m_seq', 'H_m_seq_hat': dict  供 GAN 损失使用

    compute_disc_loss / compute_gen_loss: 供 GAN 训练调用。
    """

    def __init__(self, dim, mm_features=None, user_dim=None):
        super().__init__()
        if mm_features is None:
            mm_features = ['id', 'text', 'image']
        self.dim = dim
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']

        self.mm_ae   = nn.ModuleDict({k: _AutoEncoder(dim, dim) for k in self.non_id})
        self.mm_gen  = nn.ModuleDict({k: _CGANGenerator(dim, dim) for k in self.non_id})
        self.mm_disc = nn.ModuleDict({k: _CGANDiscriminator(dim) for k in self.non_id})
        self.mm_diff = nn.ModuleDict({k: _AutoDifference(dim) for k in self.non_id})

        _udim = user_dim if user_dim is not None else dim
        self.mm_gate = nn.ModuleDict({k: _ModalInterestGate(_udim, dim) for k in mm_features})

    def getDim(self):
        return self.dim * len(self.mm_features)

    def forward(self, feats_p, feats_seq_p, user_vec, seq_pooling_fn):
        H_m, H_m_hat, H_m_prime = {}, {}, {}
        H_m_seq, H_m_seq_hat, H_m_seq_prime = {}, {}, {}
        recon_pairs = []

        for k in self.non_id:
            H_m[k], recon = self.mm_ae[k](feats_p[k])
            H_m_seq[k], recon_seq = self.mm_ae[k](feats_seq_p[k])
            recon_pairs.append((recon, feats_p[k]))
            recon_pairs.append((recon_seq, feats_seq_p[k]))
            H_m_hat[k] = self.mm_gen[k](feats_p['id'])
            H_m_seq_hat[k] = self.mm_gen[k](feats_seq_p['id'])
            H_m_prime[k] = self.mm_diff[k](H_m[k], H_m_hat[k])
            H_m_seq_prime[k] = self.mm_diff[k](H_m_seq[k], H_m_seq_hat[k])

        pool = {k: seq_pooling_fn(H_m_prime[k], H_m_seq_prime[k]) for k in self.non_id}
        gate = {k: self.mm_gate[k](user_vec) for k in self.mm_features}

        tensors = []
        for k in self.mm_features:
            tensors.append((feats_p[k] if k == 'id' else pool[k]) * gate[k])

        return {
            'fused': torch.cat(tensors, dim=-1),
            'recon_loss_pairs': recon_pairs,
            'H_m': H_m, 'H_m_hat': H_m_hat,
            'H_m_seq': H_m_seq, 'H_m_seq_hat': H_m_seq_hat,
        }

    def compute_disc_loss(self, feats_p, feats_seq_p, H_m, H_m_hat, H_m_seq, H_m_seq_hat):
        bce = nn.BCELoss()
        loss = 0.0
        for k in self.non_id:
            real     = self.mm_disc[k](feats_p['id'], H_m[k])
            real_seq = self.mm_disc[k](feats_seq_p['id'], H_m_seq[k])
            fake     = self.mm_disc[k](feats_p['id'], H_m_hat[k])
            fake_seq = self.mm_disc[k](feats_seq_p['id'], H_m_seq_hat[k])
            loss += (bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))
                     + bce(real_seq, torch.ones_like(real_seq)) + bce(fake_seq, torch.zeros_like(fake_seq)))
        return loss

    def compute_gen_loss(self, feats_p, feats_seq_p):
        bce = nn.BCELoss()
        loss = 0.0
        for k in self.non_id:
            fake     = self.mm_disc[k](feats_p['id'], self.mm_gen[k](feats_p['id']))
            fake_seq = self.mm_disc[k](feats_seq_p['id'], self.mm_gen[k](feats_seq_p['id']))
            loss += bce(fake, torch.ones_like(fake)) + bce(fake_seq, torch.ones_like(fake_seq))
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# SimCENFusion：Segmentation + MLEP（合并为一个模块）
# 来源：SimCEN 模型
# ─────────────────────────────────────────────────────────────────────────────

class _LinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        assert output_dim % 3 == 0
        one_third = output_dim // 3
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.gate = nn.Sequential(nn.Linear(one_third, one_third, bias=True), nn.Sigmoid())
        self.gate_tau = nn.Parameter(torch.ones(one_third), requires_grad=True)
        self.noise = nn.Parameter(torch.empty(2 * one_third), requires_grad=True)
        nn.init.uniform_(self.noise.data)

    def forward1(self, V):
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / self.gate_tau.clamp(min=1e-3)
        v1 = ego_gate * v1 + v1
        v2 = ego_gate * v2 + v2
        return ego, v1, v2

    def forward2(self, V):
        _, ego_v1, ego_v2 = torch.chunk(V, chunks=3, dim=-1)
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / self.gate_tau.clamp(min=1e-3)
        noise_v1, noise_v2 = torch.chunk(self.noise, chunks=2, dim=-1)
        v1 = (ego_gate * v1 + v1 + noise_v1) + ego_v1
        v2 = (ego_gate * v2 + v2 + noise_v2) + ego_v2
        return ego, v1, v2


class SimCENFusion(nn.Module):
    """
    SimCEN 融合模块（Segmentation + MLEP），可复用。

    参数:
        num_fields    : 场数量
        embedding_dim : 每个场嵌入维度
        hidden_units  : MLEP 各隐层维度列表（每个须为 3 的倍数）
        ego_dropout / v1_dropout / v2_dropout : dropout 比例
        ego_batch_norm / v1_batch_norm / v2_batch_norm : 是否 BatchNorm
        cl_temperature: InfoNCE 温度

    forward(feature_emb):
        feature_emb : Tensor(B, num_fields, embedding_dim)

    返回 dict:
        'output' : Tensor(B, hidden_units[-1])  三路拼接 [ego, ego+v1, ego+v2]
        'ego'    : Tensor  ego 表示
        'v1'     : Tensor  ego + view1
        'v2'     : Tensor  ego + view2
        'cl_loss': Tensor  InfoNCE 对比损失
    """

    def __init__(self, num_fields, embedding_dim,
                 hidden_units=None,
                 ego_dropout=0.0, v1_dropout=0.0, v2_dropout=0.0,
                 ego_batch_norm=True, v1_batch_norm=True, v2_batch_norm=True,
                 cl_temperature=0.1):
        super().__init__()
        self.cl_temperature = cl_temperature
        flatten_dim = num_fields * embedding_dim

        # ── Segmentation ──
        self.kp_dim = int(num_fields * (num_fields + 1) / 2)
        self.kp_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim), requires_grad=True)
        nn.init.xavier_normal_(self.kp_W)
        self.triu_mask = nn.Parameter(
            torch.triu(torch.ones(num_fields, num_fields), 0).bool(), requires_grad=False)
        self.tril_mask = nn.Parameter(
            torch.tril(torch.ones(num_fields, num_fields), 0).bool(), requires_grad=False)
        self.project_triu = nn.Linear(self.kp_dim, flatten_dim, bias=False)
        self.project_tril = nn.Linear(self.kp_dim, flatten_dim, bias=False)

        # ── MLEP ──
        if hidden_units is None:
            hidden_units = [480, 480, 480]
        self.hidden_units = hidden_units
        if not isinstance(ego_dropout, list):
            ego_dropout = [ego_dropout] * len(hidden_units)
        if not isinstance(v1_dropout, list):
            v1_dropout = [v1_dropout] * len(hidden_units)
        if not isinstance(v2_dropout, list):
            v2_dropout = [v2_dropout] * len(hidden_units)

        input_dim = flatten_dim * 3
        self.layer = nn.ModuleList()
        self.norm_ego = nn.ModuleList()
        self.norm_v1 = nn.ModuleList()
        self.norm_v2 = nn.ModuleList()
        self.drop_ego = nn.ModuleList()
        self.drop_v1 = nn.ModuleList()
        self.drop_v2 = nn.ModuleList()
        self.act_ego = nn.ModuleList()
        self.act_v1 = nn.ModuleList()
        self.act_v2 = nn.ModuleList()
        dims = [input_dim] + hidden_units
        for idx in range(len(hidden_units)):
            one_third = hidden_units[idx] // 3
            self.layer.append(_LinearUnit(dims[idx], hidden_units[idx]))
            if ego_batch_norm:
                self.norm_ego.append(nn.BatchNorm1d(one_third))
            if v1_batch_norm:
                self.norm_v1.append(nn.BatchNorm1d(one_third))
            if v2_batch_norm:
                self.norm_v2.append(nn.BatchNorm1d(one_third))
            if ego_dropout[idx] > 0:
                self.drop_ego.append(nn.Dropout(ego_dropout[idx]))
            if v1_dropout[idx] > 0:
                self.drop_v1.append(nn.Dropout(v1_dropout[idx]))
            if v2_dropout[idx] > 0:
                self.drop_v2.append(nn.Dropout(v2_dropout[idx]))
            self.act_ego.append(nn.ReLU())
            self.act_v1.append(nn.ReLU())
            self.act_v2.append(nn.ReLU())

    def getDim(self):
        return self.hidden_units[-1]

    def forward(self, feature_emb):
        # Segmentation
        embs_kp = torch.matmul(torch.matmul(feature_emb, self.kp_W),
                                feature_emb.transpose(1, 2))
        triu = torch.masked_select(embs_kp, self.triu_mask).view(-1, self.kp_dim)
        tril = torch.masked_select(embs_kp, self.tril_mask).view(-1, self.kp_dim)
        ego_flat = feature_emb.flatten(start_dim=1)
        view1 = self.project_triu(triu)
        view2 = self.project_tril(tril)

        # MLEP
        V_i = torch.cat([ego_flat, view1, view2], dim=-1)
        for i in range(len(self.layer)):
            if i == 0:
                ego, v1, v2 = self.layer[i].forward1(V_i)
            else:
                ego, v1, v2 = self.layer[i].forward2(V_i)
            if len(self.norm_ego) > i:
                ego = self.norm_ego[i](ego)
            if len(self.norm_v1) > i:
                v1 = self.norm_v1[i](v1)
            if len(self.norm_v2) > i:
                v2 = self.norm_v2[i](v2)
            ego = self.act_ego[i](ego)
            v1 = self.act_v1[i](v1)
            v2 = self.act_v2[i](v2)
            if len(self.drop_ego) > i:
                ego = self.drop_ego[i](ego)
            if len(self.drop_v1) > i:
                v1 = self.drop_v1[i](v1)
            if len(self.drop_v2) > i:
                v2 = self.drop_v2[i](v2)
            V_i = torch.cat([ego, v1, v2], dim=-1)

        ego, v1, v2 = torch.chunk(V_i, chunks=3, dim=-1)
        v1 = ego + v1
        v2 = ego + v2
        output = torch.cat([ego, v1, v2], dim=-1)
        cl_loss = self._info_nce(ego, v1, v2)
        return {'output': output, 'ego': ego, 'v1': v1, 'v2': v2, 'cl_loss': cl_loss}

    def _info_nce(self, ego, v1, v2):
        ego = F.normalize(ego, dim=-1)
        v1 = F.normalize(v1, dim=-1)
        v2 = F.normalize(v2, dim=-1)
        pos = ((ego * v1).sum(-1) + (ego * v2).sum(-1)) * 0.5
        pos = torch.exp(pos / self.cl_temperature)
        ttl = torch.exp(torch.matmul(v1, v2.transpose(0, 1)) / self.cl_temperature).sum(-1)
        return (-torch.log(pos / ttl + 1e-6)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# DMFFusion：互补模态建模（DTA + SimTier 双路径加权融合）
# 来源：DMF 模型
# ─────────────────────────────────────────────────────────────────────────────

class _SimTier(nn.Module):
    def __init__(self, tier_num=10, range_min=-1.0, range_max=1.0):
        super().__init__()
        self.tier_num = tier_num
        self.register_buffer('boundaries', torch.linspace(range_min, range_max, steps=tier_num + 1))

    def forward(self, similarity_scores):
        idx = torch.bucketize(similarity_scores, self.boundaries, right=False)
        idx = torch.clamp(idx - 1, 0, self.tier_num - 1)
        counts = torch.zeros(similarity_scores.size(0), self.tier_num,
                             device=similarity_scores.device)
        counts.scatter_add_(1, idx, torch.ones_like(similarity_scores))
        return counts.float()


class DMFFusion(nn.Module):
    """
    DMF 互补模态建模融合模块，可复用。

    双路径：
        ME 路径：DTA（解耦目标注意力）→ MLP
        MC 路径：SimTier 相似度直方图  → MLP
    输出 = α * ME + (1-α) * MC

    参数:
        dim                 : 投影维度
        modal_fusion_method : 非 ID 模态融合方式（如 'cat'）
        mm_features         : 模态名称列表（含 'id'）
        attention_dim       : DTA 注意力维度
        num_buckets         : DTA 离散桶数
        tier_num            : SimTier 直方图桶数
        mlp_dims            : MLP 层维度
        dropout / bn        : MLP 参数
        alpha               : ME 路径权重

    forward(feats_p, feats_seq_p, seq_len):
        feats_p     : dict {k: (B, dim)}     投影后目标物品特征
        feats_seq_p : dict {k: (B, L, dim)}  投影后序列特征
        seq_len     : int                    序列长度
    返回:
        user_interest : Tensor(B, mlp_dims[-1])
    """

    def __init__(self, dim, modal_fusion_method='cat', mm_features=None,
                 attention_dim=128, num_buckets=35, tier_num=10,
                 mlp_dims=None, dropout=0.0, bn=True, alpha=0.5):
        super().__init__()
        if mm_features is None:
            mm_features = ['id', 'text', 'image']
        if mlp_dims is None:
            mlp_dims = [1024, 512, 256]
        from .common import MultiLayerPerceptron

        self.dim = dim
        self.mm_features = mm_features
        self.alpha = alpha
        self.non_id = [k for k in mm_features if k != 'id']

        self.modal_fusion = get_fusion_layer(modal_fusion_method, dim, self.non_id)
        self.dta = DecoupledTargetAttention(dim, mm_features, attention_dim, num_buckets, dropout)
        self.simtier = _SimTier(tier_num)
        self.me_mlp = MultiLayerPerceptron(attention_dim, mlp_dims, dropout, use_bn=bn)
        self.mc_mlp = MultiLayerPerceptron(tier_num, mlp_dims, dropout, use_bn=bn)

    def getDim(self):
        if hasattr(self.me_mlp, 'mlp'):
            for layer in reversed(self.me_mlp.mlp):
                if hasattr(layer, 'out_features'):
                    return layer.out_features
        return self.dim

    def forward(self, feats_p, feats_seq_p, seq_len):
        fused = self.modal_fusion({k: feats_p[k] for k in self.non_id})
        fused_seq = torch.stack(
            [self.modal_fusion({k: feats_seq_p[k][:, i] for k in self.non_id})
             for i in range(seq_len)], dim=1)

        sim = torch.cosine_similarity(fused.unsqueeze(1), fused_seq, dim=2)

        r_me = self.me_mlp(self.dta(feats_p['id'], feats_seq_p['id'], sim))
        r_mc = self.mc_mlp(self.simtier(sim))
        return self.alpha * r_me + (1 - self.alpha) * r_mc


# ─────────────────────────────────────────────────────────────────────────────
# 融合层注册表
# ─────────────────────────────────────────────────────────────────────────────
_FUSION_MAP = {
    'maf': MAF,
    'cat': CAT,
    'add': ADD,
    'mean': MEAN,
    'lmf': LMF,
    'src': SRCModule,
    'mtfn': MTFNFusion,
    'dta': DecoupledTargetAttention,
    'fq-former': FQFormer,
    'gmmf': GMMFFusion,
    'simcen': SimCENFusion,
    'dmf': DMFFusion,
}


def get_fusion_layer(fusion_type, *args, **kwargs):
    key = fusion_type.lower()
    if key not in _FUSION_MAP:
        raise ValueError(
            f'Unknown fusion type: {fusion_type}. '
            f'Available: {list(_FUSION_MAP.keys())}'
        )
    return _FUSION_MAP[key](*args, **kwargs)
