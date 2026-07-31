import torch.nn as nn
import torch


class MaxPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, seq, dim)  -> (B, dim)
        return torch.max(x, self.dim)[0]


class SumPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, seq, dim)  -> (B, dim)
        return torch.sum(x, self.dim)


class MeanPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, seq, dim)  -> (B, dim)
        return torch.mean(x, self.dim)


# DIN Attention
class DIN_AttentionPooling(nn.Module):
    class _Dice(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.zeros(1))
            self.epsilon = 1e-9

        def forward(self, x):
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            norm_x = (x - mean) / torch.sqrt(var + self.epsilon)
            p = torch.sigmoid(norm_x)
            return self.alpha * x * (1 - p) + x * p

    class _ActivationUnit(nn.Module):
        def __init__(self, emb_dim, mlp_dims=None, dropout=0.0):
            super().__init__()
            mlp_dims = mlp_dims or [32, 16]
            layers = []
            in_dim = emb_dim * 4
            for hid in mlp_dims:
                layers += [nn.Linear(in_dim, hid),
                           DIN_AttentionPooling._Dice(),
                           nn.Dropout(dropout)]
                in_dim = hid
            layers.append(nn.Linear(in_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, query, behavior):
            # query: (B, 1, D)   behavior: (B, seq, D)
            seq_len = behavior.size(1)
            queries = query.expand(-1, seq_len, -1)  # (B, seq, D)
            diff = queries - behavior
            prod = queries * behavior
            concat = torch.cat([queries, behavior, diff, prod], dim=-1)  # (B, seq, 4D)
            return self.mlp(concat)  # (B, seq, 1)

    def __init__(self, dim, dropout=0.0, mlp_dims=None):
        super().__init__()
        self.active_unit = self._ActivationUnit(dim,
                                                mlp_dims=mlp_dims,
                                                dropout=dropout)

    def forward(self, query_ad, user_behavior):
        """
        :param query_ad:   目标 item 的 embedding  (B, D)
        :param user_behavior: 用户行为序列        (B, seq, D)
        :return: 加权求和后的用户兴趣向量         (B, D)
        """
        attn_score = self.active_unit(query_ad.unsqueeze(1),  # (B, 1, D)
                                      user_behavior)  # (B, seq, 1)
        attn_weight = torch.softmax(attn_score, dim=1)  # (B, seq, 1)
        output = (attn_weight * user_behavior).sum(dim=1)  # (B, D)
        return output


class Cosine_Weighted_Sum_Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_modal, seq_modal, eps=1e-8):
        target_modal = target_modal.unsqueeze(1)
        similarity_scores = torch.sum(target_modal * seq_modal, dim=-1)
        denominator = torch.sum(similarity_scores, dim=-1, keepdim=True)
        attention_weights = similarity_scores / (denominator + eps)
        attention_weights = nn.ReLU()(attention_weights)
        attention_weights = attention_weights.unsqueeze(-1)  # (B, L, 1)
        pooled_modal = torch.sum(attention_weights * seq_modal, dim=1)  # (B, D)

        return pooled_modal


class CrossAttention_Pooling(nn.Module):
    def __init__(self, dim, dropout=0):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, query, key, value):
        """
        query: (B, dim)
        key  : (B, seq_len, dim)
        value: (B, seq_len, dim)

        return: (B, dim)
        """
        B, L, D = key.shape
        # 把 query 广播到 seq_len 长度，再与 key concat
        q_expand = query.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)
        concat = torch.cat([key, q_expand], dim=-1)  # (B, L, 2D)

        logits = self.net(concat).squeeze(-1)  # (B, L)

        attn = nn.functional.softmax(logits, dim=-1)  # (B, L)

        # 加权求和
        out = torch.bmm(attn.unsqueeze(1), value).squeeze(1)  # (B, D)
        return out


_POOL_MAP = {
    'max': MaxPooling,
    'sum': SumPooling,
    'mean': MeanPooling,
    'din': DIN_AttentionPooling,
    'cos': Cosine_Weighted_Sum_Pooling,
    'cross_atten': CrossAttention_Pooling
}


def get_pooling(name, *args, **kwargs):
    """
    Examples:
        pool = get_pooling('max', dim=1)
        pool = get_pooling('attention', embedding_dim=64, mlp_dims=[32,16], dropout=0.1)
    """
    key = name.lower()
    if key not in _POOL_MAP:
        raise ValueError(f'Unknown pooling type: {name}. '
                         f'Available: {list(_POOL_MAP.keys())}')
    return _POOL_MAP[key](*args, **kwargs)
