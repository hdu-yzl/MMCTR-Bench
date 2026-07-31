import torch
import torch.nn.functional as F
import torch.nn as nn
from .activation import activation_layer


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_num, latent_dim, initializer=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_num, latent_dim))
        initializer(self.embedding)

    def forward(self, x):
        """
        :param x: tensor of size (batch_size, num_fields)
        :return: tensor of size (batch_size, num_fields, embedding_dim)
        """
        return F.embedding(x, self.embedding)


class FeaturesLinear(torch.nn.Module):
    def __init__(self, feature_num, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(feature_num, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return : tensor of size (batch_size, 1)
        """
        return torch.sum(torch.squeeze(self.fc(x)), dim=1, keepdim=True) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1) if reduce_sum
                  tensor of size (batch_size, embed_dim) else
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, use_bn=False, use_ln=False, activation='relu'):
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, mlp_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(mlp_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(mlp_dim))
            if activation is not None:
                layers.append(activation_layer(activation))
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        return self.mlp(x)


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields*embed_dim)``
        :return : tensor of size (batch_size, num_fields*embed_dim)
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class MultiHeadSelfAttention(torch.nn.Module):
    """AutoInt 的交互层：多头自注意力 + 残差连接

    对特征域（field）进行显式的高阶特征交互建模。
    """

    def __init__(self, embed_dim, attn_dim, num_heads=2, use_residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.use_residual = use_residual

        self.W_q = torch.nn.Linear(embed_dim, attn_dim * num_heads, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, attn_dim * num_heads, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, attn_dim * num_heads, bias=False)
        if use_residual:
            self.W_res = torch.nn.Linear(embed_dim, attn_dim * num_heads, bias=False)
        else:
            self.W_res = None

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size ``(batch_size, num_fields, attn_dim*num_heads)``
        """
        B, F_num, _ = x.shape
        q = self.W_q(x).view(B, F_num, self.num_heads, self.attn_dim).transpose(1, 2)  # (B,H,F,d)
        k = self.W_k(x).view(B, F_num, self.num_heads, self.attn_dim).transpose(1, 2)
        v = self.W_v(x).view(B, F_num, self.num_heads, self.attn_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attn_dim ** 0.5)  # (B,H,F,F)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,F,d)
        out = out.transpose(1, 2).contiguous().view(B, F_num, self.num_heads * self.attn_dim)  # (B,F,H*d)

        if self.W_res is not None:
            out = out + self.W_res(x)
        out = torch.relu(out)
        return out


class InnerProduct(torch.nn.Module):
    def __init__(self, field_num):
        super().__init__()
        self.rows = []
        self.cols = []
        for row in range(field_num):
            for col in range(row + 1, field_num):
                self.rows.append(row)
                self.cols.append(col)
        self.rows = torch.tensor(self.rows)
        self.cols = torch.tensor(self.cols)

    def forward(self, x):
        """
        :param x: Float tensor of size (batch_size, field_num, embedding_dim)
        :return: (batch_size, field_num*(field_num-1)/2)
        """
        batch_size = x.shape[0]
        trans_x = torch.transpose(x, 1, 2)

        self.rows = self.rows.to(trans_x.device)
        self.cols = self.cols.to(trans_x.device)

        gather_rows = torch.gather(trans_x, 2, self.rows.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans_x, 2, self.cols.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding


class CrossModalAttentionMH(nn.Module):
    """
    多头交叉注意力，输入 x1, x2: (B, dim)
    输出 (B, dim)
    """

    def __init__(self, dim=512, heads=8, return_attn=False):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.d_k = dim // heads
        self.scale = self.d_k ** -0.5
        self.return_attn = return_attn
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
        x1: (B, dim)  查询
        x2: (B, dim)  键/值
        return: (B, dim)
        """
        B, dim = x1.shape
        h = self.heads

        q = self.to_q(x1).view(B, h, self.d_k)
        k = self.to_k(x2).view(B, h, self.d_k)
        v = self.to_v(x2).view(B, h, self.d_k)

        dots = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)  # (B, h)

        out = attn.unsqueeze(-1) * v
        out = out.view(B, dim)  # 拼接回 (B, dim)

        out = self.to_out(out)
        if self.return_attn:
            return out, attn
        return self.gamma * out + x1

class Discretizer(nn.Module):
    """将相似度分数离散化到桶中，以实现高效的嵌入查找"""

    def __init__(self, num_buckets=35, range_min=-1.0, range_max=1.0):
        super().__init__()
        self.num_buckets = num_buckets
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, similarity_scores):
        # 将分数限制在范围内并缩放到桶索引
        scores = torch.clamp(similarity_scores, self.range_min, self.range_max)
        normalized = (scores - self.range_min) / (self.range_max - self.range_min)
        bucket_indices = (normalized * self.num_buckets).long().clamp(0, self.num_buckets - 1)
        return bucket_indices  # (B, L)

class SimTier(nn.Module):
    """模态中心的相似度直方图"""

    def __init__(self, tier_num=10, range_min=-1.0, range_max=1.0):
        super().__init__()
        self.tier_num = tier_num
        boundaries = torch.linspace(range_min, range_max, steps=tier_num + 1)
        self.register_buffer('boundaries', boundaries)

    def forward(self,  similarity_scores):
        """计算余弦相似度的直方图"""

        # 向量化直方图计算
        indices = torch.bucketize(similarity_scores, self.boundaries, right=False)
        indices = torch.clamp(indices - 1, 0, self.tier_num - 1)

        tier_counts = torch.zeros(similarity_scores.size(0), self.tier_num, device=similarity_scores.device)
        tier_counts.scatter_add_(1, indices, torch.ones_like(similarity_scores))

        return tier_counts.float()  # (B, tier_num)