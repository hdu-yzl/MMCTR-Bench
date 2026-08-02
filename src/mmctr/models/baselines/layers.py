"""Private layers used by migrated CTR baselines."""

from typing import Iterable, Optional

import torch
import torch.nn.functional as functional


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_count, embedding_dim))
        torch.nn.init.xavier_uniform_(self.embedding)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.embedding(values, self.embedding)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        dropout: float,
        batch_norm: bool = False,
        activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        layers = []
        for output_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(output_dim))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation is not None:
                raise ValueError("unsupported activation: {!r}".format(activation))
            layers.append(torch.nn.Dropout(dropout))
            input_dim = output_dim
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, layers: int) -> None:
        super().__init__()
        self.weights = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, 1, bias=False) for _ in range(layers)]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(input_dim)) for _ in range(layers)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        initial = values
        for weight, bias in zip(self.weights, self.biases):
            values = initial * weight(values) + bias + values
        return values


class FactorizationMachine(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        square_of_sum = values.sum(dim=1).pow(2)
        sum_of_squares = values.pow(2).sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_squares).sum(dim=1, keepdim=True)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        heads: int,
        residual: bool,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.attention_dim = attention_dim
        output_dim = attention_dim * heads
        self.query = torch.nn.Linear(embedding_dim, output_dim, bias=False)
        self.key = torch.nn.Linear(embedding_dim, output_dim, bias=False)
        self.value = torch.nn.Linear(embedding_dim, output_dim, bias=False)
        self.residual = (
            torch.nn.Linear(embedding_dim, output_dim, bias=False) if residual else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, field_count, _ = values.shape

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(
                batch_size, field_count, self.heads, self.attention_dim
            ).transpose(1, 2)

        query = split_heads(self.query(values))
        key = split_heads(self.key(values))
        value = split_heads(self.value(values))
        scale = self.attention_dim ** 0.5
        weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / scale, dim=-1)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, field_count, -1)
        if self.residual is not None:
            output = output + self.residual(values)
        return torch.relu(output)


class Dice(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.epsilon = 1e-9

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=0)
        variance = values.var(dim=0)
        probability = torch.sigmoid((values - mean) / torch.sqrt(variance + self.epsilon))
        return self.alpha * values * (1 - probability) + values * probability


class DinAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims, dropout: float) -> None:
        super().__init__()
        layers = []
        input_dim = embedding_dim * 4
        for hidden_dim in hidden_dims:
            layers.extend(
                [torch.nn.Linear(input_dim, hidden_dim), Dice(), torch.nn.Dropout(dropout)]
            )
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(
        self,
        target: torch.Tensor,
        history: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = target.unsqueeze(1).expand(-1, history.shape[1], -1)
        inputs = torch.cat([targets, history, targets - history, targets * history], dim=-1)
        scores = self.network(inputs).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        return (weights.unsqueeze(-1) * history).sum(dim=1)


__all__ = [
    "CrossNetwork",
    "DinAttention",
    "FactorizationMachine",
    "FeatureEmbedding",
    "MultiHeadSelfAttention",
    "MultiLayerPerceptron",
]
