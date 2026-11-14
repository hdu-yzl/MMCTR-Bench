"""ID-anchored alignment objectives from the AntM2C diagnostic protocol."""

from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from mmctr.core import ContractError


METHODS = ("kl", "contrastive", "cosine", "mmd", "adversarial")


@contextmanager
def frozen_parameters(module):
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), previous):
            parameter.requires_grad_(requires_grad)


class LegacyAlignmentObjective(torch.nn.Module):
    """Align each content modality with ID; preserve original non-Adv formulas.

    Adversarial discriminator updates are explicit and use detached representations.
    Generator updates freeze discriminator weights while preserving input gradients.
    This repairs the historical wrapper's missing discriminator optimization.
    """

    def __init__(
        self, method: str, modalities: Sequence[str], dimension: int, weight: float
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ContractError("unsupported legacy alignment objective")
        if "id" not in modalities or len(set(modalities)) != len(modalities):
            raise ContractError("alignment needs unique modalities including id")
        if len(modalities) < 2 or dimension <= 0:
            raise ContractError("alignment requires ID plus content and positive width")
        if not 0 <= weight or not torch.isfinite(torch.tensor(weight)):
            raise ContractError("alignment weight must be finite and non-negative")
        self.method = method
        self.non_id = tuple(name for name in modalities if name != "id")
        self.dimension = dimension
        self.weight = float(weight)
        self.discriminators = torch.nn.ModuleDict()
        if method == "adversarial":
            for name in self.non_id:
                self.discriminators[name] = torch.nn.Sequential(
                    torch.nn.Linear(dimension, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, 1),
                )

    def _validate(self, features: Mapping[str, torch.Tensor]) -> None:
        if set(features) != {"id", *self.non_id}:
            raise ContractError("alignment representations do not match configured modalities")
        shape = features["id"].shape
        if len(shape) != 2 or shape[0] == 0 or shape[1] != self.dimension:
            raise ContractError("alignment expects nonempty [batch, projection_dim] features")
        if any(value.shape != shape for value in features.values()):
            raise ContractError("alignment feature shapes must match")

    @staticmethod
    def mmd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        total = torch.cat((left, right), dim=0)
        # Avoid allocating [2B,2B,D]; retain the original five-RBF-kernel objective.
        distance = torch.cdist(total, total, p=2).square()
        n = len(total)
        bandwidth = distance.detach().sum() / (n * n - n)
        bandwidth = bandwidth.clamp_min(torch.finfo(total.dtype).eps) / 4
        kernels = sum(torch.exp(-distance / (bandwidth * (2**index))) for index in range(5))
        b = len(left)
        return (
            kernels[:b, :b].mean()
            + kernels[b:, b:].mean()
            - kernels[:b, b:].mean()
            - kernels[b:, :b].mean()
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate(features)
        anchor = features["id"]
        if self.weight == 0:
            return anchor.sum() * 0
        losses = []
        for name in self.non_id:
            other = features[name]
            if self.method == "kl":
                # Avoid log(0) in KL's target-probability backward when softmax
                # underflows. Both branches remain differentiable.
                log_anchor = F.log_softmax(anchor, dim=-1)
                log_other = F.log_softmax(other, dim=-1)
                loss = (log_anchor.exp() * (log_anchor - log_other)).sum(-1).mean()
            elif self.method == "contrastive":
                logits = F.normalize(anchor, dim=-1) @ F.normalize(other, dim=-1).t() / 0.07
                labels = torch.arange(len(anchor), device=anchor.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            elif self.method == "cosine":
                loss = (1 - F.cosine_similarity(anchor, other, dim=-1)).clamp_min(0).mean()
            elif self.method == "mmd":
                loss = self.mmd(anchor, other)
            else:
                with frozen_parameters(self.discriminators[name]):
                    prediction = self.discriminators[name](other)
                    loss = F.binary_cross_entropy_with_logits(
                        prediction, torch.ones_like(prediction)
                    )
            losses.append(loss)
        return self.weight * torch.stack(losses).mean()

    def discriminator_loss(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.method != "adversarial":
            raise ContractError("discriminator loss only applies to adversarial alignment")
        self._validate(features)
        losses = []
        for name in self.non_id:
            real = self.discriminators[name](features["id"].detach())
            fake = self.discriminators[name](features[name].detach())
            losses.append(
                (
                    F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                    + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
                )
                / 2
            )
        return self.weight * torch.stack(losses).mean()
