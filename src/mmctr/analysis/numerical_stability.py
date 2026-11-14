"""Precision-only safeguards for diagnosed diagnostic-model overflows."""

from typing import Optional

import torch
import torch.nn.functional as F


class WideBatchNorm1d(torch.nn.BatchNorm1d):
    """Keep activations/parameters in their dtype, accumulate moments in float64."""

    running_mean: Optional[torch.Tensor]
    running_var: Optional[torch.Tensor]

    def __init__(self, source):
        super().__init__(
            source.num_features,
            source.eps,
            source.momentum,
            source.affine,
            source.track_running_stats,
        )
        self.load_state_dict(source.state_dict())
        if self.running_mean is not None and self.running_var is not None:
            self.running_mean = self.running_mean.double()
            self.running_var = self.running_var.double()
        self.train(source.training)

    def forward(self, values):
        self._check_input_dim(values)
        factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                factor = 1.0 / float(self.num_batches_tracked)
        training = self.training or self.running_mean is None
        result = F.batch_norm(
            values.double(),
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight.double() if self.weight is not None else None,
            self.bias.double() if self.bias is not None else None,
            training,
            factor,
            self.eps,
        )
        return result.to(values.dtype)


def widen_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm1d):
            setattr(module, name, WideBatchNorm1d(child))
        else:
            widen_batch_norm(child)
