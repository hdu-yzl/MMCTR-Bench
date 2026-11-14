"""Explicit alignment composition without replacing native model training phases."""

from dataclasses import replace
from typing import Callable, Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.training.engine import AlternatingPhase
from mmctr.training.optimizers import PhasedAdam

from .legacy_alignment import LegacyAlignmentObjective


class AlignmentModel(torch.nn.Module):
    """Keep the native objective and add ID-anchored diagnostic alignment."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        objective: LegacyAlignmentObjective,
        extract: Callable[[torch.nn.Module, Batch], Mapping[str, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.alignment = objective
        self.extract = extract

    def forward(self, batch: Batch) -> ModelOutput:
        output = self.backbone(batch)
        if not self.training or self.alignment.weight == 0:
            return output
        losses = dict(output.auxiliary_losses)
        if "diagnostic_alignment" in losses:
            raise ContractError("native model already defines diagnostic_alignment")
        losses["diagnostic_alignment"] = self.alignment(self.extract(self.backbone, batch))
        return replace(output, auxiliary_losses=losses)

    def discriminator_loss(self, batch: Batch) -> torch.Tensor:
        # Recompute after the main update; no stale graph or backbone gradients.
        with torch.no_grad():
            features = self.extract(self.backbone, batch)
        return self.alignment.discriminator_loss(features)


def attach_alignment(trainer, objective: LegacyAlignmentObjective, extract) -> AlignmentModel:
    """Attach before training; preserve native optimizer groups and alternating phases.

    The discriminator uses the main-table learning rate and weight decay. All native
    groups retain their own optimizer options, including GMMF's separate rates.
    """
    engine = trainer.engine
    old = engine.optimizer
    if old.state:
        raise ContractError("alignment must be attached before any optimizer updates")
    if not isinstance(old, torch.optim.Adam):
        raise ContractError("legacy alignment requires the main-table Adam optimizer")
    if isinstance(engine.model, AlignmentModel):
        raise ContractError("alignment is already attached")
    objective.to(next(engine.model.parameters()).device)
    wrapped = AlignmentModel(engine.model, objective, extract)
    if objective.method == "adversarial" and objective.weight > 0:
        groups = {}
        rates = {}
        for index, group in enumerate(old.param_groups):
            phase = group.get("phase", "main")
            if phase in groups or (index > 0 and "phase" not in group):
                raise ContractError("optimizer needs unique explicit phases")
            groups[phase] = group["params"]
            rates[phase] = group["lr"]
        phase = "alignment_discriminator"
        if phase in groups or any(p.name == phase for p in engine.alternating_phases):
            raise ContractError("alignment discriminator phase already exists")
        groups[phase] = tuple(objective.parameters())
        rates[phase] = rates["main"]
        optimizer = PhasedAdam(groups, rates)
        for previous, current in zip(old.param_groups, optimizer.param_groups):
            current.update({k: v for k, v in previous.items() if k != "params"})
        main = next(g for g in old.param_groups if g.get("phase", "main") == "main")
        optimizer.param_groups[-1].update(
            {k: v for k, v in main.items() if k not in ("params", "phase")}
        )
        engine.optimizer = optimizer
        engine.alternating_phases = tuple(engine.alternating_phases) + (
            AlternatingPhase(phase, 0, wrapped.discriminator_loss),
        )
    trainer.model = wrapped
    engine.model = wrapped
    return wrapped
