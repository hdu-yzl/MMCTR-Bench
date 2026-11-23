"""Candidate identity must survive empty and single-item history pooling."""

from dataclasses import replace

import pytest
import torch

from mmctr.models.mm_models.dmf import DMF
from mmctr.models.mm_models.psrq import MCCA
from tests.unit.test_canonical_quantization import configs, initialized_psrq, make_batch


@pytest.mark.parametrize("name", ["dmf", "psrq"])
@pytest.mark.parametrize("empty", [True, False])
def test_candidate_identity_receives_gradient_without_distinct_content(name, empty):
    torch.manual_seed(71)
    config, data = configs()
    model = DMF(config, data) if name == "dmf" else MCCA(config, data, initialized_psrq())
    model.eval()
    batch = make_batch(batch_size=1, all_padding=empty)
    # One valid history token makes normalized attention independent of the query.
    if not empty:
        batch = replace(batch, history_mask=torch.tensor([[False, False, True]]))
    batch = replace(batch, item_features={**batch.item_features, "id": torch.tensor([[20]])})
    changed = replace(batch, item_features={**batch.item_features, "id": torch.tensor([[21]])})
    first = model(batch).logits
    second = model(changed).logits
    assert not torch.equal(first, second)
    first.sum().backward()
    assert model.embedding.embedding.grad[20].abs().sum() > 0
    if name == "psrq":
        assert all(p.grad is None for p in model.quantizer.parameters())


@pytest.mark.parametrize("name", ["dmf", "psrq"])
def test_legacy_head_can_reproduce_candidate_invariance(name):
    torch.manual_seed(71)
    config, data = configs()
    config["target_id_to_head"] = False
    model = DMF(config, data) if name == "dmf" else MCCA(config, data, initialized_psrq())
    model.eval()
    batch = make_batch(batch_size=1, all_padding=True)
    changed = replace(batch, item_features={**batch.item_features, "id": torch.tensor([[21]])})
    assert torch.equal(model(batch).logits, model(changed).logits)
