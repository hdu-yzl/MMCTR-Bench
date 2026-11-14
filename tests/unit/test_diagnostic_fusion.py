import pytest
import torch

from mmctr.analysis.fusion_operators import PAPER_FUSIONS, build_diagnostic_fusion


@pytest.mark.parametrize("method", PAPER_FUSIONS)
@pytest.mark.parametrize("modalities", [("id", "text", "image"), ("id", "text", "image", "audio")])
def test_nine_fusions_gradient_batch_rank_and_padding(method, modalities):
    torch.manual_seed(2025)
    config = {
        "simcen_hidden_units": [48, 48],
        "dmf_mlp_dims": [32, 16],
        "attention_dim": 16,
        "T": 2,
    }
    fusion = build_diagnostic_fusion(method, 16, modalities, config)
    target = {n: torch.randn(4, 16, requires_grad=True) for n in modalities}
    history = {n: torch.randn(4, 5, 16, requires_grad=True) for n in modalities}
    mask = torch.tensor(
        [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    sequential = method in ("dta", "dmf")
    result = fusion(target, history, mask) if sequential else fusion(target)
    assert result.fused.shape == (4, fusion.output_dim)
    loss = result.fused.square().mean() + result.total_auxiliary_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in target.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in fusion.parameters())
    fusion.eval()
    if sequential:
        expected = fusion(target, history, mask).fused
        altered = {
            n: torch.where(mask.unsqueeze(-1), v, torch.full_like(v, 1000))
            for n, v in history.items()
        }
        actual = fusion(target, altered, mask).fused
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
        single = fusion(
            {n: v[:1] for n, v in target.items()}, {n: v[:1] for n, v in history.items()}, mask[:1]
        )
    else:
        single = fusion({n: v[:1] for n, v in target.items()})
    assert single.fused.shape == (1, fusion.output_dim)
    assert torch.isfinite(single.fused).all()


@pytest.mark.parametrize(
    "model_name",
    [
        "lmf",
        "mtfn",
        "gmmf",
        "simcen",
        "dmf",
        "make",
        "naml",
        "em3",
        "mmmlp",
        "m3srec",
        "mb",
        "pamd",
        "marn",
        "diff_msin",
    ],
)
@pytest.mark.parametrize("method", PAPER_FUSIONS)
def test_legacy_forced_wrapper_full_fusion_matrix(model_name, method):
    torch.manual_seed(2025)
    from mmctr.analysis.fusion_backbones import build_fusion_study_model
    from mmctr.core import Batch

    model_config = {
        "latent_dim": 16,
        "projection_dim": 16,
        "mlp_dims": [32, 16],
        "batch_norm": True,
        "dropout": 0.0,
        "simcen_hidden_units": [48, 48],
        "dmf_mlp_dims": [32, 16],
        "attention_dim": 16,
        "T": 2,
    }
    data = {
        "id_feature_num": 100,
        "use_mm_features": ["id", "text", "image"],
        "use_mm_seq_features": ["id", "text", "image"],
        "mm_dims": {"text": 8, "image": 6},
        "mm_seq_dims": {"text": 8, "image": 6},
        "user_features": ["id"],
        "user_feature_dims": {},
        "seq_len": 5,
    }
    if model_name == "pamd" and method == "src":
        from mmctr.core import ContractError

        with pytest.raises(ContractError, match="two modalities"):
            build_fusion_study_model(model_name, method, model_config, data)
        return
    model = build_fusion_study_model(model_name, method, model_config, data)
    batch = Batch(
        user_features={"id": torch.ones(4, 1, dtype=torch.long)},
        item_features={
            "id": torch.arange(2, 6).reshape(4, 1),
            "text": torch.randn(4, 8),
            "image": torch.randn(4, 6),
        },
        history_features={
            "id": torch.arange(10, 30).reshape(4, 5),
            "text": torch.randn(4, 5, 8),
            "image": torch.randn(4, 5, 6),
        },
        history_mask=torch.ones(4, 5, dtype=torch.bool),
        labels=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    result = model(batch)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            result.logits.reshape(-1), batch.labels
        )
        + result.auxiliary_loss()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.structure_policy in {
        "legacy_forced_shared_wrapper",
        "legacy_preserve_similarity_tiers",
        "legacy_preserve_attention_or_cic",
        "legacy_mmmlp_mixers",
        "legacy_m3srec_shared_attention_and_experts",
        "legacy_mb_scores_and_balance",
        "legacy_pamd_pair_disentanglement",
        "legacy_marn_specific_invariant",
        "legacy_diff_msin_experts_and_two_level_gates",
    }
    assert not hasattr(model, "optimization_parameter_groups")
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in (model.user_linear if model_name == "naml" else model.output).parameters()
    )
    fusion = getattr(model, "study_fusion", getattr(model, "history_fusion", None))
    if fusion is None:
        fusion = model.target_fusion
    if model_name in {"mmmlp", "m3srec"}:
        fusion = model.history_fusion
    if model_name == "dmf" and method in ("maf", "lmf", "mtfn", "src", "fq-former"):
        # Legacy local DMF uses content only through discrete similarity buckets.
        assert all(p.grad is None for p in fusion.parameters())
    elif method != "cat":
        assert any(p.grad is not None for p in fusion.parameters())


@pytest.mark.parametrize("name", ["qarm", "psrq"])
@pytest.mark.parametrize("method", PAPER_FUSIONS)
def test_quantized_fusion_keeps_codebooks_and_handles_padding(name, method):
    from mmctr.analysis.fusion_quantized_backbones import LegacyQuantizedFusion
    from mmctr.models.mm_models.qarm import QARM
    from mmctr.models.mm_models.psrq import MCCA
    from mmctr.quantization import PSRQPretrainer
    from tests.unit.test_canonical_quantization import configs, make_rq, make_batch

    torch.manual_seed(2025)
    config, data = configs()
    config.update(
        latent_dim=16,
        projection_dim=16,
        simcen_hidden_units=[48, 48],
        dmf_mlp_dims=[16],
        T=2,
        attention_dim=16,
    )
    if name == "qarm":
        backbone = QARM(config, data, {"image": make_rq(3), "text": make_rq(2)})
    else:
        quantizer = PSRQPretrainer(config, data)
        with torch.no_grad():
            for module in quantizer.modules():
                if hasattr(module, "initted") and hasattr(module, "embedding"):
                    module.embedding.weight.normal_()
                    module.initted.fill_(True)
        backbone = MCCA(config, data, quantizer)
    model = LegacyQuantizedFusion(backbone, method, config)
    model.train()
    result = model(make_batch())
    loss = result.logits.square().mean() + result.auxiliary_loss()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if name == "psrq":
        assert not backbone.quantizer.training
        assert all(p.grad is None and not p.requires_grad for p in backbone.quantizer.parameters())
        assert not hasattr(backbone, "history_attention")
    else:
        assert hasattr(backbone, "cross")
    model.eval()
    with torch.no_grad():
        expected = model(make_batch()).logits
        changed = model(make_batch(padded_value=999)).logits
        assert torch.allclose(expected, changed, atol=1e-6, rtol=1e-5)
        assert torch.isfinite(model(make_batch(all_padding=True)).logits).all()
