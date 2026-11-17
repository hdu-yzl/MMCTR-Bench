import pytest

from mmctr.analysis.diagnostic_execution import main
from mmctr.core import ContractError


@pytest.mark.parametrize(
    "family,dataset,model,setting,error",
    [
        (
            "robustness",
            "antm2c",
            "dmf",
            ["--probability", "0.3"],
            "only cold-start and native control",
        ),
        ("fusion", "microlens", "pamd", ["--method", "SRC"], "requires at least two inputs"),
    ],
)
def test_unsupported_setting_rejected_before_loading_candidates(
    tmp_path, family, dataset, model, setting, error
):
    with pytest.raises(ContractError, match=error):
        main(
            [
                family,
                "--dataset-name",
                dataset,
                "--model-name",
                model,
                "--candidates",
                str(tmp_path / "missing.yaml"),
                "--output-root",
                str(tmp_path / "analysis"),
                "--study-id",
                "unsupported",
            ]
            + setting
        )
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize(
    "family,setting",
    [
        ("alignment", ["--method", "cosine", "--weight", "0.1"]),
        ("fusion", ["--method", "cat"]),
    ],
)
def test_dmf_projected_studies_require_explicit_variant(tmp_path, family, setting):
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("candidates:\n  - training: {lr: 0.001}\n")
    with pytest.raises(ContractError, match="similarity_source: legacy_projected"):
        main(
            [
                family,
                "--dataset-name",
                "tiktok",
                "--model-name",
                "dmf",
                "--candidates",
                str(candidates),
                "--output-root",
                str(tmp_path / "analysis"),
                "--study-id",
                "dmf-projected",
            ]
            + setting
        )
    assert not (tmp_path / "analysis").exists()
