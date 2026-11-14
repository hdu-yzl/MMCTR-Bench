import numpy as np
import torch

from mmctr.analysis.cold_populations import build_cold_populations


def write_arrays(root):
    for split, users, items in (
        ("train", [1, 1, 1, 1, 1, 2], [11, 11, 11, 11, 11, 12]),
        ("val", [1, 3, 1, 4], [13, 11, 14, 14]),
        ("test", [1, 3, 1, 4], [13, 11, 14, 14]),
    ):
        path = root / "splits" / split
        path.mkdir(parents=True)
        for field, value in {
            "user_ids": users,
            "item_ids": items,
            "labels": [0, 1, 0, 1][: len(users)] if split != "train" else [0, 0, 0, 0, 1, 1],
            "history_item_ids": [[13, 0]] * len(users),
        }.items():
            np.save(path / (field + ".npy"), np.array(value, dtype=np.int64))


def test_zero_shot_excludes_items_seen_in_train_history(tmp_path):
    write_arrays(tmp_path)
    masks, _ = build_cold_populations(tmp_path, "union", "zero")
    assert masks["test"].tolist() == [False, True, True, True]
    few, _ = build_cold_populations(tmp_path, "union", "low_lt5")
    assert few["test"].tolist() == [True, True, True, True]


def test_paper_loader_keeps_full_validation_and_maps_shared_oov_after_selection(tmp_path):
    from mmctr.analysis.benchmark_protocols import PaperColdStartLoader, reserve_oov_embeddings
    from mmctr.analysis.cold_populations import training_exposure
    from mmctr.models.common.registry import create_model
    from tests.unit.test_canonical_complex_sequence import configs, make_batch

    write_arrays(tmp_path)
    _, seen, offset = training_exposure(tmp_path)
    batch = make_batch()
    # User 1 seen, user 3 unseen; target 13 seen via history, 14 unseen.
    from dataclasses import replace

    batch = replace(
        batch,
        user_features={"id": torch.tensor([[1], [3]])},
        item_features={**batch.item_features, "id": torch.tensor([[13], [14]])},
        history_features={**batch.history_features, "id": torch.tensor([[13, 0], [14, 11]])},
    )

    class Source:
        def iter_batches(self, split):
            yield batch

    model_config, data_config = configs()
    model = create_model("dmf", model_config, data_config)
    original_size = model.embedding.embedding.shape[0]
    oov = reserve_oov_embeddings(model)
    assert oov == (original_size, original_size + 1)
    loader = PaperColdStartLoader(Source(), np.array([True, True]), seen, offset, oov)
    assert next(loader.iter_batches("train")) is batch
    assert next(loader.iter_batches("val")) is batch
    result = next(loader.iter_batches("test"))
    assert result.user_features["id"].tolist() == [[1], [oov[0]]]
    assert result.item_features["id"].tolist() == [[13], [oov[1]]]
    assert result.history_features["id"].tolist() == [[13, 0], [oov[1], 11]]
    assert torch.equal(result.item_features["text"], batch.item_features["text"])
    assert torch.equal(result.history_mask, batch.history_mask)
    assert torch.isfinite(model(result).logits).all()
    assert batch.item_features["id"].tolist() == [[13], [14]]
