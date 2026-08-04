import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mmctr.data.datasets.antm2c.array_store import (
    AntM2CArrayLoader,
    InteractionTable,
    load_array_store,
    write_array_store,
)
from mmctr.data.datasets.antm2c.item_store import build_feature_store, build_item_index
from mmctr.data.manifest import DatasetManifest
from mmctr.models.registry import create_model


class AntM2CArrayStoreTests(unittest.TestCase):
    def test_named_array_round_trip_and_item_gather(self):
        index = build_item_index(["item-a", "item-b"])
        item_store, _ = build_feature_store(
            index,
            {
                "image": {
                    "item-a": np.asarray([1.0, 2.0]),
                    "item-b": np.asarray([3.0, 4.0]),
                }
            },
            {"image": 2},
        )
        table = InteractionTable(
            event_ids=("v1:part:0", "v1:part:1"),
            user_indices=np.asarray([1, 2]),
            item_indices=np.asarray([1, 2]),
            history_item_indices=np.asarray([[0, 0], [1, 0]]),
            labels=np.asarray([1.0, 0.0]),
            context_features={"query_text": np.ones((2, 3), dtype=np.float32)},
        )
        manifest = DatasetManifest(
            name="antm2c",
            version="fixture-v1",
            storage_format="named-npy-candidate-v1",
            sequence_length=2,
            padding_id=0,
            id_offsets={"user": 0, "item": 10},
            feature_dimensions={"image": 2, "query_text": 3},
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "store"
            write_array_store(target, manifest, {"train": table}, item_store)
            loaded_manifest, splits, loaded_items = load_array_store(target)
            loader = AntM2CArrayLoader(loaded_manifest, splits, loaded_items, batch_size=2)
            batch = next(iter(loader.iter_batches("train")))
            self.assertEqual((2, 2), tuple(batch.item_features["image"].shape))
            self.assertEqual((2, 2, 2), tuple(batch.history_features["image"].shape))
            self.assertEqual((2, 3), tuple(batch.context_features["query_text"].shape))
            self.assertEqual([11, 12], batch.item_features["id"].squeeze(1).tolist())
            self.assertEqual([[0, 0], [11, 0]], batch.history_features["id"].tolist())

    def test_named_context_composes_legacy_user_view_for_canonical_model(self):
        index = build_item_index(["item-a", "item-b"])
        item_store, _ = build_feature_store(
            index,
            {
                "image": {
                    "item-a": np.asarray([1.0, 2.0]),
                    "item-b": np.asarray([3.0, 4.0]),
                }
            },
            {"image": 2},
        )
        context_names = (
            "service_text",
            "query_text",
            "bill_text",
            "entity_text",
            "time_context",
        )
        contexts = {
            name: np.full((2, 2), index_value, dtype=np.float32)
            for index_value, name in enumerate(context_names, start=1)
        }
        table = InteractionTable(
            event_ids=("v1:part:0", "v1:part:1"),
            user_indices=np.asarray([1, 2]),
            item_indices=np.asarray([1, 2]),
            history_item_indices=np.asarray([[0, 0], [1, 0]]),
            labels=np.asarray([1.0, 0.0]),
            context_features=contexts,
        )
        manifest = DatasetManifest(
            name="antm2c",
            version="fixture-v1",
            storage_format="named-npy-candidate-v1",
            sequence_length=2,
            padding_id=0,
            id_offsets={"user": 0, "item": 10},
            feature_dimensions={"image": 2, **{name: 2 for name in context_names}},
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "store"
            write_array_store(target, manifest, {"train": table}, item_store)
            batch = next(
                AntM2CArrayLoader.from_directory(target, batch_size=2).iter_batches("train")
            )

        expected_user_text = torch.cat(
            [batch.context_features[name] for name in context_names], dim=-1
        )
        self.assertTrue(torch.equal(expected_user_text, batch.context_features["text"]))
        model = create_model(
            "dnn_mm_seq",
            {
                "latent_dim": 4,
                "projection_dim": 4,
                "mlp_dims": [8],
                "dropout": 0.0,
                "batch_norm": False,
            },
            {
                "id_feature_num": 32,
                "use_mm_features": ["id", "image"],
                "mm_seq_dims": {"image": 2},
                "user_features": ["id", "text"],
                "user_features_dim": {"text": 10},
            },
        )
        output = model(batch)
        self.assertEqual((2,), tuple(output.logits.shape))
        output.logits.sum().backward()


if __name__ == "__main__":
    unittest.main()
