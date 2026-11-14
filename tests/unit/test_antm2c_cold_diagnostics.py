import json

import numpy as np

from mmctr.analysis.cold_populations import build_cold_populations


def test_shard_order_user_zero_and_independent_item_counts(tmp_path):
    layout = {"storage_format": "sharded-named-npy-v1", "splits": {}}
    arrays = {
        "train": [([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]), ([2], [3], [1])],
        "val": [([9, 0], [1, 8], [0, 1]), ([2, 9], [3, 8], [1, 0])],
        "test": [([9, 0], [1, 8], [0, 1]), ([2, 9], [3, 8], [1, 0])],
    }
    for split, shards in arrays.items():
        layout["splits"][split] = []
        for index, (users, items, labels) in enumerate(shards):
            path = split + "/" + str(10 - index)
            directory = tmp_path / path
            directory.mkdir(parents=True)
            layout["splits"][split].append({"path": path})
            for name, value in zip(("user_index", "item_index", "labels"), (users, items, labels)):
                np.save(directory / (name + ".npy"), np.array(value))
            np.save(directory / "history_item_index.npy", np.zeros((len(users), 2), dtype=np.int64))
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "antm2c"}))
    user, _ = build_cold_populations(tmp_path, "user", "zero")
    item, _ = build_cold_populations(tmp_path, "item", "zero")
    few, _ = build_cold_populations(tmp_path, "user", "low_lt5")
    assert user["val"].tolist() == [True, False, False, True]
    assert item["val"].tolist() == [False, True, False, True]
    assert few["val"].tolist() == [True, False, True, True]
