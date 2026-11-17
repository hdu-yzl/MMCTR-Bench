"""Input layout and shard discovery for AntM2C TFRecord conversion."""

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from mmctr.core import ContractError


@dataclass(frozen=True)
class LegacyAntM2CLayout:
    """Shape and ID rules required to decode the frozen legacy TFRecord schema."""

    text_dimension: int = 768
    image_dimension: int = 512
    sequence_length: int = 5
    minimum_item_id: int = 67_625

    def __post_init__(self) -> None:
        for name in ("text_dimension", "image_dimension", "sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ContractError("{} must be a positive integer".format(name))
        if isinstance(self.minimum_item_id, bool) or int(self.minimum_item_id) <= 0:
            raise ContractError("minimum_item_id must be a positive integer")

    def to_dict(self) -> Dict[str, int]:
        return {
            "text_dimension": int(self.text_dimension),
            "image_dimension": int(self.image_dimension),
            "sequence_length": int(self.sequence_length),
            "minimum_item_id": int(self.minimum_item_id),
        }


def _natural_shard_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)\.tfrecord$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def discover_legacy_records(source_dir: Path) -> Mapping[str, Tuple[Path, ...]]:
    """Discover exactly the split files consumed by the frozen legacy training loader."""

    source_dir = Path(source_dir)
    train = tuple(sorted(source_dir.glob("train_shuffle_*.tfrecord"), key=_natural_shard_key))
    if not train:
        train = tuple(sorted(source_dir.glob("train[0-9]*.tfrecord"), key=_natural_shard_key))
    result = {
        "train": train,
        "val": tuple(sorted(source_dir.glob("val[0-9]*.tfrecord"), key=_natural_shard_key)),
        "test": tuple(sorted(source_dir.glob("test[0-9]*.tfrecord"), key=_natural_shard_key)),
    }
    missing = [name for name, paths in result.items() if not paths]
    if missing:
        raise ContractError("legacy AntM2C split TFRecords are missing: {}".format(missing))
    return MappingProxyType(result)
