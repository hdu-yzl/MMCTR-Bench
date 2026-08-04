"""Strict, deterministic replay of bounded AntM2C raw-event inputs."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from mmctr.core import ContractError

from .history import Interaction, InteractionHistory, assert_no_future_leakage, build_histories
from .item_store import ItemIndex, build_item_index


RAW_REQUIRED_FIELDS = (
    "user_id",
    "item_id",
    "log_time",
    "label",
    "bill_entity_seq",
    "service_entity_seq",
    "query_entity_seq",
    "item_entity_names",
    "item_title",
    "scene",
)
TRAIN_CUTOFF = datetime(2023, 8, 3)
VALIDATION_CUTOFF = datetime(2023, 8, 5)


@dataclass(frozen=True)
class RawEvent:
    event_id: str
    user_id: str
    original_item_id: str
    timestamp: datetime
    label: float
    split: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


@dataclass(frozen=True)
class RawReplay:
    """Auditable raw-to-index-to-causal-history replay result."""

    events: Tuple[Interaction, ...]
    histories: Tuple[InteractionHistory, ...]
    item_index: ItemIndex


def split_for_timestamp(timestamp: datetime) -> str:
    """Apply the tracked legacy midnight cutoffs without silent date coercion."""

    if timestamp <= TRAIN_CUTOFF:
        return "train"
    if timestamp <= VALIDATION_CUTOFF:
        return "val"
    return "test"


def iter_raw_events(
    event_parts: Sequence[Path], limit_per_part: Optional[int] = None
) -> Iterator[RawEvent]:
    """Scan each CSV once, deriving stable IDs from source filename and physical row."""

    if not event_parts:
        raise ContractError("at least one AntM2C raw event part is required")
    if limit_per_part is not None and limit_per_part <= 0:
        raise ContractError("limit_per_part must be positive")
    seen_names = set()
    for path_value in event_parts:
        path = Path(path_value)
        if not path.is_file():
            raise ContractError("AntM2C raw event part is missing: {}".format(path))
        if path.name in seen_names:
            raise ContractError("AntM2C raw event part names must be unique")
        seen_names.add(path.name)
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = [
                name for name in RAW_REQUIRED_FIELDS if name not in (reader.fieldnames or ())
            ]
            if missing:
                raise ContractError("AntM2C raw header is missing fields: {}".format(missing))
            for source_row, values in enumerate(reader):
                if limit_per_part is not None and source_row >= limit_per_part:
                    break
                try:
                    timestamp = datetime.strptime(values["log_time"], "%Y-%m-%d %H:%M:%S")
                    label = float(values["label"])
                except (TypeError, ValueError) as error:
                    raise ContractError(
                        "invalid AntM2C timestamp/label at {} row {}".format(path, source_row)
                    ) from error
                if label not in {0.0, 1.0}:
                    raise ContractError("AntM2C raw labels must be binary")
                user_id = values["user_id"]
                original_item_id = values["item_id"]
                if not user_id or not original_item_id:
                    raise ContractError("AntM2C raw user/item IDs cannot be empty")
                yield RawEvent(
                    event_id="{}:{:012d}".format(path.name, source_row),
                    user_id=user_id,
                    original_item_id=original_item_id,
                    timestamp=timestamp,
                    label=label,
                    split=split_for_timestamp(timestamp),
                    fields={name: values.get(name) for name in RAW_REQUIRED_FIELDS[4:]},
                )


def replay_raw_events(
    event_parts: Sequence[Path],
    sequence_length: int = 5,
    limit_per_part: Optional[int] = None,
) -> RawReplay:
    """Replay a bounded real-input slice through item indexing and causal histories."""

    raw_events = tuple(iter_raw_events(event_parts, limit_per_part=limit_per_part))
    if not raw_events:
        raise ContractError("AntM2C raw event parts contain no events")
    item_index = build_item_index(event.original_item_id for event in raw_events)
    interactions = [
        Interaction(
            event_id=event.event_id,
            user_id=event.user_id,
            item_index=item_index.encode(event.original_item_id),
            timestamp=event.timestamp,
            label=event.label,
            split=event.split,
            metadata={"original_item_id": event.original_item_id, **dict(event.fields)},
        )
        for event in raw_events
    ]
    histories = tuple(build_histories(interactions, sequence_length=sequence_length))
    assert_no_future_leakage(histories)
    return RawReplay(
        events=tuple(record.interaction for record in histories),
        histories=histories,
        item_index=item_index,
    )


__all__ = [
    "RAW_REQUIRED_FIELDS",
    "RawEvent",
    "RawReplay",
    "iter_raw_events",
    "replay_raw_events",
    "split_for_timestamp",
]
