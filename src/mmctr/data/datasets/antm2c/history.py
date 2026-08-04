"""Leakage-safe single-scan AntM2C history construction."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Deque, Dict, Iterable, List, Mapping, Tuple

from mmctr.core import ContractError


@dataclass(frozen=True)
class Interaction:
    """One versioned event whose identity disambiguates repeated item interactions."""

    event_id: str
    user_id: Any
    item_index: int
    timestamp: datetime
    label: float
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ContractError("event_id is required")
        if not isinstance(self.timestamp, datetime):
            raise ContractError("timestamp must be a datetime")
        if self.item_index <= 0:
            raise ContractError("item_index must be greater than the padding ID")
        if float(self.label) not in {0.0, 1.0}:
            raise ContractError("interaction label must be binary")
        if self.split not in {"train", "val", "test"}:
            raise ContractError("interaction split must be train, val, or test")
        object.__setattr__(self, "label", float(self.label))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class InteractionHistory:
    interaction: Interaction
    history_item_indices: Tuple[int, ...]
    history_length: int


def build_histories(
    interactions: Iterable[Interaction],
    sequence_length: int,
    padding_id: int = 0,
    positive_label: float = 1.0,
) -> List[InteractionHistory]:
    """Sort once, then scan once to attach strictly earlier positive events."""

    if sequence_length <= 0:
        raise ContractError("sequence_length must be positive")
    records = list(interactions)
    event_ids = [record.event_id for record in records]
    if len(event_ids) != len(set(event_ids)):
        raise ContractError("event_id values must be unique")
    # The stable event ID is the final tie-breaker for equal-timestamp repeats.
    records.sort(key=lambda record: (str(record.user_id), record.timestamp, record.event_id))

    histories: Dict[Any, Deque[int]] = {}
    output: List[InteractionHistory] = []
    for record in records:
        history = histories.setdefault(record.user_id, deque(maxlen=sequence_length))
        values = tuple(history)
        padding = (padding_id,) * (sequence_length - len(values))
        output.append(
            InteractionHistory(
                interaction=record,
                history_item_indices=padding + values,
                history_length=len(values),
            )
        )
        if record.label == float(positive_label):
            history.append(record.item_index)
    return output


def assert_no_future_leakage(records: Iterable[InteractionHistory]) -> None:
    """Validate lengths/padding and monotonic per-user event order."""

    last_key_by_user: Dict[Any, Tuple[datetime, str]] = {}
    for record in records:
        interaction = record.interaction
        current_key = interaction.timestamp, interaction.event_id
        previous_key = last_key_by_user.get(interaction.user_id)
        if previous_key is not None and current_key < previous_key:
            raise ContractError("history records are not in stable temporal order")
        last_key_by_user[interaction.user_id] = current_key
        if record.history_length < 0 or record.history_length > len(record.history_item_indices):
            raise ContractError("invalid history_length")
        padding_count = len(record.history_item_indices) - record.history_length
        if any(value != 0 for value in record.history_item_indices[:padding_count]):
            raise ContractError("history must use left padding")


__all__ = ["Interaction", "InteractionHistory", "assert_no_future_leakage", "build_histories"]
