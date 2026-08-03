"""Named AntM2C field ownership and split protocol."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Set, Tuple

from mmctr.core import ContractError


ANTM2C_SCHEMA_VERSION = 1


class FeatureOwner(str, Enum):
    INTERACTION_CONTEXT = "interaction_context"
    ITEM = "item"
    ITEM_CANDIDATE = "item_candidate"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    owner: FeatureOwner
    encoded_name: str
    encoded_dim: int
    share_across_splits: bool
    missing_policy: str

    def __post_init__(self) -> None:
        if not self.name or not self.encoded_name:
            raise ContractError("AntM2C field names cannot be empty")
        if self.encoded_dim <= 0:
            raise ContractError("encoded_dim must be positive")
        if self.missing_policy not in {"zero_with_manifest", "empty_text_with_manifest"}:
            raise ContractError("unknown missing policy: {!r}".format(self.missing_policy))


ANTM2C_FIELDS = MappingProxyType(
    {
        "service_entity_seq": FieldSpec(
            "service_entity_seq",
            FeatureOwner.INTERACTION_CONTEXT,
            "service_text",
            768,
            False,
            "empty_text_with_manifest",
        ),
        "query_entity_seq": FieldSpec(
            "query_entity_seq",
            FeatureOwner.INTERACTION_CONTEXT,
            "query_text",
            768,
            False,
            "empty_text_with_manifest",
        ),
        "bill_entity_seq": FieldSpec(
            "bill_entity_seq",
            FeatureOwner.INTERACTION_CONTEXT,
            "bill_text",
            768,
            False,
            "empty_text_with_manifest",
        ),
        "item_entity_names": FieldSpec(
            "item_entity_names",
            FeatureOwner.ITEM_CANDIDATE,
            "entity_text",
            768,
            True,
            "empty_text_with_manifest",
        ),
        "item_title": FieldSpec(
            "item_title",
            FeatureOwner.ITEM,
            "title_text",
            768,
            True,
            "empty_text_with_manifest",
        ),
        "log_time": FieldSpec(
            "log_time",
            FeatureOwner.INTERACTION_CONTEXT,
            "time_context",
            768,
            False,
            "empty_text_with_manifest",
        ),
        "image": FieldSpec(
            "image",
            FeatureOwner.ITEM,
            "image",
            512,
            True,
            "zero_with_manifest",
        ),
    }
)


@dataclass(frozen=True)
class OwnershipAudit:
    field: str
    items: int
    missing_values: int
    conflicting_items: Tuple[Any, ...]

    @property
    def can_promote_to_item(self) -> bool:
        return not self.conflicting_items


def audit_item_candidate(
    records: Iterable[Mapping[str, Any]],
    field: str = "item_entity_names",
) -> OwnershipAudit:
    """Find item IDs that have multiple non-empty values for a candidate item field."""

    if field not in ANTM2C_FIELDS:
        raise ContractError("unknown AntM2C field: {!r}".format(field))
    values_by_item: Dict[Any, Set[str]] = {}
    missing = 0
    for record in records:
        if "item_id" not in record:
            raise ContractError("ownership audit records require item_id")
        item_id = record["item_id"]
        value = record.get(field)
        if value is None or str(value).strip() == "":
            missing += 1
            continue
        values_by_item.setdefault(item_id, set()).add(str(value))
    conflicts = tuple(
        sorted(
            (item_id for item_id, values in values_by_item.items() if len(values) > 1),
            key=str,
        )
    )
    return OwnershipAudit(field, len(values_by_item), missing, conflicts)


@dataclass(frozen=True)
class AntM2CProtocol:
    """Versioned interaction identity, split, history, and ID rules."""

    data_version: str
    sequence_length: int = 5
    padding_id: int = 0
    item_index_start: int = 1
    train_end: datetime = datetime(2023, 8, 3, 0, 0, 0)
    validation_end: datetime = datetime(2023, 8, 5, 0, 0, 0)
    schema_version: int = ANTM2C_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.data_version:
            raise ContractError("AntM2C data_version is required")
        if self.sequence_length <= 0:
            raise ContractError("sequence_length must be positive")
        if self.item_index_start <= self.padding_id:
            raise ContractError("item indices must not overlap the padding ID")
        if self.train_end >= self.validation_end:
            raise ContractError("train_end must precede validation_end")

    def event_id(self, source_shard: str, source_row: int) -> str:
        shard = str(source_shard).strip()
        if not shard or "/" in shard or "\\" in shard:
            raise ContractError("source_shard must be a stable basename")
        if isinstance(source_row, bool) or source_row < 0:
            raise ContractError("source_row must be a non-negative integer")
        return "{}:{}:{}".format(self.data_version, shard, int(source_row))

    def split_for(self, timestamp: datetime) -> str:
        if not isinstance(timestamp, datetime):
            raise ContractError("timestamp must be a datetime")
        if timestamp <= self.train_end:
            return "train"
        if timestamp <= self.validation_end:
            return "val"
        return "test"

    @staticmethod
    def history_order_key(record: Mapping[str, Any]):
        try:
            return record["user_id"], record["timestamp"], record["event_id"]
        except KeyError as error:
            raise ContractError(
                "history records require user_id, timestamp, and event_id"
            ) from error


def field_spec(name: str) -> FieldSpec:
    try:
        return ANTM2C_FIELDS[name]
    except KeyError as error:
        raise ContractError("unknown AntM2C field: {!r}".format(name)) from error


__all__ = [
    "ANTM2C_FIELDS",
    "ANTM2C_SCHEMA_VERSION",
    "AntM2CProtocol",
    "FeatureOwner",
    "FieldSpec",
    "OwnershipAudit",
    "audit_item_candidate",
    "field_spec",
]
