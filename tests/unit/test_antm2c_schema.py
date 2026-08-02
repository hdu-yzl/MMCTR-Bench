import unittest
from datetime import datetime

from mmctr.data.datasets.antm2c import (
    ANTM2C_FIELDS,
    AntM2CProtocol,
    FeatureOwner,
    audit_item_candidate,
)


class AntM2CSchemaTests(unittest.TestCase):
    def test_field_ownership_is_named_and_not_positional(self):
        self.assertEqual(
            FeatureOwner.INTERACTION_CONTEXT,
            ANTM2C_FIELDS["service_entity_seq"].owner,
        )
        self.assertEqual(FeatureOwner.ITEM, ANTM2C_FIELDS["item_title"].owner)
        self.assertEqual(
            FeatureOwner.ITEM_CANDIDATE,
            ANTM2C_FIELDS["item_entity_names"].owner,
        )

    def test_candidate_audit_detects_item_conflicts(self):
        audit = audit_item_candidate(
            [
                {"item_id": 1, "item_entity_names": "alpha"},
                {"item_id": 1, "item_entity_names": "beta"},
                {"item_id": 2, "item_entity_names": "same"},
                {"item_id": 2, "item_entity_names": "same"},
            ]
        )
        self.assertFalse(audit.can_promote_to_item)
        self.assertEqual((1,), audit.conflicting_items)

    def test_event_identity_and_split_boundaries_are_stable(self):
        protocol = AntM2CProtocol("fixture-v1")
        self.assertEqual("fixture-v1:part0:42", protocol.event_id("part0", 42))
        self.assertEqual("train", protocol.split_for(datetime(2023, 8, 3)))
        self.assertEqual("val", protocol.split_for(datetime(2023, 8, 4)))
        self.assertEqual("test", protocol.split_for(datetime(2023, 8, 6)))


if __name__ == "__main__":
    unittest.main()
