import os
import json
import pytest
from src.planogram_matcher import PlanogramMatcher, compute_iou, compute_center_distance
from src.compliance_engine import ComplianceEngine

def test_compute_iou():
    box1 = [0.0, 0.0, 1.0, 1.0]
    box2 = [0.0, 0.0, 1.0, 1.0]
    assert compute_iou(box1, box2) == 1.0

    box3 = [2.0, 2.0, 3.0, 3.0]
    assert compute_iou(box1, box3) == 0.0

def test_hungarian_matching():
    sample_planogram = {
        "shelf_id": "TEST-SHELF",
        "category": "Test",
        "slots": [
            {
                "slot_id": "T1-P01",
                "shelf_tier": 1,
                "expected_sku": "Amul Butter 500g",
                "brand": "Amul",
                "box_norm": [0.1, 0.1, 0.3, 0.4]
            },
            {
                "slot_id": "T1-P02",
                "shelf_tier": 1,
                "expected_sku": "Mother Dairy Butter 500g",
                "brand": "Mother Dairy",
                "box_norm": [0.4, 0.1, 0.6, 0.4]
            }
        ]
    }

    matcher = PlanogramMatcher(sample_planogram)

    # 1 detected item matching T1-P01, T1-P02 is empty
    detections = [
        {
            "sku_name": "Amul Butter 500g",
            "confidence": 0.95,
            "box_norm": [0.11, 0.10, 0.31, 0.40]
        }
    ]

    matched = matcher.match(detections)
    assert len(matched) == 2
    
    # Slot 1 should be compliant
    slot1 = next(s for s in matched if s["slot_id"] == "T1-P01")
    assert slot1["status"] == "COMPLIANT"

    # Slot 2 should be OOS
    slot2 = next(s for s in matched if s["slot_id"] == "T1-P02")
    assert slot2["status"] == "OOS"

    # Compliance report
    report = ComplianceEngine.audit_report("TEST-SHELF", "Test", matched)
    assert report["compliance_score"] == 50.0
    assert report["oos_facings"] == 1
    assert report["compliant_facings"] == 1
