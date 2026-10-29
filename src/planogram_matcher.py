"""Planogram Matcher using Bipartite Matching (Hungarian Algorithm).

Matches detected SKU bounding boxes against predefined shelf slots using a
spatial cost matrix that blends IoU overlap and center-point Euclidean distance.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(box_a: List[float], box_b: List[float]) -> float:
    """Computes normalized Euclidean distance between box centroids."""
    ca_x = (box_a[0] + box_a[2]) / 2.0
    ca_y = (box_a[1] + box_a[3]) / 2.0
    cb_x = (box_b[0] + box_b[2]) / 2.0
    cb_y = (box_b[1] + box_b[3]) / 2.0
    return float(np.sqrt((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2))


class PlanogramMatcher:
    """Performs optimal spatial assignment of detected SKUs to expected planogram slots."""

    def __init__(self, planogram: Dict[str, Any], min_iou_threshold: float = 0.25):
        self.planogram = planogram
        self.slots = planogram.get("slots", [])
        self.min_iou_threshold = min_iou_threshold

    def match(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Matches a list of detections to the planogram slots via Hungarian algorithm.

        Args:
            detections: List of detection dicts with 'box_norm', 'sku_name', 'confidence'.

        Returns:
            List of slot audit results with match status ('COMPLIANT', 'MISPLACED', 'OOS', 'ENCROACHED').
        """
        num_slots = len(self.slots)
        num_dets = len(detections)

        if num_slots == 0:
            return []

        # If no detections found at all -> all slots are OOS
        if num_dets == 0:
            return [
                {
                    "slot_id": slot["slot_id"],
                    "shelf_tier": slot["shelf_tier"],
                    "expected_sku": slot["expected_sku"],
                    "expected_brand": slot["brand"],
                    "status": "OOS",
                    "detected_sku": None,
                    "confidence": 0.0,
                    "iou": 0.0,
                    "box_norm": slot["box_norm"],
                }
                for slot in self.slots
            ]

        # Build Cost Matrix (Size: num_slots x num_dets)
        # Cost = 1.0 - IoU + 0.5 * CenterDistance
        cost_matrix = np.zeros((num_slots, num_dets), dtype=np.float32)

        for i, slot in enumerate(self.slots):
            slot_box = slot["box_norm"]
            for j, det in enumerate(detections):
                det_box = det["box_norm"]
                iou = compute_iou(slot_box, det_box)
                dist = compute_center_distance(slot_box, det_box)
                cost = (1.0 - iou) + (0.5 * dist)
                cost_matrix[i, j] = cost

        # Solve optimal bipartite matching
        slot_indices, det_indices = linear_sum_assignment(cost_matrix)

        assigned_slots = set()
        assigned_dets = set()
        matched_results = []

        for slot_idx, det_idx in zip(slot_indices, det_indices):
            slot = self.slots[slot_idx]
            det = detections[det_idx]
            iou = compute_iou(slot["box_norm"], det["box_norm"])

            # Valid spatial match candidate
            if iou >= self.min_iou_threshold:
                assigned_slots.add(slot_idx)
                assigned_dets.add(det_idx)

                expected_sku = slot["expected_sku"].strip().lower()
                detected_sku = det["sku_name"].strip().lower()

                if "encroaching" in detected_sku or "competitor" in detected_sku:
                    status = "ENCROACHED"
                elif expected_sku == detected_sku:
                    status = "COMPLIANT"
                else:
                    status = "MISPLACED"

                matched_results.append({
                    "slot_id": slot["slot_id"],
                    "shelf_tier": slot["shelf_tier"],
                    "expected_sku": slot["expected_sku"],
                    "expected_brand": slot["brand"],
                    "status": status,
                    "detected_sku": det["sku_name"],
                    "confidence": det.get("confidence", 0.95),
                    "iou": round(float(iou), 3),
                    "box_norm": det["box_norm"],
                })

        # Process unassigned slots (Empty / Out-Of-Stock)
        for i, slot in enumerate(self.slots):
            if i not in assigned_slots:
                matched_results.append({
                    "slot_id": slot["slot_id"],
                    "shelf_tier": slot["shelf_tier"],
                    "expected_sku": slot["expected_sku"],
                    "expected_brand": slot["brand"],
                    "status": "OOS",
                    "detected_sku": None,
                    "confidence": 0.0,
                    "iou": 0.0,
                    "box_norm": slot["box_norm"],
                })

        # Sort results sequentially by shelf tier and slot ID
        matched_results.sort(key=lambda x: (x["shelf_tier"], x["slot_id"]))
        return matched_results
