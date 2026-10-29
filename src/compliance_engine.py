"""Compliance Engine: Computes executive retail shelf metrics and audits.

Calculates:
1. Overall Planogram Compliance Score (%)
2. Out-of-Stock (OOS) Rate & Affected Facings
3. Misplaced & Rogue Facings
4. Brand Share-of-Shelf (SoS) Distribution
"""

from typing import Any, Dict, List
import pandas as pd


class ComplianceEngine:
    """Processes matched slot audit records into aggregated business KPIs."""

    @staticmethod
    def audit_report(
        shelf_id: str,
        category: str,
        matched_slots: List[Dict[str, Any]],
        inference_ms: float = 85.0
    ) -> Dict[str, Any]:
        total_slots = len(matched_slots)
        if total_slots == 0:
            return {
                "shelf_id": shelf_id,
                "category": category,
                "compliance_score": 0.0,
                "total_slots": 0,
                "summary": "No slots found on planogram.",
            }

        compliant_count = sum(1 for s in matched_slots if s["status"] == "COMPLIANT")
        oos_count = sum(1 for s in matched_slots if s["status"] == "OOS")
        misplaced_count = sum(1 for s in matched_slots if s["status"] == "MISPLACED")
        encroached_count = sum(1 for s in matched_slots if s["status"] == "ENCROACHED")

        compliance_score = round((compliant_count / total_slots) * 100.0, 1)

        # OOS Slots breakdown
        oos_slots = [
            {
                "slot_id": s["slot_id"],
                "expected_sku": s["expected_sku"],
                "expected_brand": s["expected_brand"],
                "shelf_tier": s["shelf_tier"],
            }
            for s in matched_slots
            if s["status"] == "OOS"
        ]

        # Misplaced SKUs breakdown
        misplaced_slots = [
            {
                "slot_id": s["slot_id"],
                "expected_sku": s["expected_sku"],
                "detected_sku": s["detected_sku"],
                "shelf_tier": s["shelf_tier"],
            }
            for s in matched_slots
            if s["status"] == "MISPLACED"
        ]

        # Encroached SKUs breakdown
        encroached_slots = [
            {
                "slot_id": s["slot_id"],
                "expected_sku": s["expected_sku"],
                "detected_sku": s["detected_sku"],
                "shelf_tier": s["shelf_tier"],
            }
            for s in matched_slots
            if s["status"] == "ENCROACHED"
        ]

        # Brand Share of Shelf (calculated on actual detected on-shelf items)
        detected_brands = []
        for s in matched_slots:
            if s["status"] in ["COMPLIANT", "MISPLACED"]:
                # determine brand
                detected_sku = s["detected_sku"] or ""
                brand = s["expected_brand"]
                if "Amul" in detected_sku:
                    brand = "Amul"
                elif "Mother Dairy" in detected_sku:
                    brand = "Mother Dairy"
                elif "Nestle" in detected_sku:
                    brand = "Nestle"
                elif "Britannia" in detected_sku:
                    brand = "Britannia"
                elif "Milkbasket" in detected_sku:
                    brand = "Milkbasket Private Label"
                detected_brands.append(brand)
            elif s["status"] == "ENCROACHED":
                detected_brands.append("Competitor Rogue")

        brand_share = {}
        if detected_brands:
            brand_series = pd.Series(detected_brands).value_counts(normalize=True) * 100.0
            brand_share = {brand: round(float(pct), 1) for brand, pct in brand_series.items()}

        return {
            "shelf_id": shelf_id,
            "category": category,
            "compliance_score": compliance_score,
            "total_slots": total_slots,
            "compliant_facings": compliant_count,
            "oos_facings": oos_count,
            "misplaced_facings": misplaced_count,
            "encroached_facings": encroached_count,
            "oos_rate_pct": round((oos_count / total_slots) * 100.0, 1),
            "share_of_shelf_pct": brand_share,
            "oos_slots": oos_slots,
            "misplaced_slots": misplaced_slots,
            "encroached_slots": encroached_slots,
            "inference_latency_ms": round(inference_ms, 1),
            "matched_slots": matched_slots,
        }
