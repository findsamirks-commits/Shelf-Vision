"""Data preparation and synthetic dark-store shelf image generator.

Generates realistic retail shelf audit images with brand packaging colors,
text labeling, deliberate OOS empty slots, and encroaching items for testing.
"""

import json
import os
import random
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont


BRAND_PALETTE = {
    "Amul": (220, 38, 38),              # Vibrant Red
    "Mother Dairy": (37, 99, 235),       # Deep Blue
    "Nestle": (217, 119, 6),            # Golden Amber
    "Britannia": (16, 185, 129),        # Emerald Green
    "Milkbasket Private Label": (147, 51, 234), # Purple
    "Competitor Encroaching SKU": (75, 85, 99), # Slate Grey
}


def load_planogram(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_shelf_image(
    planogram: dict,
    output_image_path: str,
    oos_slot_ids: List[str] = None,
    encroached_slot_ids: Dict[str, str] = None,
    misplaced_slot_ids: Dict[str, str] = None,
) -> Tuple[str, List[dict]]:
    """Draws a synthetic dark store shelf photo with designated OOS or misplaced SKUs.

    Returns:
        output_image_path: Path where the generated image is saved.
        ground_truth_detections: List of bounding box dicts actually rendered.
    """
    oos_slot_ids = oos_slot_ids or []
    encroached_slot_ids = encroached_slot_ids or {}
    misplaced_slot_ids = misplaced_slot_ids or {}

    width = planogram["shelf_dimensions"]["width_px"]
    height = planogram["shelf_dimensions"]["height_px"]

    # Shelf background: industrial dark store shelving (cool dark metallic)
    img = Image.new("RGB", (width, height), color=(30, 41, 59))
    draw = ImageDraw.Draw(img)

    # Draw shelf rails for 3 tiers
    for tier_y in [0.08, 0.38, 0.68, 0.98]:
        y_px = int(tier_y * height)
        draw.rectangle([0, y_px - 8, width, y_px + 8], fill=(71, 85, 105))
        draw.line([0, y_px, width, y_px], fill=(148, 163, 184), width=2)

    rendered_detections = []

    for slot in planogram["slots"]:
        slot_id = slot["slot_id"]
        box_norm = slot["box_norm"]
        x1 = int(box_norm[0] * width)
        y1 = int(box_norm[1] * height)
        x2 = int(box_norm[2] * width)
        y2 = int(box_norm[3] * height)

        # Scenario 1: Slot is Out of Stock (OOS) - leave empty shelf space
        if slot_id in oos_slot_ids:
            # Draw empty slot boundary indicator (dark faint dashed style)
            draw.rectangle([x1 + 4, y1 + 4, x2 - 4, y2 - 4], fill=(15, 23, 42), outline=(51, 65, 85), width=2)
            draw.text((x1 + 10, y1 + (y2 - y1) // 2 - 10), "EMPTY / OOS", fill=(100, 116, 139))
            continue

        # Scenario 2: Competitor Encroachment
        if slot_id in encroached_slot_ids:
            sku_name = encroached_slot_ids[slot_id]
            brand = "Competitor Encroaching SKU"
            fill_color = BRAND_PALETTE[brand]
        # Scenario 3: Misplaced Product
        elif slot_id in misplaced_slot_ids:
            sku_name = misplaced_slot_ids[slot_id]
            brand = "Amul" if "Amul" in sku_name else "Nestle"
            fill_color = BRAND_PALETTE.get(brand, (100, 100, 100))
        # Scenario 4: Compliant Expected Product
        else:
            sku_name = slot["expected_sku"]
            brand = slot["brand"]
            fill_color = BRAND_PALETTE.get(brand, (99, 102, 241))

        # Add slight natural jitter to coordinates (real store camera angle)
        jitter_x = random.randint(-4, 4)
        jitter_y = random.randint(-4, 4)
        bx1, by1 = max(0, x1 + jitter_x), max(0, y1 + jitter_y)
        bx2, by2 = min(width, x2 + jitter_x), min(height, y2 + jitter_y)

        # Render SKU packaging block
        draw.rectangle([bx1, by1, bx2, by2], fill=fill_color, outline=(255, 255, 255), width=2)

        # Inner packaging label decoration
        label_y1 = by1 + int((by2 - by1) * 0.25)
        label_y2 = by2 - int((by2 - by1) * 0.25)
        draw.rectangle([bx1 + 4, label_y1, bx2 - 4, label_y2], fill=(248, 250, 252))

        # Text label (Brand & SKU)
        text_preview = sku_name.split()[0] + " " + sku_name.split()[1] if len(sku_name.split()) > 1 else sku_name
        draw.text((bx1 + 8, label_y1 + 4), brand[:12], fill=(15, 23, 42))
        draw.text((bx1 + 8, label_y1 + 20), text_preview[:15], fill=(30, 41, 59))

        rendered_detections.append({
            "sku_name": sku_name,
            "brand": brand,
            "box_norm": [round(bx1 / width, 4), round(by1 / height, 4), round(bx2 / width, 4), round(by2 / height, 4)],
            "box_px": [bx1, by1, bx2, by2],
            "confidence": round(random.uniform(0.88, 0.98), 3),
        })

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    img.save(output_image_path, quality=95)
    return output_image_path, rendered_detections


if __name__ == "__main__":
    planogram_file = os.path.join(os.path.dirname(__file__), "..", "configs", "planogram_sample.json")
    demo_image = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_shelf.jpg")

    plano = load_planogram(planogram_file)
    # Generate test image with 2 OOS slots and 1 misplaced item
    img_path, dets = generate_synthetic_shelf_image(
        planogram=plano,
        output_image_path=demo_image,
        oos_slot_ids=["S1-P03", "S2-P04"], # Mother Dairy Butter OOS, Nestle Milkmaid OOS
        misplaced_slot_ids={"S2-P02": "Nestle Everyday Dairy Whitener 1kg"},
        encroached_slot_ids={"S3-P04": "Competitor Rogue Ghee 1L"},
    )
    print(f"Generated synthetic dark-store shelf image: {img_path}")
    print(f"Rendered {len(dets)} detected facings across shelf grid.")
