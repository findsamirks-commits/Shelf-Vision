"""Data preparation and synthetic dark-store shelf image generator.

Generates realistic retail shelf audit images with brand packaging colors,
text labeling, deliberate OOS empty slots, and encroaching items for testing.
Also generates full YOLOv8 formatted datasets with images/ and labels/ splits.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


BRAND_PALETTE = {
    "Amul": (220, 38, 38),              # Vibrant Red
    "Mother Dairy": (37, 99, 235),       # Deep Blue
    "Nestle": (217, 119, 6),            # Golden Amber
    "Britannia": (16, 185, 129),        # Emerald Green
    "Milkbasket Private Label": (147, 51, 234), # Purple
    "Competitor Encroaching SKU": (75, 85, 99), # Slate Grey
}

CLASS_TO_ID = {
    "Amul Butter 500g": 0,
    "Mother Dairy Butter 500g": 1,
    "Milkbasket Fresh Paneer 200g": 2,
    "Amul Cheese Slices 200g": 3,
    "Britannia Cheese Block 200g": 4,
    "Nestle Milkmaid 400g": 5,
    "Milkbasket Ghee 500ml": 6,
    "Nestle Everyday Dairy Whitener 1kg": 7,
    "Amul Spray Milk Powder 500g": 8,
    "Mother Dairy Cow Ghee 1L": 9,
    "Milkbasket Organic Cow Ghee 1L": 10,
    "Competitor Encroaching SKU": 11,
}


def load_planogram(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_shelf_image(
    planogram: dict,
    output_image_path: str,
    oos_slot_ids: Optional[List[str]] = None,
    encroached_slot_ids: Optional[Dict[str, str]] = None,
    misplaced_slot_ids: Optional[Dict[str, str]] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
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
    bg = bg_color or (random.randint(25, 38), random.randint(35, 48), random.randint(50, 65))
    img = Image.new("RGB", (width, height), color=bg)
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
            sku_name = "Competitor Encroaching SKU"
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


def generate_yolo_dataset(
    planogram_path: str,
    output_dir: str = "data",
    n_train: int = 60,
    n_val: int = 15,
    seed: int = 42,
) -> Dict[str, int]:
    """Generates complete YOLOv8 formatted training and validation splits."""
    random.seed(seed)
    planogram = load_planogram(planogram_path)
    width = planogram["shelf_dimensions"]["width_px"]
    height = planogram["shelf_dimensions"]["height_px"]

    all_slots = [s["slot_id"] for s in planogram["slots"]]
    sample_skus = list(CLASS_TO_ID.keys())

    splits = [("train", n_train), ("val", n_val)]
    counts = {"train": 0, "val": 0, "total_boxes": 0}

    for split_name, num_samples in splits:
        img_dir = os.path.join(output_dir, "images", split_name)
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(1, num_samples + 1):
            file_stem = f"shelf_{split_name}_{i:04d}"
            img_path = os.path.join(img_dir, f"{file_stem}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{file_stem}.txt")

            # Stochastically inject realistic retail failure modes
            num_oos = random.choices([0, 1, 2, 3], weights=[0.25, 0.35, 0.25, 0.15])[0]
            oos_slots = random.sample(all_slots, k=num_oos) if num_oos > 0 else []

            remaining_slots = [s for s in all_slots if s not in oos_slots]
            num_misplaced = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
            misplaced_slots = {}
            if num_misplaced > 0 and len(remaining_slots) >= num_misplaced:
                chosen = random.sample(remaining_slots, k=num_misplaced)
                for c in chosen:
                    misplaced_slots[c] = random.choice(sample_skus[:8])
                    remaining_slots.remove(c)

            num_encroached = random.choices([0, 1], weights=[0.75, 0.25])[0]
            encroached_slots = {}
            if num_encroached > 0 and len(remaining_slots) >= num_encroached:
                chosen = random.sample(remaining_slots, k=num_encroached)
                for c in chosen:
                    encroached_slots[c] = "Competitor Encroaching SKU"

            # Render image
            _, detections = generate_synthetic_shelf_image(
                planogram=planogram,
                output_image_path=img_path,
                oos_slot_ids=oos_slots,
                encroached_slot_ids=encroached_slots,
                misplaced_slot_ids=misplaced_slots,
            )

            # Write YOLO label format: <class_id> <x_center> <y_center> <w> <h>
            label_lines = []
            for det in detections:
                sku_name = det["sku_name"]
                cls_id = CLASS_TO_ID.get(sku_name, 11)
                bx1, by1, bx2, by2 = det["box_px"]

                x_center = ((bx1 + bx2) / 2.0) / width
                y_center = ((by1 + by2) / 2.0) / height
                w_norm = (bx2 - bx1) / float(width)
                h_norm = (by2 - by1) / float(height)

                # Clamp within [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                counts["total_boxes"] += 1

            with open(lbl_path, "w", encoding="utf-8") as f:
                f.writelines(label_lines)

            counts[split_name] += 1

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shelf-Vision data preparation & dataset generator.")
    parser.add_argument("--generate-yolo", action="store_true", help="Generate full YOLOv8 dataset")
    parser.add_argument("--train-samples", type=int, default=60, help="Number of training images")
    parser.add_argument("--val-samples", type=int, default=15, help="Number of validation images")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    planogram_file = os.path.join(os.path.dirname(__file__), "..", "configs", "planogram_sample.json")
    demo_image = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_shelf.jpg")

    plano = load_planogram(planogram_file)

    if args.generate_yolo:
        out_path = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
        print(f"Generating YOLOv8 dataset in: {out_path}")
        stats = generate_yolo_dataset(
            planogram_path=planogram_file,
            output_dir=out_path,
            n_train=args.train_samples,
            n_val=args.val_samples,
        )
        print(f"✅ Generated {stats['train']} training images, {stats['val']} validation images ({stats['total_boxes']} total SKU bounding boxes).")
    else:
        # Generate default fixture demo image
        img_path, dets = generate_synthetic_shelf_image(
            planogram=plano,
            output_image_path=demo_image,
            oos_slot_ids=["S1-P03", "S2-P04"],
            misplaced_slot_ids={"S2-P02": "Nestle Everyday Dairy Whitener 1kg"},
            encroached_slot_ids={"S3-P04": "Competitor Rogue Ghee 1L"},
        )
        print(f"Generated synthetic dark-store shelf image: {img_path}")
        print(f"Rendered {len(dets)} detected facings across shelf grid.")
