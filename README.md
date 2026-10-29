# Shelf-Vision 🛒👁️
**Planogram Compliance & Out-of-Stock (OOS) Detection for Retail Shelves — powered by YOLOv8 object detection.**

A deep learning computer vision pipeline that audits retail shelf photos to automatically answer three questions store ops teams ask every day:
1. **Are my products on the shelf?** (OOS detection)
2. **Are they in the right place?** (planogram compliance)
3. **Is competitor product encroaching on my space?** (share-of-shelf)

Built with **Ultralytics YOLOv8**, **PyTorch**, **OpenCV**, and a **Streamlit** interactive audit UI.

---

## 🎯 Business Problem

In quick-commerce dark stores and modern-trade retail, **shelf audits are manual, inconsistent, and lag reality by hours or days**. A single hub picker photographs ~200 shelves per shift; a merchandiser visits 5-10 stores a day. The result:

| Pain | Estimated Loss | Shelf-Vision Solution |
|---|---|---|
| **Phantom inventory → OOS at pick time** | 3–7% of GMV lost to cancellations | Automated empty slot detection alerts ERP/WMS instantly |
| **Planogram non-compliance** | 15–25% sales lift forgone on launch SKUs | Hungarian spatial assignment checks SKU placements |
| **Competitor over-facing / encroachment** | 5–10% share-of-shelf erosion per quarter | Bounding box brand classifier calculates live Share-of-Shelf % |

**Shelf-Vision** replaces the clipboard with a camera + a model. Upload a shelf photo → get a structured JSON compliance report in **< 100 ms**.

---

## 🏗️ Architecture

```
                          ┌──────────────────────┐
   shelf photo  ─────►    │  YOLOv8 detector     │ ──► SKU bounding boxes + class
   (.jpg/.png)            │  (fine-tuned on      │     + confidence
                          │   SKU-110K / custom) │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Planogram Matcher   │ ──► {expected vs actual}
                          │  (spatial IoU +      │     per shelf slot
                          │   Hungarian match)   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Compliance Engine   │ ──► JSON + annotated image
                          │  - OOS slots         │
                          │  - Misplaced SKUs    │
                          │  - Share-of-shelf %  │
                          │  - Compliance score  │
                          └──────────────────────┘
```

### Why YOLOv8 (not DETR / Faster R-CNN)?
- **Real-time inference** on edge devices (store CCTV, picker phone) — ~45 FPS on T4 GPU, ~12 FPS on CPU.
- **Mature ecosystem** (Ultralytics) — one-command fine-tune, built-in augmentations.
- **Dense small-object regime** — retail shelves have 50–200 SKUs per image; YOLOv8 handles dense overlapping bounding boxes effectively.
- **Easy export** to ONNX / CoreML / TensorRT for mobile and edge deployment.

### Why a separate Planogram-Matcher (not end-to-end)?
Planograms change frequently (promo swaps, seasonal resets, store-size variants). Decoupling detection from compliance logic means:
- Detector trains once, planograms update via a clean JSON config file.
- Category managers can edit planograms without retraining neural weights.
- The same detector powers OOS, compliance, AND share-of-shelf analysis.

---

## 📂 Project Structure

```
Shelf-Vision/
├── README.md                      # Architecture, business impact, evaluation metrics
├── requirements.txt               # Dependencies (ultralytics, torch, opencv, scipy, streamlit)
├── .gitignore
├── configs/
│   ├── planogram_sample.json      # Expected SKU layout, tier definitions, normalized boxes
│   └── shelf_data.yaml            # YOLOv8 dataset configuration
├── data/                          # Dataset directory (raw & YOLO-format splits)
├── src/
│   ├── __init__.py
│   ├── data_prep.py               # Synthetic shelf generator & annotation builder
│   ├── planogram_matcher.py       # Bipartite Hungarian matching (linear_sum_assignment)
│   ├── compliance_engine.py       # OOS, misplacement, brand share-of-shelf metrics
│   ├── inference.py               # End-to-end detection, spatial assignment & markup pipeline
│   └── train.py                   # YOLOv8 fine-tuning CLI harness
├── app/
│   └── streamlit_app.py           # Interactive visual audit UI
├── tests/
│   └── test_planogram.py          # Pytest suite for Hungarian matcher & compliance
└── assets/
    └── demo_shelf.jpg             # Sample dark-store shelf audit fixture
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/findsamirks-commits/Shelf-Vision.git
cd Shelf-Vision
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Generate or Audit Sample Shelf
Run the standalone pipeline directly:
```bash
python -c "from src.inference import ShelfVisionPipeline; p = ShelfVisionPipeline('configs/planogram_sample.json'); r, img = p.audit_shelf('assets/demo_shelf.jpg'); print(r)"
```

### 3. Launch the Streamlit Interactive UI
```bash
streamlit run app/streamlit_app.py
```
Upload or regenerate shelf audit photos → inspect bounding boxes, compliance flags, brand share-of-shelf distributions, and export machine-readable JSON reports.

---

## 📊 Example Output

```json
{
  "shelf_id": "MILKBASKET-HUB-A4-BAY2",
  "category": "Dairy & Breakfast Spreads",
  "compliance_score": 78.6,
  "total_slots": 14,
  "compliant_facings": 11,
  "oos_facings": 2,
  "misplaced_facings": 1,
  "encroached_facings": 0,
  "oos_rate_pct": 14.3,
  "share_of_shelf_pct": {
    "Amul": 41.7,
    "Mother Dairy": 25.0,
    "Nestle": 16.7,
    "Milkbasket Private Label": 16.7
  },
  "oos_slots": [
    {
      "slot_id": "S1-P03",
      "expected_sku": "Mother Dairy Butter 500g",
      "expected_brand": "Mother Dairy",
      "shelf_tier": 1
    },
    {
      "slot_id": "S2-P04",
      "expected_sku": "Nestle Milkmaid 400g",
      "expected_brand": "Nestle",
      "shelf_tier": 2
    }
  ],
  "misplaced_slots": [
    {
      "slot_id": "S2-P02",
      "expected_sku": "Amul Cheese Slices 200g",
      "detected_sku": "Nestle Everyday Dairy Whitener 1kg",
      "shelf_tier": 2
    }
  ],
  "inference_latency_ms": 21.2
}
```

---

## 🧪 Evaluation Metrics

| Metric | Target | Measurement |
|---|---|---|
| **mAP@0.5 (Detection)** | $\ge 0.80$ | YOLOv8 built-in COCO-format validator |
| **OOS Recall** | $\ge 0.90$ | $\frac{\text{True Empty Slots Detected}}{\text{All Actual Empty Slots}}$ |
| **Planogram F1** | $\ge 0.85$ | Harmonic mean of correct slot assignments |
| **Inference Latency** | $< 150 \text{ ms}$ | Evaluated on full $1280 \times 720$ shelf photo (CPU/GPU) |

---

## 🔗 Companion Repositories

- [`Agentic-Inventory-Optimizer`](https://github.com/findsamirks-commits/Agentic-Inventory-Optimizer) — Shelf-Vision OOS alerts feed directly into automated replenishment Purchase Orders.
- [`Perishable-Markdown-Optimizer`](https://github.com/findsamirks-commits/Perishable-Markdown-Optimizer) — Over-facing and slow-moving shelf inventory feeds dynamic discount ladders.
- [`A2A-Vendor-Negotiation-Arena`](https://github.com/findsamirks-commits/A2A-Vendor-Negotiation-Arena) — Displaced or encroached brand share feeds vendor negotiation counter-demands.

---

## 📄 License
MIT License. Built by [Samir Sandhwar](https://github.com/findsamirks-commits) — Category Head & Retail AI Engineer.
