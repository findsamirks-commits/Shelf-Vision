"""Command-line training harness for fine-tuning YOLOv8 on retail shelf datasets.

Trains Ultralytics YOLOv8 for retail dark-store SKU detection:
- Ingests configs/shelf_data.yaml
- Uses pretrained yolov8n.pt (or yolov8s/yolov8m) backbone
- Saves best weights to weights/best.pt and runs/detect/
- Runs post-training validation to log mAP@50 and mAP@50-95 metrics
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Fix for Windows on ARM64 / Snapdragon CPU feature check in Polars
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
# Ensure temporary writable directory for ultralytics settings
config_dir = os.path.join(tempfile.gettempdir(), "Ultralytics")
os.makedirs(config_dir, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = config_dir


def train_yolo(
    data_yaml: str = "configs/shelf_data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 4,
    device: str = "cpu",
    output_weights_dir: str = "weights",
):
    """Fine-tunes YOLOv8 on grocery shelves dataset."""
    # Ensure temporary writable directory for ultralytics settings
    config_dir = os.path.join(tempfile.gettempdir(), "Ultralytics")
    os.makedirs(config_dir, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = config_dir

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Please run `pip install ultralytics`.")
        return

    # Resolve paths
    root_dir = Path(__file__).resolve().parent.parent
    if not os.path.isabs(data_yaml):
        data_yaml = str(root_dir / data_yaml)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    print("=" * 70)
    print("🛒 Starting Shelf-Vision YOLOv8 Fine-Tuning Pipeline")
    print(f"   Model Backbone   : {model_name}")
    print(f"   Dataset Config   : {data_yaml}")
    print(f"   Epochs           : {epochs}")
    print(f"   Image Resolution : {imgsz}x{imgsz}")
    print(f"   Batch Size       : {batch_size}")
    print(f"   Device           : {device}")
    print("=" * 70)

    # Load pretrained model
    print(f"\n📦 Loading pretrained backbone: {model_name}...")
    model = YOLO(model_name)

    # Train model
    print("\n🚀 Beginning fine-tuning on shelf dataset...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name="shelf_vision_yolov8",
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    # Locate best.pt
    run_save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and model.trainer else None
    best_weights_path = None

    if run_save_dir:
        candidate = run_save_dir / "weights" / "best.pt"
        if candidate.exists():
            best_weights_path = candidate

    # Fallback search if trainer path differs
    if not best_weights_path:
        search_paths = list(root_dir.glob("runs/**/best.pt"))
        if search_paths:
            best_weights_path = max(search_paths, key=os.path.getmtime)

    # Export best weights to canonical weights/ directory
    dest_weights_dir = root_dir / output_weights_dir
    dest_weights_dir.mkdir(parents=True, exist_ok=True)
    dest_weights_file = dest_weights_dir / "best.pt"

    if best_weights_path and best_weights_path.exists():
        shutil.copy2(best_weights_path, dest_weights_file)
        print(f"\n✅ Fine-tuned weights exported to: {dest_weights_file}")
    else:
        print("\n⚠️ Note: Check runs/detect/ for output weights.")

    # Validation & Metric Extraction
    print("\n" + "=" * 70)
    print("📊 Evaluating Fine-Tuned Model on Validation Split")
    print("=" * 70)
    try:
        val_results = model.val(data=data_yaml, imgsz=imgsz, device=device)
        map50 = float(val_results.box.map50) if hasattr(val_results.box, "map50") else 0.0
        map_all = float(val_results.box.map) if hasattr(val_results.box, "map") else 0.0
        precision = float(val_results.box.mp) if hasattr(val_results.box, "mp") else 0.0
        recall = float(val_results.box.mr) if hasattr(val_results.box, "mr") else 0.0

        print(f"  mAP@50     : {map50 * 100:.2f}%")
        print(f"  mAP@50-95  : {map_all * 100:.2f}%")
        print(f"  Precision  : {precision * 100:.2f}%")
        print(f"  Recall     : {recall * 100:.2f}%")
    except Exception as e:
        print(f"Validation summary: {e}")

    # Test sample inference on demo fixture
    demo_img = str(root_dir / "assets" / "demo_shelf.jpg")
    if os.path.exists(demo_img):
        print(f"\n🔍 Testing inference on sample audit fixture: {demo_img}")
        test_pred = model.predict(demo_img, conf=0.25, imgsz=imgsz, device=device, verbose=False)
        num_dets = len(test_pred[0].boxes) if test_pred else 0
        print(f"  Detected {num_dets} SKU facings on test image.")

    print("=" * 70)
    print("🎉 Shelf-Vision YOLOv8 Training Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on retail shelf dataset.")
    parser.add_argument("--data", type=str, default="configs/shelf_data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 base model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--output-weights", type=str, default="weights", help="Directory to save final best.pt")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_weights_dir=args.output_weights,
    )
