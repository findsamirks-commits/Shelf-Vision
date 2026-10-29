"""Command-line training harness for fine-tuning YOLOv8 on retail shelf datasets."""

import argparse
import os


def train_yolo(data_yaml: str, model_name: str = "yolov8m.pt", epochs: int = 50, imgsz: int = 1280, batch_size: int = 8):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Please run `pip install ultralytics`.")
        return

    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training on {data_yaml} for {epochs} epochs...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name="shelf_vision_yolov8",
        save=True,
    )
    print("Training complete! Model weights saved in `runs/detect/shelf_vision_yolov8/weights/best.pt`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on retail shelf dataset.")
    parser.add_argument("--data", type=str, default="configs/shelf_data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLOv8 base model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")

    args = parser.parse_args()
    train_yolo(args.data, args.model, args.epochs, args.imgsz, args.batch)
