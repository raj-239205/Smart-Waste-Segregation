"""Train and validate the Smart Waste Segregation YOLOv8 model."""
from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train the waste-segmentation detector")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Optional CUDA device, e.g. 0")
    parser.add_argument("--model", default="yolov8n.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting YOLOv8 training for Smart Waste Segregation…")
    print(f"Dataset: data.yaml | Epochs: {args.epochs} | Image size: {args.imgsz} | Batch: {args.batch}")

    model = YOLO(args.model)
    train_kwargs = dict(
        data="data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        patience=12,
        seed=42,
        project="runs/detect",
        name="waste_seg",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    best_model_path = os.path.join("runs", "detect", "waste_seg", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Training finished but best.pt was not found at {best_model_path}")

    os.makedirs("model", exist_ok=True)
    shutil.copy2(best_model_path, "model/best.pt")
    print(f"\nBest model copied to: {os.path.abspath('model/best.pt')}")

    print("\nRunning validation on the best checkpoint…")
    metrics = model.val(data="data.yaml", imgsz=args.imgsz, plots=True)
    try:
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    except AttributeError:
        print("Validation completed. See runs/detect/waste_seg for detailed metrics.")


if __name__ == "__main__":
    main()
