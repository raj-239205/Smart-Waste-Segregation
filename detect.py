"""Standalone image and webcam inference for Smart Waste Segregation."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import cv2
from ultralytics import YOLO

from utils.waste_info import CLASS_COLORS

TARGET_CLASSES = ("plastic", "paper", "metal", "glass", "organic")
UNKNOWN_CLASS = "unknown"

# OpenCV uses BGR while the shared project colors are stored as RGB.
CLASS_COLORS_BGR = {key: (value[2], value[1], value[0]) for key, value in CLASS_COLORS.items()}


def _box_area(box: list[int]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(box_a: list[int], box_b: list[int]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    return intersection / union if union else 0.0


def _containment(box_a: list[int], box_b: list[int]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(_box_area(box_a), _box_area(box_b))
    return intersection / smaller if smaller else 0.0


def load_model(model_path: str | None = None) -> tuple[Any | None, str | None]:
    """Load the requested model, then the production checkpoint, then YOLOv8n."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [model_path] if model_path else []
    candidates.extend([
        os.path.join(base_dir, "model", "best.pt"),
        os.path.join(base_dir, "yolov8n.pt"),
    ])

    seen = set()
    for path in candidates:
        if not path or path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            return YOLO(path), path
        except Exception as exc:
            print(f"Could not load model {path}: {exc}")
    return None, None


def normalize_class_name(raw_name: str | None) -> str | None:
    if raw_name is None:
        return None
    name = str(raw_name).strip().lower()
    aliases = {
        "biodegradable": "organic",
        "bio": "organic",
        "cardboard": "paper",
        "carton": "paper",
    }
    if name in TARGET_CLASSES:
        return name
    return aliases.get(name)


def prepare_detections(results: Any, class_names: Any, accept_conf: float = 0.35) -> list[dict[str, Any]]:
    """Convert YOLO output into accepted or review-state detections."""
    names = class_names if isinstance(class_names, dict) else dict(enumerate(class_names))
    detections: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = [int(v) for v in box.xyxy[0].tolist()]
            raw_name = names.get(cls_id, str(cls_id))
            mapped = normalize_class_name(raw_name)
            accepted = mapped in TARGET_CLASSES and confidence >= accept_conf
            detections.append({
                "Class": mapped if accepted else UNKNOWN_CLASS,
                "RawClass": str(raw_name),
                "Confidence": f"{confidence:.0%}",
                "conf_val": confidence,
                "Status": "Accepted" if accepted else "Review",
                "box": coords,
            })
    return detections


def filter_detections(detections: list[dict[str, Any]], same_class_iou: float = 0.45, containment: float = 0.72) -> list[dict[str, Any]]:
    """Suppress obvious duplicate or nested boxes without scene-specific rules."""
    ordered = sorted(detections, key=lambda item: item.get("conf_val", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []

    for det in ordered:
        duplicate = False
        for existing in kept:
            iou = _iou(det["box"], existing["box"])
            contained = _containment(det["box"], existing["box"])
            if det["Class"] == existing["Class"] and (iou >= same_class_iou or contained >= containment):
                duplicate = True
                break
            if det["Class"] == UNKNOWN_CLASS and existing["Class"] == UNKNOWN_CLASS and (iou >= 0.50 or contained >= 0.80):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def annotate_frame(frame_bgr, detections):
    """Draw color-coded boxes directly on a BGR OpenCV frame."""
    height, width = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        color = CLASS_COLORS_BGR.get(det["Class"], CLASS_COLORS_BGR[UNKNOWN_CLASS])
        label = f"{det['Class'].title()}  {det['Confidence']}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y1 = max(0, y1 - th - 10)
        label_y2 = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, label_y1), (min(width - 1, x1 + tw + 10), label_y2), color, -1)
        cv2.putText(canvas, label, (x1 + 5, label_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def detect_frame(model, frame, accept_conf=0.35, iou=0.45):
    """Run inference and return an annotated BGR frame plus detection records."""
    infer_conf = max(0.10, min(accept_conf - 0.10, 0.30))
    results = model(frame, conf=infer_conf, iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    detections = filter_detections(detections)
    return annotate_frame(frame, detections), detections


def run_image(model, image_path, output_path="result.jpg", accept_conf=0.35, iou=0.45):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated, detections = detect_frame(model, frame, accept_conf, iou)
    if not cv2.imwrite(output_path, annotated):
        raise IOError(f"Could not write output image: {output_path}")

    print(f"Saved result to: {output_path}")
    print(f"Detections: {len(detections)}")
    for index, det in enumerate(detections, start=1):
        print(f"  {index}. {det['Class']} ({det['Confidence']}) - {det['Status']}")
    return detections


def run_webcam(model, camera_index=0, accept_conf=0.35, iou=0.45):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check camera permissions and device availability.")

    print("Webcam detection started. Press Q to quit.")
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps = 0.85 * fps + 0.15 * instant_fps if fps else instant_fps

            annotated, detections = detect_frame(model, frame, accept_conf, iou)
            known = sum(det["Class"] != UNKNOWN_CLASS for det in detections)
            unknown = len(detections) - known

            cv2.rectangle(annotated, (10, 10), (315, 78), (15, 23, 42), -1)
            cv2.putText(annotated, f"FPS  {fps:.1f}", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Objects  {known}  |  Review  {unknown}", (22, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow("Smart Waste Segregation - Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Waste Segregation image/webcam detector")
    parser.add_argument("source", nargs="?", help="Image path. Omit to use webcam, or pass 'webcam'.")
    parser.add_argument("--model", default=None, help="Optional path to a YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.35, help="Acceptance confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold (default: 0.45)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--output", default="result.jpg", help="Output path for image detection")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    if not 0.0 < args.iou < 1.0:
        raise SystemExit("--iou must be between 0 and 1")

    model, model_path = load_model(args.model)
    if model is None:
        print("Error: no usable model was found. Expected model/best.pt or yolov8n.pt.")
        sys.exit(1)

    print(f"Model: {os.path.relpath(model_path, os.getcwd())}")
    if args.source and args.source.lower() != "webcam":
        run_image(model, args.source, args.output, args.conf, args.iou)
    else:
        run_webcam(model, args.camera, args.conf, args.iou)


if __name__ == "__main__":
    main()
