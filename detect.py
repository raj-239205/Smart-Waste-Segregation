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
    """Load the production checkpoint, with YOLOv8n as a development fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates.extend([
        os.path.join(base_dir, "model", "best.pt"),
        os.path.join(base_dir, "yolov8n.pt"),
    ])

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        try:
            return YOLO(path), path
        except Exception:
            continue
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
    """Convert YOLO results into stable detections with an explicit review state."""
    detections: list[dict[str, Any]] = []
    names = class_names if isinstance(class_names, dict) else dict(enumerate(class_names))

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
    """Suppress obvious duplicate/sub-box detections using geometry and confidence."""
    if not detections:
        return []

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
            if det["Class"] == UNKNOWN_CLASS and existing["Class"] == UNKNOWN_CLASS:
                if iou >= 0.50 or contained >= 0.80:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return kept


def detect_frame(model: Any, frame_bgr, accept_conf: float = 0.35, iou: float = 0.45):
    """Run inference on one BGR frame and return annotated BGR output plus detections."""
    infer_conf = max(0.10, min(accept_conf - 0.10, 0.30))
    results = model(frame_bgr, conf=infer_conf, iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    detections = filter_detections(detections)

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        rgb = CLASS_COLORS.get(det["Class"], CLASS_COLORS[UNKNOWN_CLASS])
        bgr = (rgb[2], rgb[1], rgb[0])
        label = f"{det['Class'].title()}  {det['Confidence']}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y1 = max(0, y1 - th - 10)
        label_y2 = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, label_y1), (min(width - 1, x1 + tw + 10), label_y2), bgr, -1)
        cv2.putText(canvas, label, (x1 + 5, label_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return canvas, detections


def run_image(model: Any, image_path: str, output_path: str = "result.jpg", accept_conf: float = 0.35, iou: float = 0.45):
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


def run_batch(model: Any, folder: str, output_dir: str = "runs/waste_tests", accept_conf: float = 0.35, iou: float = 0.45):
    """Run the same pipeline over every image in a folder and print a compact report."""
    os.makedirs(output_dir, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in extensions
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {folder}")

    total = accepted = review = 0
    print(f"Batch test: {len(image_paths)} image(s)")
    for image_path in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        detections = run_image(model, image_path, output_path, accept_conf, iou)
        total += len(detections)
        accepted += sum(det["Status"] == "Accepted" for det in detections)
        review += sum(det["Status"] == "Review" for det in detections)

    print("\nBatch summary")
    print(f"  Images tested: {len(image_paths)}")
    print(f"  Objects detected: {total}")
    print(f"  Accepted: {accepted}")
    print(f"  Review/Unknown: {review}")
    print(f"  Acceptance rate: {accepted / total:.1%}" if total else "  Acceptance rate: 0.0%")
    print(f"  Annotated outputs: {os.path.abspath(output_dir)}")


def run_webcam(model: Any, camera_index: int = 0, accept_conf: float = 0.35, iou: float = 0.45):
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

            cv2.rectangle(annotated, (10, 10), (290, 78), (15, 23, 42), -1)
            cv2.putText(annotated, f"FPS  {fps:.1f}", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Objects  {known}  |  Review  {unknown}", (22, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow("Smart Waste Segregation - Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Waste Segregation image/webcam detector")
    parser.add_argument("source", nargs="?", help="Image path, folder path, or 'webcam'. Omit to use webcam.")
    parser.add_argument("--model", default=None, help="Optional path to a YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.35, help="Acceptance confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold (default: 0.45)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--output", default="result.jpg", help="Output path for single-image detection")
    parser.add_argument("--batch-output", default="runs/waste_tests", help="Output directory for folder tests")
    return parser.parse_args()


def main():
    args = parse_args()
    model, model_path = load_model(args.model)
    if model is None:
        print("Error: no usable model was found. Expected model/best.pt.")
        sys.exit(1)

    print(f"Model: {os.path.relpath(model_path)}")
    if not args.source or args.source.lower() == "webcam":
        run_webcam(model, args.camera, args.conf, args.iou)
    elif os.path.isdir(args.source):
        run_batch(model, args.source, args.batch_output, args.conf, args.iou)
    else:
        run_image(model, args.source, args.output, args.conf, args.iou)


if __name__ == "__main__":
    main()
