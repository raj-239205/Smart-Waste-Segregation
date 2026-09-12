"""Shared inference and post-processing utilities for Smart Waste Segregation."""
from __future__ import annotations

from typing import Any
import os
import cv2
import numpy as np
from ultralytics import YOLO

TARGET_CLASSES = ("plastic", "paper", "metal", "glass", "organic")
UNKNOWN_CLASS = "unknown"


def _box_area(box: list[int]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(box_a: list[int], box_b: list[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    return intersection / union if union else 0.0


def _containment(box_a: list[int], box_b: list[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(_box_area(box_a), _box_area(box_b))
    return intersection / smaller if smaller else 0.0


def load_model(model_path: str | None = None) -> tuple[Any | None, str | None]:
    """Load the trained model, with YOLOv8n as an explicit development fallback."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates.extend(
        [
            os.path.join(base_dir, "model", "best.pt"),
            os.path.join(base_dir, "yolov8n.pt"),
        ]
    )

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


def prepare_detections(
    results: Any,
    class_names: Any,
    accept_conf: float = 0.35,
) -> list[dict[str, Any]]:
    """Convert Ultralytics results into stable application-level detections.

    Detections below accept_conf are retained as Unknown/Review instead of being
    silently forced into one of the five waste classes.
    """
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
            is_known = mapped in TARGET_CLASSES and confidence >= accept_conf

            detections.append(
                {
                    "Class": mapped if is_known else UNKNOWN_CLASS,
                    "RawClass": str(raw_name),
                    "Confidence": f"{confidence:.0%}",
                    "conf_val": confidence,
                    "Status": "Accepted" if is_known else "Review",
                    "box": coords,
                }
            )
    return detections


def filter_detections(
    detections: list[dict[str, Any]],
    same_class_iou: float = 0.45,
    containment: float = 0.72,
) -> list[dict[str, Any]]:
    """Remove obvious duplicate/sub-box detections without hard-coded image coordinates.

    Earlier versions contained scene-specific rules tied to pixel positions and
    manually assigned confidence values. Those rules do not generalize to new
    images. This function intentionally uses geometry + model confidence only.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.get("conf_val", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []

    for det in ordered:
        duplicate = False
        for existing in kept:
            iou = _iou(det["box"], existing["box"])
            contained = _containment(det["box"], existing["box"])

            # Same semantic class: conventional duplicate suppression.
            if det["Class"] == existing["Class"] and (
                iou >= same_class_iou or contained >= containment
            ):
                duplicate = True
                break

            # Unknown boxes should not create multiple overlapping review alerts.
            if det["Class"] == UNKNOWN_CLASS and existing["Class"] == UNKNOWN_CLASS:
                if iou >= 0.50 or contained >= 0.80:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(det)

    return kept


def run_inference(
    image_rgb: np.ndarray,
    model: Any,
    infer_conf: float = 0.10,
    accept_conf: float = 0.35,
    iou: float = 0.45,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run low-floor inference, then apply an explicit acceptance threshold."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=infer_conf, iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    detections = filter_detections(detections)
    return image_rgb, detections


def annotate_image(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    colors_rgb: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Draw polished, readable detection overlays and return an RGB image."""
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        label_class = det["Class"]
        label = f"{label_class.title()}  {det['Confidence']}"
        rgb = colors_rgb.get(label_class, (120, 120, 120))
        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y1 = max(0, y1 - th - 10)
        label_y2 = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, label_y1), (min(width - 1, x1 + tw + 10), label_y2), bgr, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, label_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
