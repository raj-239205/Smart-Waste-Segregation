"""Standalone image and webcam inference for Smart Waste Segregation."""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from utils.detection import annotate_image, filter_detections, load_model, prepare_detections
from utils.waste_info import CLASS_COLORS


# OpenCV needs BGR colors; the shared utility stores RGB values.
CLASS_COLORS_BGR = {key: (value[2], value[1], value[0]) for key, value in CLASS_COLORS.items()}


def detect_frame(model, frame, accept_conf=0.35, iou=0.45):
    results = model(frame, conf=max(0.10, min(accept_conf - 0.10, 0.30)), iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    detections = filter_detections(detections)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_image(rgb, detections, CLASS_COLORS)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR), detections


def run_image(model, image_path, output_path="result.jpg", accept_conf=0.35, iou=0.45):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated, detections = detect_frame(model, frame, accept_conf, iou)
    cv2.imwrite(output_path, annotated)

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
            known = sum(det["Class"] != "unknown" for det in detections)
            unknown = len(detections) - known

            cv2.rectangle(annotated, (10, 10), (290, 78), (15, 23, 42), -1)
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
    model, model_path = load_model(args.model)
    if model is None:
        print("Error: no usable model was found. Expected model/best.pt.")
        sys.exit(1)

    print(f"Model: {os.path.relpath(model_path)}")
    if args.source and args.source.lower() != "webcam":
        run_image(model, args.source, args.output, args.conf, args.iou)
    else:
        run_webcam(model, args.camera, args.conf, args.iou)


if __name__ == "__main__":
    main()
