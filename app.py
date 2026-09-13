"""Streamlit interface for Smart Waste Segregation."""
from __future__ import annotations

import io
import os
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.waste_info import CLASS_COLORS, get_waste_info

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
    """Load the production checkpoint, with YOLOv8n only as a development fallback."""
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
    """Convert YOLO results into stable app detections.

    Predictions below accept_conf are retained as Unknown/Review rather than
    silently being presented as a confident waste class.
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
            detections.append({
                "Class": mapped if is_known else UNKNOWN_CLASS,
                "RawClass": str(raw_name),
                "Confidence": f"{confidence:.0%}",
                "conf_val": confidence,
                "Status": "Accepted" if is_known else "Review",
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


def run_inference(image_rgb: np.ndarray, model: Any, infer_conf: float = 0.10, accept_conf: float = 0.35, iou: float = 0.45):
    """Run low-floor YOLO inference, then apply an explicit acceptance threshold."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=infer_conf, iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    return image_rgb, filter_detections(detections)


def annotate_image(image_rgb: np.ndarray, detections: list[dict[str, Any]], colors_rgb: dict[str, tuple[int, int, int]]) -> np.ndarray:
    """Draw readable detection overlays and return RGB image."""
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        label = f"{det['Class'].title()}  {det['Confidence']}"
        rgb = colors_rgb.get(det["Class"], (120, 120, 120))
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        label_y1 = max(0, y1 - th - 10)
        label_y2 = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, label_y1), (min(width - 1, x1 + tw + 10), label_y2), bgr, -1)
        cv2.putText(canvas, label, (x1 + 5, label_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Smart Waste Segregation", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Theme-aware base colors: works with Streamlit light and dark themes. */
.stApp {
    background: var(--background-color);
    color: var(--text-color);
}
[data-testid="stHeader"] {
    background: var(--background-color);
}
.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f172a 0%, #164e63 55%, #166534 100%);
    color: #ffffff;
    margin-bottom: 1.2rem;
    box-shadow: 0 16px 40px rgba(15, 23, 42, .16);
}
.hero h1 { margin: 0; font-size: 2.45rem; letter-spacing: -1px; color: #ffffff; }
.hero p { margin: .55rem 0 0; opacity: .86; font-size: 1.03rem; color: #ffffff; }
.section-title {
    color: var(--text-color);
    font-size: 1.25rem;
    font-weight: 700;
    margin: .5rem 0 .7rem;
}
.info-card {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(127, 127, 127, .25);
    border-radius: 16px;
    background: var(--secondary-background-color);
    color: var(--text-color);
    box-shadow: 0 6px 18px rgba(15, 23, 42, .08);
    margin-bottom: .7rem;
}
.info-card h4,
.info-card p,
.info-card b { color: var(--text-color); }
.info-card h4 { margin: 0 0 .35rem; }
.muted { color: var(--text-color); opacity: .72; font-size: .9rem; }
.stButton > button { border-radius: 12px; font-weight: 700; min-height: 2.7rem; }
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid rgba(127, 127, 127, .25);
    padding: .8rem;
    border-radius: 15px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--text-color); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


def build_download_image(image_rgb):
    output = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=94)
    return buffer.getvalue()


def render_summary(detections):
    known = [d for d in detections if d["Class"] in TARGET_CLASSES]
    unknown = [d for d in detections if d["Class"] == UNKNOWN_CLASS]
    total = len(detections)
    avg_conf = sum(d["conf_val"] for d in known) / len(known) if known else 0
    cols = st.columns(4)
    cols[0].metric("Objects", total)
    cols[1].metric("Classified", len(known))
    cols[2].metric("Needs review", len(unknown))
    cols[3].metric("Avg. confidence", f"{avg_conf:.0%}" if known else "—")
    counts = {name: sum(d["Class"] == name for d in detections) for name in TARGET_CLASSES}
    counts[UNKNOWN_CLASS] = len(unknown)
    return counts


def render_waste_guide(classes):
    if not classes:
        return
    st.markdown('<div class="section-title">♻️ Segregation guide</div>', unsafe_allow_html=True)
    columns = st.columns(min(3, len(classes)))
    for index, class_name in enumerate(classes):
        info = get_waste_info(class_name)
        with columns[index % len(columns)]:
            st.markdown(f"""
            <div class="info-card">
                <h4>{class_name.title()}</h4>
                <div class="muted">{info['category']} · Suggested bin: {info['bin_color']}</div>
                <p><b>Examples:</b> {', '.join(info['examples']) if info['examples'] else '—'}</p>
                <p><b>Guidance:</b> {info['disposal']}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="hero">
        <h1>♻️ Smart Waste Segregation</h1>
        <p>AI-powered computer vision for practical waste classification and segregation guidance.</p>
    </div>
    """, unsafe_allow_html=True)

    model, model_path = get_model()
    if model is None:
        st.error("No usable YOLO model was found. Place the trained model at `model/best.pt` and install the dependencies from `requirements.txt`.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Detection settings")
        accept_conf = st.slider("Acceptance confidence", min_value=0.20, max_value=0.85, value=0.35, step=0.05,
                                help="Predictions below this score are retained as Unknown/Review instead of being presented as a confident class.")
        iou_thresh = st.slider("NMS IoU threshold", min_value=0.20, max_value=0.70, value=0.45, step=0.05,
                               help="Controls how strongly overlapping detections are suppressed by YOLO.")
        st.divider()
        st.caption("Model")
        st.code(os.path.relpath(model_path, os.path.dirname(os.path.abspath(__file__))), language="text")
        st.caption("Supported classes")
        st.write(" · ".join(name.title() for name in TARGET_CLASSES))
        with st.expander("About Unknown / Review"):
            st.write("Unknown is an uncertainty/rejection state. It is used when the model's confidence is below the acceptance threshold or its class name is not one of the supported waste categories. It is not a separately trained sixth class.")

    st.markdown('<div class="section-title">📷 Choose an input</div>', unsafe_allow_html=True)
    upload_col, camera_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    with camera_col:
        captured = st.camera_input("Take a photo")

    source = captured if captured is not None else uploaded
    if source is None:
        st.info("Upload a waste image or take a photo to start detection.")
        st.markdown("""
        <div class="info-card">
            <h4>How it works</h4>
            <p>1. Provide an image → 2. YOLOv8 detects candidate objects → 3. Low-confidence predictions are flagged for review → 4. The app shows categories, confidence, and disposal guidance.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        image = Image.open(source).convert("RGB")
        st.image(image, caption="Input image", width="stretch")
    except Exception as exc:
        st.error(f"Could not read the selected image: {exc}")
        return

    if st.button("🔎 Detect Waste", type="primary", use_container_width=True):
        with st.spinner("Analyzing image with YOLOv8…"):
            image_rgb, detections = run_inference(np.array(image), model=model, infer_conf=max(0.10, min(accept_conf - 0.10, 0.30)), accept_conf=accept_conf, iou=iou_thresh)
            annotated = annotate_image(image_rgb, detections, CLASS_COLORS)

        counts = render_summary(detections)
        st.divider()
        result_col, detail_col = st.columns([1.35, 1])
        with result_col:
            st.markdown('<div class="section-title">🖼️ Detection result</div>', unsafe_allow_html=True)
            st.image(annotated, width="stretch")
            st.download_button("⬇️ Download annotated image", data=build_download_image(annotated), file_name="smart_waste_detection.jpg", mime="image/jpeg", use_container_width=True)

        with detail_col:
            st.markdown('<div class="section-title">📊 Detection details</div>', unsafe_allow_html=True)
            if detections:
                rows = [{"#": index, "Class": d["Class"].title(), "Confidence": d["Confidence"], "Status": d["Status"]} for index, d in enumerate(detections, start=1)]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.warning("No candidate objects were detected. Try a clearer image or lower the acceptance confidence.")

            st.markdown('<div class="section-title">📦 Category counts</div>', unsafe_allow_html=True)
            count_rows = [{"Category": name.title(), "Count": count} for name, count in counts.items() if count > 0]
            if count_rows:
                st.dataframe(pd.DataFrame(count_rows), hide_index=True, use_container_width=True)

        known_classes = [name for name in TARGET_CLASSES if counts.get(name, 0) > 0]
        render_waste_guide(known_classes)
        if counts.get(UNKNOWN_CLASS, 0):
            st.warning(f"{counts[UNKNOWN_CLASS]} detection(s) need manual review. Do not make an automatic disposal decision from an Unknown result.")

    st.divider()
    st.caption("Smart Waste Segregation · YOLOv8 · Computer Vision · B.Tech Project")


if __name__ == "__main__":
    main()
