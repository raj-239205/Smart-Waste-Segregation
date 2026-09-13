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
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr, conf=infer_conf, iou=iou, agnostic_nms=False, verbose=False)
    detections = prepare_detections(results, model.names, accept_conf=accept_conf)
    return image_rgb, filter_detections(detections)


def annotate_image(image_rgb: np.ndarray, detections: list[dict[str, Any]], colors_rgb: dict[str, tuple[int, int, int]]) -> np.ndarray:
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


def inject_css(dark: bool = True):
    if dark:
        bg, panel, panel2, text, muted, border = "#06141d", "#0b202c", "#102936", "#f3f7f8", "#a9bcc4", "#214452"
        accent, accent2 = "#25c76b", "#1ed760"
    else:
        bg, panel, panel2, text, muted, border = "#f4f8f7", "#ffffff", "#edf5f1", "#102329", "#60747a", "#d4e3de"
        accent, accent2 = "#138a52", "#18a85d"

    st.markdown(f"""
    <style>
    :root {{ --sw-bg:{bg}; --sw-panel:{panel}; --sw-panel2:{panel2}; --sw-text:{text}; --sw-muted:{muted}; --sw-border:{border}; --sw-accent:{accent}; --sw-accent2:{accent2}; }}
    .stApp {{ background:var(--sw-bg); color:var(--sw-text); }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stToolbar"] {{ visibility:hidden; }}
    .block-container {{ max-width:1450px; padding:1.25rem 2rem 2rem; }}
    [data-testid="stSidebar"] {{ background:#071720; border-right:1px solid #173743; }}
    [data-testid="stSidebar"] > div {{ padding-top:1.2rem; }}
    [data-testid="stSidebar"] * {{ color:#edf7f5 !important; }}
    .brand {{ display:flex; align-items:center; gap:12px; padding:0 8px 18px; border-bottom:1px solid #21404b; }}
    .brand-icon {{ font-size:2.25rem; line-height:1; }}
    .brand-title {{ font-size:1.15rem; font-weight:800; line-height:1.05; }}
    .brand-title span {{ color:#2bd875; }}
    .brand-sub {{ color:#91a9b0 !important; font-size:.72rem; margin:8px 8px 18px; }}
    .nav-item {{ padding:11px 12px; border-radius:10px; margin:5px 0; font-size:.92rem; color:#dce9eb; }}
    .nav-active {{ background:linear-gradient(90deg,#15985a,#1eae67); color:white; font-weight:700; }}
    .side-spacer {{ height:20vh; }}
    .eco-side {{ padding:16px 12px; border-top:1px solid #21404b; color:#8eb0b6; text-align:center; }}
    .eco-side .big {{ color:#2bd875; font-weight:800; font-size:1rem; }}
    .hero-row {{ display:flex; align-items:center; justify-content:space-between; gap:20px; margin:4px 0 16px; }}
    .hero-title {{ margin:0; font-size:2.55rem; font-weight:850; letter-spacing:-1.4px; color:var(--sw-text); }}
    .hero-title .green {{ color:#2bd875; }}
    .hero-sub {{ margin:4px 0 0; color:var(--sw-muted); font-size:1rem; }}
    .hero-eco {{ text-align:right; color:#d9e8e7; font-size:.9rem; line-height:1.25; }}
    .hero-eco .globe {{ font-size:3.3rem; display:inline-block; vertical-align:middle; margin-right:8px; }}
    .banner {{ background:linear-gradient(110deg,#09252c,#0b3033); border:1px solid #17605b; border-radius:12px; padding:15px 18px; margin:8px 0 18px; display:flex; justify-content:space-between; align-items:center; gap:20px; }}
    .banner-title {{ font-weight:700; color:#f1faf7; font-size:.98rem; }}
    .banner-copy {{ color:#a9c5c3; font-size:.83rem; margin-top:3px; }}
    .banner-quote {{ color:#57e293; font-style:italic; font-size:.85rem; text-align:right; }}
    .panel {{ background:var(--sw-panel); border:1px solid var(--sw-border); border-radius:14px; padding:16px; height:100%; box-shadow:0 10px 30px rgba(0,0,0,.10); }}
    .panel-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:13px; }}
    .step {{ width:30px; height:30px; display:inline-flex; align-items:center; justify-content:center; border-radius:50%; background:#1fb965; color:white; font-weight:800; margin-right:9px; }}
    .panel-title {{ color:var(--sw-text); font-weight:800; font-size:1.02rem; }}
    .panel-copy {{ color:var(--sw-muted); font-size:.78rem; margin:2px 0 0 40px; }}
    .detect-pill {{ background:#0b5d3d; color:#8df3bd; padding:8px 13px; border-radius:20px; font-size:.78rem; font-weight:800; }}
    .image-card {{ background:#071820; border:1px solid #214452; border-radius:9px; padding:8px; }}
    .image-label {{ color:#e9f2f2; font-size:.8rem; font-weight:700; margin:0 0 7px 2px; }}
    .drop-zone {{ border:2px dashed #4d6873; border-radius:10px; min-height:150px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; color:#9db1b8; background:#091923; margin-top:12px; }}
    .drop-icon {{ font-size:2.5rem; color:#9fb9c2; }}
    .drop-title {{ color:#eaf4f3; font-size:.96rem; margin-top:6px; }}
    .drop-sub {{ font-size:.74rem; margin-top:6px; }}
    .summary {{ margin-top:16px; background:var(--sw-panel); border:1px solid var(--sw-border); border-radius:14px; padding:16px; }}
    .summary-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
    .summary-title {{ color:var(--sw-text); font-weight:800; font-size:1.12rem; }}
    .summary-copy {{ color:var(--sw-muted); font-size:.76rem; margin-top:2px; }}
    .summary-download {{ text-align:right; }}
    .waste-card {{ border:1px solid; border-radius:11px; padding:13px; min-height:190px; background:rgba(255,255,255,.025); }}
    .waste-icon {{ width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.35rem; margin-bottom:9px; }}
    .waste-name {{ font-size:.95rem; font-weight:800; }}
    .waste-count {{ color:var(--sw-text); font-size:.8rem; margin-top:2px; }}
    .confidence {{ color:var(--sw-text); font-size:.76rem; margin-top:13px; }}
    .bar {{ height:8px; background:rgba(255,255,255,.12); border-radius:8px; overflow:hidden; margin-top:6px; }}
    .bar > span {{ display:block; height:100%; border-radius:8px; }}
    .dispose {{ margin-top:12px; color:var(--sw-muted); font-size:.72rem; }}
    .dispose strong {{ display:block; color:var(--sw-text); font-size:.82rem; margin-top:2px; }}
    .unknown-box {{ margin-top:14px; border:1px solid #1d6b4c; background:#09261f; border-radius:11px; padding:14px 16px; color:#dbeee8; }}
    .unknown-title {{ color:#f0c24a; font-weight:800; font-size:.9rem; }}
    .unknown-copy {{ color:#a9c1bb; font-size:.76rem; margin-top:4px; }}
    .footer {{ border-top:1px solid var(--sw-border); margin-top:18px; padding-top:13px; display:flex; justify-content:space-between; color:var(--sw-muted); font-size:.75rem; }}
    .stButton > button {{ border-radius:9px; min-height:2.55rem; font-weight:750; border:1px solid #315466; background:#0b202c; color:#edf7f7; }}
    .stButton > button:hover {{ border-color:#2bd875; color:#ffffff; }}
    button[kind="primary"] {{ background:linear-gradient(90deg,#19a85f,#22c66d) !important; border:none !important; color:white !important; }}
    [data-testid="stFileUploader"] {{ background:#091923; border:1px solid #214452; border-radius:10px; padding:7px; }}
    [data-testid="stFileUploaderDropzone"] {{ background:transparent; border:none; }}
    [data-testid="stCameraInput"] {{ background:#091923; border:1px solid #214452; border-radius:10px; padding:7px; }}
    [data-testid="stImage"] img {{ border-radius:7px; }}
    [data-testid="stMetric"] {{ background:var(--sw-panel2); border:1px solid var(--sw-border); border-radius:10px; }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--sw-border); border-radius:9px; overflow:hidden; }}
    @media (max-width: 900px) {{ .hero-title {{font-size:2rem;}} .hero-eco {{display:none;}} .block-container {{padding:1rem;}} }}
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


def render_summary_cards(detections):
    known = [d for d in detections if d["Class"] in TARGET_CLASSES]
    unknown = [d for d in detections if d["Class"] == UNKNOWN_CLASS]
    counts = {name: sum(d["Class"] == name for d in detections) for name in TARGET_CLASSES}

    st.markdown('<div class="summary">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-head">
      <div><div class="summary-title">📊 Detection Summary</div><div class="summary-copy">Category-wise detection results and disposal recommendations</div></div>
      <div class="detect-pill">✓ {len(detections)} objects detected</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    icons = {"plastic": "♻", "paper": "▤", "metal": "▣", "glass": "♢", "organic": "⌁"}
    border = {"plastic": "#208ee8", "paper": "#9a57ef", "metal": "#f05b61", "glass": "#f0ad35", "organic": "#1fc878"}
    icon_bg = {"plastic": "#143d68", "paper": "#39205c", "metal": "#5a2229", "glass": "#5a4219", "organic": "#154b35"}

    for idx, name in enumerate(TARGET_CLASSES):
        count = counts[name]
        info = get_waste_info(name)
        avg = [d["conf_val"] for d in known if d["Class"] == name]
        confidence = sum(avg) / len(avg) if avg else 0
        with cols[idx]:
            st.markdown(f"""
            <div class="waste-card" style="border-color:{border[name]}">
              <div class="waste-icon" style="background:{icon_bg[name]};color:{border[name]}">{icons[name]}</div>
              <div class="waste-name" style="color:{border[name]}">{name.title()}</div>
              <div class="waste-count">{count} object{'s' if count != 1 else ''}</div>
              <div class="confidence">Confidence: {confidence:.0%}</div>
              <div class="bar"><span style="width:{max(3, confidence*100):.0f}%;background:{border[name]}"></span></div>
              <div class="dispose">Dispose in:<strong>{info['bin_color']} / {info['category']}</strong></div>
            </div>
            """, unsafe_allow_html=True)

    if unknown:
        st.markdown(f"""
        <div class="unknown-box">
          <div class="unknown-title">⚠️ Unknown Items</div>
          <div class="unknown-copy">{len(unknown)} item(s) need manual review because the system could not confidently assign a supported waste category.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="unknown-box">
          <div class="unknown-title">⚠️ Unknown Items</div>
          <div class="unknown-copy">No unknown items detected. All returned detections passed the current acceptance threshold.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return counts


def main():
    dark_mode = st.session_state.get("dark_mode", True)
    inject_css(dark_mode)

    with st.sidebar:
        st.markdown("""
        <div class="brand"><div class="brand-icon">🍃</div><div class="brand-title">Smart Waste<br><span>Segregation</span></div></div>
        <div class="brand-sub">Detect · Classify · A Cleaner Tomorrow</div>
        <div class="nav-item nav-active">⌂ &nbsp; Home</div>
        <div class="nav-item">▧ &nbsp; Upload Image</div>
        <div class="nav-item">▣ &nbsp; Camera</div>
        <div class="nav-item">⚙ &nbsp; How It Works</div>
        <div class="nav-item">▤ &nbsp; Waste Guide</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="side-spacer"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="eco-side"><div style="font-size:3rem">🌱</div><div class="big">Small Actions<br>Big Impact</div><div style="margin-top:10px">Clean Today<br>Greener Tomorrow</div></div>
        """, unsafe_allow_html=True)
        if st.toggle("Dark mode", value=dark_mode, key="dark_mode", help="Switch between the project dark and light theme.") != dark_mode:
            st.rerun()

    model, model_path = get_model()
    if model is None:
        st.error("No usable YOLO model was found. Place the trained model at model/best.pt and install the dependencies.")
        st.stop()

    st.markdown("""
    <div class="hero-row">
      <div><h1 class="hero-title">Smart <span class="green">Waste Segregation</span></h1><p class="hero-sub">Using Computer Vision for a Cleaner and Greener Tomorrow</p></div>
      <div class="hero-eco"><span class="globe">🌍</span><strong>Reduce ♻ Reuse ♻ Recycle</strong></div>
    </div>
    <div class="banner"><div><div class="banner-title">🍃 &nbsp; Upload an image or use your camera to detect and classify waste items.</div><div class="banner-copy">The system identifies waste into categories and suggests the correct disposal method.</div></div><div class="banner-quote">“Small Actions<br>Make a Big Difference”</div></div>
    """, unsafe_allow_html=True)

    input_col, result_col = st.columns([1, 1.55], gap="medium")

    with input_col:
        st.markdown("""
        <div class="panel">
          <div class="panel-head"><div><span class="step">1</span><span class="panel-title">Upload Image or Use Camera</span><div class="panel-copy">Upload an image or use your camera</div></div></div>
        """, unsafe_allow_html=True)
        tab_upload, tab_camera = st.tabs(["📤  Upload Image", "📷  Use Camera"])
        with tab_upload:
            uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if uploaded is None:
                st.markdown('<div class="drop-zone"><div class="drop-icon">☁</div><div class="drop-title">Drag and drop an image here<br>or click to browse</div><div class="drop-sub">Supported formats: JPG, JPEG, PNG</div></div>', unsafe_allow_html=True)
        with tab_camera:
            captured = st.camera_input("Take a photo", label_visibility="collapsed")
            if captured is None:
                st.markdown('<div class="drop-zone"><div class="drop-icon">📷</div><div class="drop-title">Capture a waste image with your camera</div><div class="drop-sub">Allow camera access when prompted</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        source = captured if captured is not None else uploaded

        with st.expander("⚙ Detection settings"):
            accept_conf = st.slider("Acceptance confidence", 0.20, 0.85, 0.35, 0.05)
            iou_thresh = st.slider("NMS IoU threshold", 0.20, 0.70, 0.45, 0.05)
            st.caption(f"Model: `{os.path.relpath(model_path, os.path.dirname(os.path.abspath(__file__)))} | Classes: {', '.join(x.title() for x in TARGET_CLASSES)}`")

        detect_clicked = st.button("🔎  Detect Waste", type="primary", use_container_width=True)

    with result_col:
        st.markdown("""
        <div class="panel">
          <div class="panel-head"><div><span class="step">2</span><span class="panel-title">Detection Result</span><div class="panel-copy">Detected objects with category and confidence</div></div><div class="detect-pill">✓ Ready</div></div>
        """, unsafe_allow_html=True)

        if source is None:
            st.markdown('<div class="image-card"><div class="image-label">🖼 Original Image</div><div class="drop-zone" style="min-height:245px"><div class="drop-icon">♻</div><div class="drop-title">Your detection result will appear here</div><div class="drop-sub">Upload an image or capture one to begin</div></div></div>', unsafe_allow_html=True)
        else:
            try:
                image = Image.open(source).convert("RGB")
                if detect_clicked:
                    with st.spinner("Analyzing image with YOLOv8…"):
                        image_rgb, detections = run_inference(np.array(image), model=model, infer_conf=max(0.10, min(accept_conf - 0.10, 0.30)), accept_conf=accept_conf, iou=iou_thresh)
                        annotated = annotate_image(image_rgb, detections, CLASS_COLORS)
                    st.session_state["last_detection"] = (image_rgb, annotated, detections)
                last = st.session_state.get("last_detection")
                if last is not None:
                    image_rgb, annotated, detections = last
                    st.markdown('<div class="image-label">Original Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Annotated Result</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(image_rgb, use_container_width=True)
                    with c2:
                        st.image(annotated, use_container_width=True)
                else:
                    st.image(image, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read the selected image: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    if "last_detection" in st.session_state:
        image_rgb, annotated, detections = st.session_state["last_detection"]
        counts = render_summary_cards(detections)
        action1, action2, action3 = st.columns([1, 1, 1])
        with action1:
            st.download_button("⬇ Download Annotated Image", data=build_download_image(annotated), file_name="smart_waste_detection.jpg", mime="image/jpeg", use_container_width=True)
        with action2:
            st.download_button("⬇ Export Detection Data", data=pd.DataFrame([{k: d[k] for k in ["Class", "Confidence", "Status"]} for d in detections]).to_csv(index=False) if detections else "Class,Confidence,Status\n", file_name="waste_detection.csv", mime="text/csv", use_container_width=True)
        with action3:
            if st.button("↻ Run Another Detection", use_container_width=True):
                st.session_state.pop("last_detection", None)
                st.rerun()

        with st.expander("📋 Detection details"):
            rows = [{"#": i, "Class": d["Class"].title(), "Confidence": d["Confidence"], "Status": d["Status"]} for i, d in enumerate(detections, 1)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        known_classes = [name for name in TARGET_CLASSES if counts.get(name, 0) > 0]
        if known_classes:
            st.markdown('<div class="section-title">♻ Segregation Guide</div>', unsafe_allow_html=True)
            guide_cols = st.columns(min(3, len(known_classes)))
            for i, class_name in enumerate(known_classes):
                info = get_waste_info(class_name)
                with guide_cols[i % len(guide_cols)]:
                    st.info(f"**{class_name.title()}** — {info['disposal']}")

    st.markdown('<div class="footer"><span>Smart Waste Segregation &nbsp;|&nbsp; YOLOv8 · Computer Vision · Streamlit</span><span>Reduce ♻ &nbsp; Reuse ♻ &nbsp; Recycle</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
