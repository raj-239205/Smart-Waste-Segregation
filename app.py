"""Streamlit interface for Smart Waste Segregation."""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.detection import TARGET_CLASSES, annotate_image, load_model, run_inference
from utils.waste_info import CLASS_COLORS, get_waste_info


st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    [data-testid="stHeader"] { background: rgba(246,248,251,0.92); }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #164e63 55%, #166534 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 16px 40px rgba(15, 23, 42, .16);
    }
    .hero h1 { margin: 0; font-size: 2.45rem; letter-spacing: -1px; }
    .hero p { margin: .55rem 0 0; opacity: .86; font-size: 1.03rem; }
    .section-title { font-size: 1.25rem; font-weight: 700; margin: .5rem 0 .7rem; }
    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background: white;
        box-shadow: 0 6px 18px rgba(15, 23, 42, .05);
        margin-bottom: .7rem;
    }
    .info-card h4 { margin: 0 0 .35rem; }
    .muted { color: #64748b; font-size: .9rem; }
    .stButton > button { border-radius: 12px; font-weight: 700; min-height: 2.7rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; padding: .8rem; border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    unknown = [d for d in detections if d["Class"] == "unknown"]
    total = len(detections)
    avg_conf = sum(d["conf_val"] for d in known) / len(known) if known else 0

    cols = st.columns(4)
    cols[0].metric("Objects", total)
    cols[1].metric("Classified", len(known))
    cols[2].metric("Needs review", len(unknown))
    cols[3].metric("Avg. confidence", f"{avg_conf:.0%}" if known else "—")

    counts = {name: sum(d["Class"] == name for d in detections) for name in TARGET_CLASSES}
    counts["unknown"] = len(unknown)
    return counts


def render_waste_guide(classes):
    if not classes:
        return
    st.markdown('<div class="section-title">♻️ Segregation guide</div>', unsafe_allow_html=True)
    columns = st.columns(min(3, len(classes)))
    for index, class_name in enumerate(classes):
        info = get_waste_info(class_name)
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>{class_name.title()}</h4>
                    <div class="muted">{info['category']} · Suggested bin: {info['bin_color']}</div>
                    <p><b>Examples:</b> {', '.join(info['examples']) if info['examples'] else '—'}</p>
                    <p><b>Guidance:</b> {info['disposal']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>♻️ Smart Waste Segregation</h1>
            <p>AI-powered computer vision for practical waste classification and segregation guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, model_path = get_model()
    if model is None:
        st.error(
            "No usable YOLO model was found. Place the trained model at `model/best.pt` "
            "and install the dependencies from `requirements.txt`."
        )
        st.stop()

    with st.sidebar:
        st.header("⚙️ Detection settings")
        accept_conf = st.slider(
            "Acceptance confidence",
            min_value=0.20,
            max_value=0.85,
            value=0.35,
            step=0.05,
            help="Predictions below this score are retained as Unknown/Review instead of being presented as a confident class.",
        )
        iou_thresh = st.slider(
            "NMS IoU threshold",
            min_value=0.20,
            max_value=0.70,
            value=0.45,
            step=0.05,
            help="Controls how strongly overlapping detections are suppressed by YOLO.",
        )
        st.divider()
        st.caption("Model")
        st.code(os.path.relpath(model_path, os.path.dirname(os.path.abspath(__file__))), language="text")
        st.caption("Supported classes")
        st.write(" · ".join(name.title() for name in TARGET_CLASSES))
        with st.expander("About Unknown / Review"):
            st.write(
                "Unknown is an uncertainty/rejection state. It is used when the model's "
                "confidence is below the acceptance threshold or its class name is not "
                "one of the supported waste categories. It is not a separately trained sixth class."
            )

    st.markdown('<div class="section-title">📷 Choose an input</div>', unsafe_allow_html=True)
    upload_col, camera_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    with camera_col:
        captured = st.camera_input("Take a photo")

    source = captured if captured is not None else uploaded
    if source is None:
        st.info("Upload a waste image or take a photo to start detection.")
        st.markdown(
            """
            <div class="info-card">
                <h4>How it works</h4>
                <p>1. Provide an image → 2. YOLOv8 detects candidate objects → 3. Low-confidence predictions are flagged for review → 4. The app shows categories, confidence, and disposal guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        image = Image.open(source).convert("RGB")
        st.image(image, caption="Input image", width="stretch")
    except Exception as exc:
        st.error(f"Could not read the selected image: {exc}")
        return

    if st.button("🔎 Detect Waste", type="primary", use_container_width=True):
        with st.spinner("Analyzing image with YOLOv8…"):
            image_rgb, detections = run_inference(
                image_rgb=__import__("numpy").array(image),
                model=model,
                infer_conf=max(0.10, min(accept_conf - 0.10, 0.30)),
                accept_conf=accept_conf,
                iou=iou_thresh,
            )
            annotated = annotate_image(image_rgb, detections, CLASS_COLORS)

        counts = render_summary(detections)
        st.divider()

        result_col, detail_col = st.columns([1.35, 1])
        with result_col:
            st.markdown('<div class="section-title">🖼️ Detection result</div>', unsafe_allow_html=True)
            st.image(annotated, width="stretch")
            st.download_button(
                "⬇️ Download annotated image",
                data=build_download_image(annotated),
                file_name="smart_waste_detection.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        with detail_col:
            st.markdown('<div class="section-title">📊 Detection details</div>', unsafe_allow_html=True)
            if detections:
                rows = [
                    {
                        "#": index,
                        "Class": d["Class"].title(),
                        "Confidence": d["Confidence"],
                        "Status": d["Status"],
                    }
                    for index, d in enumerate(detections, start=1)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.warning("No candidate objects were detected. Try a clearer image or lower the acceptance confidence.")

            st.markdown('<div class="section-title">📦 Category counts</div>', unsafe_allow_html=True)
            count_rows = [
                {"Category": name.title(), "Count": count}
                for name, count in counts.items()
                if count > 0
            ]
            if count_rows:
                st.dataframe(pd.DataFrame(count_rows), hide_index=True, use_container_width=True)

        known_classes = [name for name in TARGET_CLASSES if counts.get(name, 0) > 0]
        render_waste_guide(known_classes)

        if counts.get("unknown", 0):
            st.warning(
                f"{counts['unknown']} detection(s) need manual review. "
                "Do not make an automatic disposal decision from an Unknown result."
            )

    st.divider()
    st.caption("Smart Waste Segregation · YOLOv8 · Computer Vision · B.Tech Project")


if __name__ == "__main__":
    main()
