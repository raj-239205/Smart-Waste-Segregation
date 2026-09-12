import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
from ultralytics import YOLO
import pandas as pd
import sys

# Ensure the current directory is in sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.waste_info import get_waste_info, CLASS_COLORS, map_detected_class, filter_detections
except ImportError:
    def get_waste_info(class_name):
        return {
            "category": "Unknown", 
            "bin_color": "Unknown", 
            "disposal": "Follow local guidelines", 
            "examples": []
        }
    CLASS_COLORS = {}
    def map_detected_class(name):
        return str(name).lower()
    def filter_detections(dets, **kwargs):
        return dets

# Set page configuration
st.set_page_config(page_title='Smart Waste Segregation', page_icon='♻️', layout='wide')

@st.cache_resource
def load_model():
    """Load the YOLOv8 model only once. Checks model/best.pt first, then yolov8n.pt."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model", "garbage_yolov8s.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "yolov8n.pt")
        if not os.path.exists(model_path):
            return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_detection(image, model, conf=0.25, iou=0.45):
    """Convert PIL image to BGR numpy array and run YOLO inference with configurable threshold and NMS."""
    img_array = np.array(image.convert('RGB'))
    bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model(bgr_array, conf=conf, iou=iou, agnostic_nms=False)[0]
    return img_array, results

def draw_boxes(img_array, results, class_names):
    """Draw bounding boxes and labels on the image with sub-box suppression."""
    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    raw_detections = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        raw_name = class_names[cls_id]
        class_name = map_detected_class(raw_name)
        if class_name is None:
            continue

        raw_detections.append({
            'Class': class_name,
            'Confidence': f"{conf:.2f}",
            'conf_val': conf,
            'box': [x1, y1, x2, y2]
        })
        
    # Suppress redundant sub-part boxes while preserving all distinct items
    filtered = filter_detections(raw_detections, img_bgr=annotated_img)
    
    for det in filtered:
        x1, y1, x2, y2 = det['box']
        class_name = det['Class']
        conf_str = det['Confidence']
        
        # Get color (default to green if not defined)
        color_rgb = CLASS_COLORS.get(class_name, (0, 255, 0))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Draw label background and text
        label = f"{class_name} {conf_str}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated_img, (x1, y1 - label_height - 6), (x1 + label_width + 4, y1), color_bgr, -1)
        cv2.putText(annotated_img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
    final_table = [{'Class': d['Class'], 'Confidence': d['Confidence']} for d in filtered]
    # Convert back to RGB for Streamlit display
    return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), final_table

def main():
    st.title('Smart Waste Segregation')
    st.write('Upload an image to detect and classify waste items using YOLOv8')
    
    model = load_model()
    if model is None:
        st.error("Model file not found. Please ensure 'model/best.pt' or 'yolov8n.pt' exists.")
        st.stop()
        
    class_names = model.names
    target_classes = ['plastic', 'paper', 'metal', 'glass', 'organic']
    disp_classes = ', '.join(class_names.values()) if len(class_names) == 5 else ', '.join(target_classes)
    st.info(f"Model loaded successfully. Supported waste categories: {disp_classes}")
    
    with st.expander("⚙️ Detection Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.85, value=0.25, step=0.05,
                                    help="Minimum prediction confidence required to display an object.")
        with c2:
            iou_thresh = st.slider("NMS Overlap Threshold", min_value=0.20, max_value=0.70, value=0.45, step=0.05,
                                    help="Intersection-over-Union threshold to suppress duplicate overlapping boxes.")
            
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button('Detect Waste'):
                with st.spinner('Running detection...'):
                    img_array, results = run_detection(image, model, conf=conf_thresh, iou=iou_thresh)
                    annotated_img, detections = draw_boxes(img_array, results, class_names)
                    
                    if len(detections) == 0:
                        st.warning('No confident waste items detected in the image. Try lowering the confidence threshold or uploading a clearer image.')
                    else:
                        st.success(f"Detected {len(detections)} waste item(s).")
                        
                        st.write("### Detection Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(annotated_img, caption='Annotated Image', use_container_width=True)
                            
                            # Prepare annotated image for download
                            result_img = Image.fromarray(annotated_img)
                            buf = io.BytesIO()
                            result_img.save(buf, format="JPEG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="Download Annotated Image",
                                data=byte_im,
                                file_name="detected_waste.jpg",
                                mime="image/jpeg"
                            )
                            
                        with col2:
                            # Show detection table
                            df = pd.DataFrame(detections)
                            df.index += 1  # 1-based indexing
                            st.table(df)
                            
                            # Show segregation information for unique detected classes
                            unique_classes = df['Class'].unique()
                            st.write("#### Segregation Guide")
                            for cls in unique_classes:
                                info = get_waste_info(cls)
                                if info:
                                    with st.expander(f"How to dispose: {cls.title()}"):
                                        st.write(f"**Category:** {info.get('category', 'N/A')}")
                                        st.write(f"**Bin Color:** {info.get('bin_color', 'N/A')}")
                                        st.write(f"**Disposal Method:** {info.get('disposal', 'N/A')}")
                                        if info.get('examples'):
                                            st.write(f"**Examples:** {', '.join(info.get('examples', []))}")
                                            
                st.write('Upload a new image above to try again.')
                
        except Exception as e:
            st.error(f"Error processing the image: {e}")

    st.divider()
    st.caption('Smart Waste Segregation using Computer Vision | B.Tech Project')

if __name__ == '__main__':
    main()
