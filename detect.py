"""
Smart Waste Segregation - Detection Script
Supports image and webcam inference
"""
import sys
import os
import time
import cv2
from ultralytics import YOLO

# Import shared colors and mapping; convert RGB to BGR for OpenCV
try:
    from utils.waste_info import CLASS_COLORS as _RGB_COLORS, map_detected_class, filter_detections
    CLASS_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in _RGB_COLORS.items()}
except ImportError:
    CLASS_COLORS_BGR = {
        'plastic': (0, 200, 0), 'paper': (255, 100, 0),
        'metal': (50, 50, 255), 'glass': (200, 0, 200),
        'organic': (0, 200, 255)
    }
    def map_detected_class(name):
        return str(name).lower()
    def filter_detections(dets, **kwargs):
        return dets

def load_model(model_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, "model", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model", "garbage_yolov8s.pt")
    if not os.path.exists(model_path):
        if os.path.exists(os.path.join(base_dir, "yolov8n.pt")):
            model_path = os.path.join(base_dir, "yolov8n.pt")
        else:
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
    return YOLO(model_path)

def draw_boxes(frame, results, class_colors, class_names):
    raw_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            raw_name = class_names[cls_id]
            cls_name = map_detected_class(raw_name)
            if cls_name is None:
                continue
                
            raw_boxes.append({
                'Class': cls_name,
                'Confidence': f"{conf:.2f}",
                'conf_val': conf,
                'box': [x1, y1, x2, y2]
            })
            
    filtered = filter_detections(raw_boxes, img_bgr=frame)
    for det in filtered:
        x1, y1, x2, y2 = det['box']
        cls_name = det['Class']
        conf_str = det['Confidence']
        color = class_colors.get(cls_name, (255, 255, 255))
        
        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf_str}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 6), (x1 + label_width + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame

def run_image(model, image_path):
    print(f"Running inference on {image_path}...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
        
    class_names = model.names
    
    results = model(frame, conf=0.25, iou=0.45, agnostic_nms=False)
    annotated_frame = draw_boxes(frame, results, CLASS_COLORS_BGR, class_names)
    
    cv2.imwrite('result.jpg', annotated_frame)
    print("Result saved as 'result.jpg'.")
    
    try:
        cv2.imshow('Detection Result', annotated_frame)
        print("Press any key in the window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Could not display window ({e}). 'result.jpg' was saved successfully.")

def run_webcam(model):
    print("Starting webcam detection... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    class_names = model.names
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
            
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        results = model(frame, conf=0.30, iou=0.45, agnostic_nms=False, verbose=False)
        annotated_frame = draw_boxes(frame, results, CLASS_COLORS_BGR, class_names)
        
        # Display FPS and instructions
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press q to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        cv2.imshow('Smart Waste Segregation - Real Time Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model()
    
    if len(sys.argv) > 1 and sys.argv[1] != 'webcam':
        image_path = sys.argv[1]
        run_image(model, image_path)
    else:
        run_webcam(model)
