"""
Smart Waste Segregation Project
Model Training Script
"""
import os
import shutil
from ultralytics import YOLO

def main():
    print("Starting YOLOv8 training for Waste Segregation...")
    print("Dataset config: data.yaml | Epochs: 50 | Batch: 16")
    
    # Load the pretrained model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="waste_seg",
        project="runs/detect"
    )
    
    print("\nTraining completed!")
    
    # Path where best model is saved
    best_model_path = os.path.join("runs", "detect", "waste_seg", "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")
    
    # Copy best model to models/best.pt
    os.makedirs("model", exist_ok=True)
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, "model/best.pt")
        print("Copied best.pt to model/best.pt")
    else:
        print("Warning: best.pt not found to copy.")

if __name__ == "__main__":
    main()
