# ♻️ Smart Waste Segregation using Computer Vision

An AI-powered waste detection application built with **YOLOv8, OpenCV and Streamlit**. The system identifies five supported waste categories and provides practical segregation guidance.

## ✨ Features

- 🖼️ Image upload and camera capture through Streamlit
- 🎯 YOLOv8 object detection with configurable confidence and IoU thresholds
- ♻️ Five supported categories: **Plastic, Paper, Metal, Glass, Organic**
- ⚠️ Explicit **Unknown / Review** state for low-confidence predictions
- 🧹 Geometry-based duplicate/sub-box suppression instead of scene-specific pixel rules
- 📊 Detection count, classification count, review count and average confidence
- 📋 Downloadable annotated detection image
- 🎥 Standalone real-time webcam detector
- 🧪 Reproducible training configuration with validation metrics and plots
- 📚 Category-specific disposal and segregation guidance

## 🧠 How the system works

```text
Image / Camera
      ↓
YOLOv8 inference
      ↓
Candidate detections
      ↓
Confidence acceptance
      ↓
Duplicate / overlap filtering
      ↓
┌───────────────────────┐
│ Known waste category  │ → Plastic / Paper / Metal / Glass / Organic
│ Low-confidence        │ → Unknown / Review
└───────────────────────┘
      ↓
Annotated result + statistics + segregation guidance
```

### Important note about Unknown

`Unknown / Review` is an **uncertainty rejection state**, not a separately trained sixth class. A prediction is sent to review when its confidence is below the configured acceptance threshold or when its class name is outside the supported categories. A true open-set unknown detector would require dedicated unknown/open-set training and evaluation.

## 🛠️ Technologies

- Python 3.11
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Streamlit
- Pillow
- Matplotlib

## 📦 Dataset

The project uses a YOLO-format waste detection dataset with five classes:

```text
0 → plastic
1 → paper
2 → metal
3 → glass
4 → organic
```

The dataset itself is intentionally not committed to the repository. See `download_dataset.py` for the expected directory structure.

## 🤖 Model

- Base checkpoint: **YOLOv8n pretrained on COCO**
- Fine-tuned checkpoint: `model/best.pt`
- Default training: 50 epochs, 640×640, batch size 16
- Optimizer: AdamW
- Reproducibility seed: 42
- Early stopping patience: 12 epochs
- Training/validation plots are written to `runs/detect/waste_seg/`

## 🚀 Installation

```bash
git clone https://github.com/raj-239205/Smart-Waste-Segregation.git
cd Smart-Waste-Segregation

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Run the Streamlit application

```bash
streamlit run app.py
```

The application supports either an uploaded image or a camera capture. Use the sidebar to adjust the acceptance confidence and NMS IoU threshold.

## 🎥 Run webcam detection

```bash
python detect.py
```

Press **Q** to exit the webcam window.

## 🖼️ Detect a single image from the command line

```bash
python detect.py path/to/image.jpg
```

Optional settings:

```bash
python detect.py path/to/image.jpg --conf 0.40 --iou 0.45 --output result.jpg
```

## 🏋️ Train the model

Place the dataset in the structure expected by `data.yaml`, then run:

```bash
python train.py
```

Optional training parameters:

```bash
python train.py --epochs 50 --imgsz 640 --batch 16 --device 0
```

After training, the best checkpoint is copied automatically to:

```text
model/best.pt
```

The script also runs validation and reports mAP@50 and mAP@50–95 when available.

## 📁 Project structure

```text
Smart-Waste-Segregation/
├── app.py                       # Streamlit application
├── detect.py                    # Image + webcam inference
├── train.py                     # Training + validation
├── data.yaml                    # Dataset configuration
├── download_dataset.py          # Dataset setup instructions
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── result.jpg                   # Example output
│
├── model/
│   └── best.pt                  # Production trained checkpoint
│
├── utils/
│   ├── detection.py             # Shared inference/post-processing
│   └── waste_info.py             # Categories and segregation guidance
│
├── assets/                      # Project assets/sample images
├── notebooks/                   # Experiments and analysis
├── dataset/                     # Local dataset; not committed
└── runs/                        # Local training outputs; not committed
```

## 📈 Evaluation

For a meaningful model evaluation, report:

- Precision
- Recall
- mAP@50
- mAP@50–95
- Per-class performance
- Confusion matrix
- Validation examples

Training outputs are generated under `runs/detect/waste_seg/` and should be used when preparing the project report and presentation.

## ⚠️ Limitations

- The detector supports only five trained waste categories.
- Unknown/review handling is confidence-based and should not be described as a true open-set classifier.
- Performance depends on dataset diversity, lighting, viewpoint, object size and occlusion.
- Automatic disposal guidance should complement, not replace, local waste-management rules.
- The dataset is not included in the repository, so retraining requires obtaining the source dataset separately.

## 🔮 Future scope

- Add a dedicated unknown/open-set detection strategy
- Expand to e-waste, hazardous waste and additional recyclable materials
- Increase dataset size and environmental diversity
- Add object tracking and temporal smoothing for webcam inference
- Deploy to edge/mobile hardware
- Integrate with smart-bin or IoT systems

## 👨‍💻 Author

Developed as part of B.Tech Industrial Training.
