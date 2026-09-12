# ♻️ Smart Waste Segregation using Computer Vision

An AI-powered waste detection application built with **YOLOv8, OpenCV and Streamlit**. The system identifies five supported waste categories and provides practical segregation guidance.

## Features

- Image upload and camera capture through Streamlit
- YOLOv8 detection with configurable confidence and IoU thresholds
- Five categories: Plastic, Paper, Metal, Glass, Organic
- Unknown / Review state for low-confidence predictions
- Geometry-based duplicate/sub-box suppression
- Detection statistics and category counts
- Annotated-image download
- Standalone real-time webcam detection
- Reproducible training configuration with validation metrics and plots
- Category-specific disposal guidance

## How it works

```text
Image / Camera -> YOLOv8 -> Candidate detections -> Confidence acceptance
             -> Duplicate filtering -> Known category / Unknown Review
             -> Annotated result + statistics + segregation guidance
```

### Unknown / Review

Unknown is an uncertainty rejection state, not a separately trained sixth class. A prediction is sent to review when its confidence is below the configured acceptance threshold or its class is outside the supported categories. A true open-set unknown detector would require dedicated unknown/open-set training and evaluation.

## Technologies

- Python 3.11
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Streamlit
- Pillow
- Matplotlib

## Dataset

The project uses the **Garbage Classification** YOLO dataset from Roboflow Universe:

urlGarbage Classification datasethttps://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2

The downloaded source dataset contains 10,464 labeled images across train/validation/test splits and six source classes:

```text
BIODEGRADABLE
CARDBOARD
GLASS
METAL
PAPER
PLASTIC
```

This project intentionally uses five classes. `download_dataset.py` converts the source dataset as follows:

```text
BIODEGRADABLE -> organic
CARDBOARD     -> paper
GLASS         -> glass
METAL         -> metal
PAPER         -> paper
PLASTIC       -> plastic
```

The resulting project class IDs are:

```text
0 -> plastic
1 -> paper
2 -> metal
3 -> glass
4 -> organic
```

### Prepare the dataset

After downloading the ZIP, place it beside the project and run:

```bash
python download_dataset.py "waste datasets.zip"
```

The script extracts the source dataset, converts the six source classes to the five project classes, creates the expected `dataset/images/...` and `dataset/labels/...` structure, and removes the temporary extraction directory.

**Why the 191 MB ZIP is not committed directly to GitHub:** GitHub's normal repository file interface has a 100 MB per-file limit. The supplied dataset ZIP is larger than that limit, so the repository stores the reproducible preparation workflow and exact dataset source instead of pretending the binary dataset has been committed.

## Model

- Base checkpoint: YOLOv8n pretrained on COCO
- Fine-tuned checkpoint: `model/best.pt`
- Default training: 50 epochs, 640x640, batch size 16
- Optimizer: AdamW
- Seed: 42
- Early stopping patience: 12 epochs
- Training outputs: `runs/detect/waste_seg/`

## Installation

```bash
git clone https://github.com/raj-239205/Smart-Waste-Segregation.git
cd Smart-Waste-Segregation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Streamlit application:

```bash
streamlit run app.py
```

Webcam detection:

```bash
python detect.py
```

Single image detection:

```bash
python detect.py path/to/image.jpg
```

## Training

First prepare the dataset with `download_dataset.py`, then run:

```bash
python train.py
```

The best checkpoint is copied to `model/best.pt`. Validation reports mAP@50 and mAP@50-95 when available.

## Evaluation

For a proper final evaluation, use a held-out labeled test/validation set and report:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Per-class performance
- Confusion matrix
- Validation examples
- Real-world tests across lighting, viewpoints, backgrounds and object sizes

Ultralytics provides validation through `model.val()` and exposes metrics such as `map50`, `map` (mAP@50-95), per-class mAP and confusion-matrix information.

## Project structure

```text
Smart-Waste-Segregation/
├── app.py                       # Streamlit UI + inference pipeline
├── detect.py                    # Standalone image/webcam inference
├── train.py                     # Training + validation
├── data.yaml                    # Dataset configuration
├── download_dataset.py          # Dataset preparation + class conversion
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── result.jpg                   # Example output
├── model/
│   └── best.pt                  # Production checkpoint
├── utils/
│   ├── __init__.py
│   └── waste_info.py            # Waste categories and guidance
├── assets/                      # Project assets
└── notebooks/                   # Experiments and analysis
```

## Limitations

- Only five trained waste categories are supported.
- Unknown handling is confidence-based, not a true open-set classifier.
- Performance depends on dataset diversity, lighting, viewpoint, object size and occlusion.
- The current checkpoint can produce visually plausible but semantically incorrect predictions on objects outside its training distribution; review the result before making disposal decisions.
- Disposal guidance should complement local waste-management rules.

## Future scope

- Dedicated open-set unknown detection
- More waste categories such as e-waste and hazardous waste
- Larger and more diverse datasets
- Object tracking and temporal smoothing
- Mobile/edge deployment
- Smart-bin and IoT integration

## Author

Developed as part of B.Tech Industrial Training.
