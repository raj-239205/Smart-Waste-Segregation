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
- Batch testing for a folder of images
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

YOLO-format dataset with five classes:

```text
0 -> plastic
1 -> paper
2 -> metal
3 -> glass
4 -> organic
```

The dataset is not committed to the repository. See `download_dataset.py` for setup instructions.

**Important:** the project ZIP used during the code review contained the dataset directory structure but no train/validation/test image or label files. Therefore, no new training run or honest accuracy/mAP claim is made from that ZIP alone.

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

Batch-test a folder of images:

```bash
python detect.py assets/sample_images --batch-output runs/waste_tests
```

The batch mode uses the same model loading, confidence acceptance, duplicate filtering and annotation pipeline as single-image detection. It prints total detections, accepted detections, review/unknown detections and acceptance rate, and saves annotated outputs.

## Training

Place the dataset in the structure defined by `data.yaml`, then run:

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

Ultralytics provides validation through `model.val()` and exposes metrics such as `map50`, `map` (mAP@50-95), per-class mAP and confusion-matrix information. See the official validation documentation for the exact API. 

Because the supplied project ZIP did not contain labeled dataset images, those metrics cannot be recomputed from the supplied files without first restoring the dataset.

## Project structure

```text
Smart-Waste-Segregation/
├── app.py                       # Streamlit UI + inference pipeline
├── detect.py                    # Standalone image/webcam/batch inference
├── train.py                     # Training + validation
├── data.yaml                    # Dataset configuration
├── download_dataset.py          # Dataset setup instructions
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
