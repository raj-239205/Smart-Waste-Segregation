# Smart Waste Segregation using Computer Vision

## Overview
This project uses computer vision to automatically detect and classify different types of waste into five categories: plastic, paper, metal, glass, and organic. It uses a fine-tuned YOLOv8 object detection model to draw bounding boxes around waste items in images or real-time video streams. The system includes both a Streamlit web application for image uploads and a standalone script for webcam detection.

## Objective
This project was developed to automate the waste segregation process using deep learning techniques. The goal is to build an efficient, edge-deployable system that can assist in identifying recyclable and compostable materials to improve waste management workflows.

## Technologies Used
- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Jupyter Notebook

## Dataset
- **Source:** Roboflow Universe
- **Format:** YOLOv8 (images + YOLO-format .txt labels)
- **Classes:** plastic, paper, metal, glass, organic
- **Split:** ~1287 train, ~215 validation images
- **Note:** Dataset not included in repository. See `download_dataset.py` for instructions.

## Model
- **Base:** YOLOv8n (nano) pretrained on COCO
- **Fine-tuned:** On custom waste detection dataset
- **Training:** 50 epochs, 640x640 image size, batch size 16, AdamW optimizer

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Smart-Waste-Segregation

# Install Python 3.11 (if not installed)
# Download from https://www.python.org/downloads/

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup
To download and prepare the dataset for training, refer to the instructions inside the `download_dataset.py` script.

## Training the Model
```bash
python train.py
```
*Note: Training requires a GPU (e.g., NVIDIA GeForce RTX 3050) and takes approximately 30-60 minutes.*

## Running the Application

```bash
# Streamlit web app
streamlit run app.py

# Webcam real-time detection
python detect.py

# Single image detection
python detect.py path/to/image.jpg
```

## How It Works
1. User uploads an image via the web app or provides a camera feed.
2. The image is preprocessed and passed to the YOLOv8 model.
3. The model detects waste objects and draws bounding boxes around them.
4. Each detected object is classified into one of the 5 categories.
5. The results are displayed with confidence scores and segregation information.

## Project Structure
```text
Smart-Waste-Segregation/
├── app.py                 # Streamlit web application
├── detect.py              # Standalone detection (webcam / image)
├── train.py               # Model training script
├── data.yaml              # YOLOv8 dataset configuration
├── download_dataset.py    # Dataset download instructions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── model/
│   └── best.pt            # Trained model weights (after training)
├── utils/
│   └── waste_info.py      # Waste category and disposal info
├── dataset/               # Dataset directory (not in repo)
│   ├── images/
│   └── labels/
└── runs/                  # Training output (auto-generated)
```

## Results
*Results will be available after training. Training metrics are saved in `runs/detect/waste_seg/`*

## Limitations
- Limited to 5 specific waste categories.
- Detection performance heavily depends on image quality, angle, and lighting conditions.
- May struggle to detect very small or partially hidden objects in cluttered environments.
- Trained on a relatively small dataset (~1500 images).

## Future Scope
- Expand the model to recognize more waste categories (e-waste, medical waste).
- Improve model robustness by training on a larger, more diverse dataset.
- Deploy the system as a mobile application.
- Integrate with IoT systems for smart waste bin applications.

## Author
Developed as part of B.Tech Industrial Training.
