"""
Helper script to download the waste detection dataset.
"""

def main():
    print("=" * 60)
    print("Dataset Download Instructions")
    print("=" * 60)
    print("1. Go to https://universe.roboflow.com")
    print("2. Search for 'waste detection' or 'garbage classification'")
    print("3. Look for a dataset with 5 classes: plastic, paper, metal, glass, organic")
    print("4. Export the dataset in 'YOLOv8' format")
    print("\nAlternatively, use the roboflow API if you have an account:")
    print("Uncomment the code below and insert your details.\n")
    
    # --- Roboflow API Example ---
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="YOUR_API_KEY")
    # project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
    # version = project.version(VERSION_NUMBER)
    # dataset = version.download("yolov8")
    
    print("\nManual Folder Structure Alternative:")
    print("If creating manually, ensure your 'dataset' folder has this structure:")
    print("dataset/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  └── labels/")
    print("      ├── train/")
    print("      ├── val/")
    print("      └── test/")
    print("=" * 60)

if __name__ == "__main__":
    main()
