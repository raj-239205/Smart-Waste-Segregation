"""Download/prepare the waste dataset used by this project.

The supplied source dataset has six classes:
    0 BIODEGRADABLE -> organic
    1 CARDBOARD     -> paper
    2 GLASS         -> glass
    3 METAL         -> metal
    4 PAPER         -> paper
    5 PLASTIC       -> plastic

This project intentionally trains five classes, so the script converts the
source labels into the project's class order defined in data.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

SOURCE_CLASSES = {
    0: "organic",      # BIODEGRADABLE
    1: "paper",        # CARDBOARD
    2: "glass",
    3: "metal",
    4: "paper",
    5: "plastic",
}

TARGET_IDS = {
    "plastic": 0,
    "paper": 1,
    "metal": 2,
    "glass": 3,
    "organic": 4,
}

SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def find_dataset_root(extracted_root: Path) -> Path:
    """Find the source folder containing train/valid/test YOLO directories."""
    candidates = [extracted_root] + [p for p in extracted_root.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "valid", "test")):
            return candidate
    raise FileNotFoundError("Could not find train/valid/test folders in the ZIP.")


def convert_dataset(source_root: Path, output_root: Path) -> None:
    """Convert the six-class source dataset into this project's five classes."""
    if output_root.exists():
        shutil.rmtree(output_root)

    for source_split, target_split in SPLIT_MAP.items():
        source_images = source_root / source_split / "images"
        source_labels = source_root / source_split / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            continue

        target_images = output_root / "images" / target_split
        target_labels = output_root / "labels" / target_split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in source_images.iterdir():
            if image_path.is_file():
                shutil.copy2(image_path, target_images / image_path.name)

        for label_path in source_labels.glob("*.txt"):
            output_lines = []
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.split()
                if len(parts) < 5:
                    continue
                source_id = int(parts[0])
                class_name = SOURCE_CLASSES.get(source_id)
                if class_name is None:
                    continue
                target_id = TARGET_IDS[class_name]
                output_lines.append(" ".join([str(target_id), *parts[1:]]))
            (target_labels / label_path.name).write_text(
                "\n".join(output_lines) + ("\n" if output_lines else ""),
                encoding="utf-8",
            )

    print(f"Prepared dataset at: {output_root.resolve()}")
    print("Classes: plastic, paper, metal, glass, organic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Smart Waste Segregation dataset")
    parser.add_argument(
        "zip_file",
        nargs="?",
        help="Path to the downloaded waste datasets.zip file",
    )
    parser.add_argument(
        "--output",
        default="dataset",
        help="Output dataset directory (default: dataset)",
    )
    args = parser.parse_args()

    if not args.zip_file:
        print("Dataset source:")
        print("https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2")
        print("\nDownload the YOLOv8 ZIP, then run:")
        print("python download_dataset.py \"waste datasets.zip\"")
        print("\nThe script converts BIODEGRADABLE -> organic and CARDBOARD -> paper.")
        return

    zip_path = Path(args.zip_file).expanduser().resolve()
    output_root = Path(args.output).expanduser()
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    extraction_root = Path(".dataset_extract")
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extraction_root)

    source_root = find_dataset_root(extraction_root)
    convert_dataset(source_root, output_root)
    shutil.rmtree(extraction_root, ignore_errors=True)


if __name__ == "__main__":
    main()
