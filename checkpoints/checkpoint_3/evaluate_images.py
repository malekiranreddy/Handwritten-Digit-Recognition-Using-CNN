"""Evaluate labeled single-digit images and save predictions to a CSV file."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf

from predict_image import MODEL_PATH, prepare_image

tf.get_logger().setLevel(logging.ERROR)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_folder",
        type=Path,
        help="Folder containing subfolders named 0 through 9 with labeled images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "custom_image_evaluation.csv",
        help="CSV output path (default: results/custom_image_evaluation.csv)",
    )
    args = parser.parse_args()

    image_folder = args.image_folder.resolve()
    if not image_folder.is_dir():
        raise SystemExit(f"Image folder not found: {image_folder}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    labeled_images = []
    for path in sorted(image_folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            actual_digit = int(path.parent.name)
        except ValueError:
            raise SystemExit(
                "Images must be placed in subfolders named 0 through 9. "
                f"Invalid folder: {path.parent}"
            )
        if actual_digit not in range(10):
            raise SystemExit(f"Folder name must be 0 through 9: {path.parent}")
        labeled_images.append((path, actual_digit))

    if not labeled_images:
        raise SystemExit(
            f"No supported images found under {image_folder}. "
            "Add images inside subfolders named 0 through 9."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    rows = []
    class_counts = defaultdict(lambda: [0, 0])
    for path, actual_digit in labeled_images:
        try:
            probabilities = model.predict(prepare_image(path), verbose=0)[0]
        except Exception as exc:
            raise SystemExit(f"Could not process {path}: {exc}") from exc

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        correct = predicted_digit == actual_digit
        class_counts[actual_digit][0] += int(correct)
        class_counts[actual_digit][1] += 1
        rows.append(
            {
                "image": str(path.relative_to(image_folder)),
                "actual_digit": actual_digit,
                "predicted_digit": predicted_digit,
                "confidence": f"{confidence:.6f}",
                "correct": correct,
            }
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "actual_digit", "predicted_digit", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    correct_total = sum(correct for correct, _ in class_counts.values())
    total = len(rows)
    print(f"Correct: {correct_total}/{total} ({correct_total / total:.1%})")
    print("Per-digit results:")
    for digit in sorted(class_counts):
        correct, count = class_counts[digit]
        print(f"  {digit}: {correct}/{count} correct")
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
