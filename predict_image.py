"""Command-line entry point for predicting one digit image."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf

from digit_recognition.model import LOW_CONFIDENCE_THRESHOLD, MODEL_PATH
from digit_recognition.preprocessing import prepare_image

tf.get_logger().setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image containing one digit")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(prepare_image(args.image), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.1%}")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        print("Low confidence: this prediction may be unreliable. Try a clearer image.")


if __name__ == "__main__":
    main()
