"""CNN architecture and shared model settings."""

from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "digit_cnn.keras"
LOW_CONFIDENCE_THRESHOLD = 0.60


def build_model() -> tf.keras.Model:
    """Create a compact CNN for 28x28 grayscale digit images."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="handwritten_digit_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
