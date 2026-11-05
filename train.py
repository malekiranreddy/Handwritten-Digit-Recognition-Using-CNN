"""Train and evaluate a CNN on the MNIST handwritten digit dataset."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from digit_recognition.model import MODEL_PATH, build_model

tf.get_logger().setLevel(logging.ERROR)

MODEL_DIR = MODEL_PATH.parent
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_training_plot(history: tf.keras.callbacks.History) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="MNIST Test Set Confusion Matrix", xlabel="Predicted digit", ylabel="True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    set_seed()
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "classification_report": report,
                "epochs_completed": len(history.history["loss"]),
            },
            file,
            indent=2,
        )
    save_training_plot(history)
    save_confusion_matrix(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved plots and metrics to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
