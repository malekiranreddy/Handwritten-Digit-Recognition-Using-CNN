"""Small desktop interface for predicting a digit from an image."""

from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from predict_image import LOW_CONFIDENCE_THRESHOLD, MODEL_PATH, prepare_image

tf.get_logger().setLevel(logging.ERROR)


class DigitRecognitionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("520x570")
        self.root.minsize(460, 520)

        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.model: tf.keras.Model | None = None

        container = ttk.Frame(root, padding=24)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Handwritten Digit Recognition",
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(0, 8))
        ttk.Label(
            container,
            text="Choose an image containing one digit.",
            font=("Segoe UI", 10),
        ).pack(pady=(0, 16))

        self.preview = ttk.Label(container, text="No image selected", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=8)

        controls = ttk.Frame(container)
        controls.pack(pady=12)
        ttk.Button(controls, text="Choose Image", command=self.choose_image).pack(
            side="left", padx=6
        )
        ttk.Button(controls, text="Predict Digit", command=self.predict).pack(
            side="left", padx=6
        )

        self.prediction_label = ttk.Label(
            container, text="Prediction: —", font=("Segoe UI", 16, "bold")
        )
        self.prediction_label.pack(pady=(12, 4))
        self.confidence_label = ttk.Label(container, text="Confidence: —")
        self.confidence_label.pack(pady=3)
        self.warning_label = ttk.Label(
            container, text="", foreground="#a15c00", wraplength=440, justify="center"
        )
        self.warning_label.pack(pady=6)

        ttk.Label(
            container,
            text="One digit per image. Clear, high-contrast images work best.",
            font=("Segoe UI", 9),
            wraplength=440,
            justify="center",
        ).pack(pady=(8, 0))

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a digit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self.image_path = Path(selected)
        try:
            image = Image.open(self.image_path).convert("RGB")
            image.thumbnail((360, 300), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_photo, text="")
            self.prediction_label.configure(text="Prediction: —")
            self.confidence_label.configure(text="Confidence: —")
            self.warning_label.configure(text="")
        except Exception as exc:
            self.image_path = None
            messagebox.showerror("Could not open image", str(exc), parent=self.root)

    def predict(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("Choose an image", "Choose a digit image first.", parent=self.root)
            return
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model not found",
                f"Could not find the trained model at:\n{MODEL_PATH}\n\nRun train.py first.",
                parent=self.root,
            )
            return

        try:
            if self.model is None:
                self.model = tf.keras.models.load_model(MODEL_PATH)
            probabilities = self.model.predict(prepare_image(self.image_path), verbose=0)[0]
            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit])

            self.prediction_label.configure(text=f"Prediction: {digit}")
            self.confidence_label.configure(text=f"Confidence: {confidence:.1%}")
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                self.warning_label.configure(
                    text="Low confidence: this prediction may be unreliable. Try a clearer image."
                )
            else:
                self.warning_label.configure(text="")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc), parent=self.root)


def main() -> None:
    root = tk.Tk()
    DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
