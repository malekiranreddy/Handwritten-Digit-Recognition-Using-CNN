"""Prepare external digit images in the format expected by the CNN."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def otsu_threshold(pixels: np.ndarray) -> int:
    """Find a foreground/background threshold from an image histogram."""
    histogram = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    values = np.arange(256, dtype=np.float64)
    weight_left = np.cumsum(probabilities)
    mean_left = np.cumsum(probabilities * values)
    mean_total = mean_left[-1]
    denominator = weight_left * (1.0 - weight_left)
    scores = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    scores[valid] = (
        (mean_total * weight_left[valid] - mean_left[valid]) ** 2 / denominator[valid]
    )
    return int(np.argmax(scores))


def center_of_mass(canvas: np.ndarray) -> np.ndarray:
    """Shift foreground toward the center, as MNIST digits are centered."""
    mass = float(canvas.sum())
    if mass == 0:
        return canvas
    rows, columns = np.indices(canvas.shape)
    center_y = float((rows * canvas).sum() / mass)
    center_x = float((columns * canvas).sum() / mass)
    shift_y = int(round(13.5 - center_y))
    shift_x = int(round(13.5 - center_x))

    shifted = np.zeros_like(canvas)
    source_y0 = max(0, -shift_y)
    source_y1 = min(28, 28 - shift_y)
    source_x0 = max(0, -shift_x)
    source_x1 = min(28, 28 - shift_x)
    target_y0 = max(0, shift_y)
    target_x0 = max(0, shift_x)
    target_y1 = target_y0 + source_y1 - source_y0
    target_x1 = target_x0 + source_x1 - source_x0
    shifted[target_y0:target_y1, target_x0:target_x1] = canvas[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def prepare_image(image_path: Path) -> np.ndarray:
    """Normalize one digit image into a centered 28x28 MNIST-like tensor."""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("L")
    image = ImageOps.autocontrast(image, cutoff=1).filter(ImageFilter.MedianFilter(size=3))
    pixels = np.asarray(image, dtype=np.uint8)

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    light_background = float(np.median(border)) >= 127
    threshold = otsu_threshold(pixels)
    foreground = pixels < threshold if light_background else pixels > threshold

    rows, columns = np.where(foreground)
    if len(rows) == 0:
        raise ValueError("No digit strokes found. Use a clear, high-contrast image of one digit.")

    pixels_float = pixels.astype("float32")
    if light_background:
        stroke = np.clip((threshold - pixels_float) / max(threshold, 1) * 255.0, 0, 255)
    else:
        stroke = np.clip((pixels_float - threshold) / max(255 - threshold, 1) * 255.0, 0, 255)
    stroke[~foreground] = 0

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    cropped = Image.fromarray(stroke[y0:y1, x0:x1].astype("uint8"))
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas_image = Image.new("L", (28, 28), color=0)
    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2
    canvas_image.paste(cropped, (x, y))
    array = np.asarray(canvas_image, dtype="float32") / 255.0
    array = center_of_mass(array)
    return array[np.newaxis, ..., np.newaxis]
