# Handwritten Digit Recognition Using CNN

This project trains a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset (classes 0 through 9). The code is kept compact and produces evaluation metrics and plots.

## Project contents

- `train.py` downloads MNIST through Keras, trains the CNN, evaluates it on the held-out test set, and saves the model and results.
- `predict_image.py` runs the saved model on a user-provided image containing one digit.
- `app.py` opens a desktop window for choosing an image and viewing the prediction.
- `evaluate_images.py` evaluates a collection of labeled one-digit images and saves every prediction to CSV.
- `requirements.txt` lists the Python packages.
- `models/digit_cnn.keras` is created after training.
- `results/` contains test metrics, a training history plot, and a confusion matrix.

## Setup and run

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install the requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

The first run needs an internet connection so Keras can download MNIST. Training time depends on your computer. The script uses a fixed random seed, holds out 10% of the training examples for validation, and uses early stopping. The separate MNIST test set is used only for final evaluation.

After training, try an image of one digit (preferably dark ink on a plain light background):

```powershell
python predict_image.py path\to\your_digit.png
```

You can also launch the desktop demo:

```powershell
python .\app.py
```

Choose an image, then select **Predict Digit**. The window shows the predicted digit, confidence, and a low-confidence notice when the score is below 60%.

The script adjusts contrast, filters small image noise, thresholds and crops the digit, then centers it before prediction. Use one digit per image; it is not designed for a full page or a busy background. The model was trained on MNIST, so predictions on paper photos may still be less accurate than on MNIST images.

If the predicted class confidence is below 60%, `predict_image.py` displays a low-confidence notice and suggests trying a clearer image. This is a review cue, not a calibrated probability or guarantee of correctness.

The project includes `demo/mnist_test_digit_7.png` as a sample input. Run `python predict_image.py demo/mnist_test_digit_7.png` to see the prediction workflow. This sample is from the MNIST test set; use a separately written digit image for a personal handwriting demonstration.

## Evaluate your own image collection

Create a folder such as `custom_test_images` and place each image inside a subfolder named for its correct digit. Each image should contain one digit. For example:

```text
custom_test_images/
  0/zero_a.jpg
  0/zero_b.png
  1/one_a.jpg
  2/two_a.png
  ...
  9/nine_a.jpg
```

Run the evaluation from the project folder:

```powershell
python .\evaluate_images.py .\custom_test_images
```

The terminal shows overall and per-digit accuracy. Detailed predictions, confidence values, and correctness are saved to `results/custom_image_evaluation.csv`. Use images not previously used to tune the model or its preprocessing for a fairer evaluation. Report this score separately from MNIST test accuracy because it measures performance only on your image collection.

## Model design

The model takes a 28x28 grayscale image with one channel. Two convolution and max-pooling blocks learn visual features. The feature maps are flattened, passed through a 128-unit dense layer with dropout, and classified by a 10-unit softmax output layer. Pixels are scaled from 0–255 to 0–1. The model uses Adam and sparse categorical cross-entropy because each label is an integer from 0 to 9.

## Reviewing the results

Review test accuracy and loss in `results/metrics.json`, the learning curves in `results/training_history.png`, and class-level performance in `results/confusion_matrix.png`.

## Limitations

MNIST contains small, centered, grayscale images. Performance on photos or handwritten digits with different backgrounds, colors, sizes, or writing styles may be lower. This is a classification demonstration, not a general-purpose handwriting reader.
