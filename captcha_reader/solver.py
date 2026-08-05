"""CNN based solver for the 5 digit numeric CAPTCHA.

It loads a small Keras CNN that was trained on single digits (0-9) and uses it
to read every digit produced by ``segmentation.split_digits``. The whole
CAPTCHA is only considered solved when all 5 digits are read; otherwise an
empty string is returned so the caller can retry with a fresh CAPTCHA.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

from .segmentation import binarize, decode_image_bytes, split_digits

# Default model shipped with the repo. It was trained for the Sahra OMS
# CAPTCHA (https://identity-bbi.ephoenix.ir). See scripts/train.py to train
# your own model for a different site.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "digit_cnn_light.keras",
)

# The single-digit input size the CNN expects.
DIGIT_SIZE = 28


class CaptchaSolver:
    """Reads a numeric CAPTCHA image and returns the digits as a string."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)

    def _preprocess_digit(self, digit_img):
        """Resize one digit to the CNN input size and scale pixels to [0, 1]."""
        resized = cv2.resize(
            digit_img, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA
        )
        resized = resized.astype("float32") / 255.0
        return resized.reshape(1, DIGIT_SIZE, DIGIT_SIZE, 1)

    def solve_thresh(self, thresh_img):
        """Solve an already binarized CAPTCHA. Returns "" if segmentation failed."""
        digits = split_digits(thresh_img)
        if digits is None:
            return ""
        prediction = ""
        for digit_img in digits:
            x = self._preprocess_digit(digit_img)
            pred = self.model.predict(x, verbose=0)
            prediction += str(int(np.argmax(pred)))
        return prediction

    def solve_gray(self, gray_img):
        """Binarize a grayscale CAPTCHA and solve it."""
        return self.solve_thresh(binarize(gray_img))

    def solve_bytes(self, image_bytes):
        """Decode raw image bytes (e.g. from the API) and solve them."""
        return self.solve_gray(decode_image_bytes(image_bytes))
