"""Digit segmentation for the numeric CAPTCHA.

The CAPTCHA is a single image that contains exactly 5 digits. This module
turns that image into 5 cropped digit images, ordered left to right, so they
can be labeled (during training) or read by the CNN (during solving).

Steps:
  1. Binarize the image with Otsu thresholding (white digits on black).
  2. Find the external contours.
  3. Keep only boxes big enough to be a digit (drop small noise).
  4. Sort the boxes left to right.

If the result is not exactly 5 boxes the segmentation is treated as failed,
because we cannot trust which digit is which.
"""

import cv2
import numpy as np

# A box smaller than this is considered noise, not a digit.
MIN_DIGIT_WIDTH = 5
MIN_DIGIT_HEIGHT = 10

# How many digits every CAPTCHA is expected to contain.
EXPECTED_DIGITS = 5


def binarize(gray_img):
    """Return a black/white image: white digits (255) on a black (0) background."""
    _, thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


def find_digit_boxes(thresh_img):
    """Return digit bounding boxes (x, y, w, h), sorted left to right."""
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(c) for c in contours]
    # Drop boxes that are too small to be a real digit.
    boxes = [b for b in boxes if b[2] > MIN_DIGIT_WIDTH and b[3] > MIN_DIGIT_HEIGHT]
    boxes.sort(key=lambda b: b[0])  # left to right
    return boxes


def split_digits(thresh_img):
    """Split a binarized CAPTCHA into 5 digit crops.

    Returns a list of 5 grayscale digit images (left to right), or None if the
    image did not segment into exactly 5 digits.
    """
    boxes = find_digit_boxes(thresh_img)
    if len(boxes) != EXPECTED_DIGITS:
        return None
    return [thresh_img[y:y + h, x:x + w] for (x, y, w, h) in boxes]


def decode_image_bytes(image_bytes):
    """Decode raw image bytes (e.g. a PNG from the API) to a grayscale image."""
    array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
