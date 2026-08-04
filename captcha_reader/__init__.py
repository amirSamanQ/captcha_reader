"""captcha_reader - read a 5 digit numeric CAPTCHA and log in to a Sahra OMS.

Public API:
    CaptchaSolver     - CNN based solver (image -> "12345")
    login             - log in to a Sahra OMS (ephoenix identity) account
    login_futures     - log in to the exphoenixfuture OMS variant
    split_digits      - segment a binarized CAPTCHA into 5 digit crops
    binarize          - Otsu threshold a grayscale CAPTCHA
"""

from .segmentation import binarize, find_digit_boxes, split_digits
from .solver import CaptchaSolver
from .login import login, login_futures

__all__ = [
    "CaptchaSolver",
    "login",
    "login_futures",
    "split_digits",
    "find_digit_boxes",
    "binarize",
]
