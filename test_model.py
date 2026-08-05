"""Quick win-rate test: how many CAPTCHAs out of N does the model read right?

It fetches fresh CAPTCHAs, reads them with the model, and submits each with FAKE
credentials. The server tells us if the reading was correct:

    errorCode == -1000  -> WRONG reading
    any other errorCode -> CORRECT reading (server confirmed)

At the end it prints e.g. "17 / 20 correct (85.0%)". No real account is used,
so there is no ban risk.

Usage:
    python test_model.py                 # test models/digit_cnn.keras, 20 tries
    python test_model.py --tries 50
    python test_model.py --model models/digit_cnn_light.keras   # compare baseline
"""
import argparse
import base64
import pathlib
import sys
import time

import requests
import urllib3

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from captcha_reader import CaptchaSolver  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BROKER = "bbi"
CAPTCHA_URL = f"https://identity-{BROKER}.ephoenix.ir/api/Captcha/GetCaptcha"
LOGIN_URL = f"https://identity-{BROKER}.ephoenix.ir/api/v2/accounts/login"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0",
    "Referer": f"https://{BROKER}mobile.ephoenix.ir/auth/login",
}
FAKE_USER = "0000000000"
FAKE_PASS = "not_a_real_password"
CAPTCHA_WRONG_CODE = -1000

DEFAULT_MODEL = ROOT / "models" / "digit_cnn.keras"
DELAY = 4.0  # polite pause between tries


def run_test(model_path, tries):
    solver = CaptchaSolver(str(model_path))
    session = requests.Session()

    correct = wrong = no_segment = 0
    done = 0
    attempt = 0
    print(f"Testing model: {model_path}\n")

    while done < tries:
        attempt += 1

        try:
            cj = session.get(CAPTCHA_URL, headers=HEADERS, timeout=10).json()
        except Exception as exc:
            print(f"  network error: {exc}; waiting 5s")
            time.sleep(5)
            continue
        if "captchaByteData" not in cj:  # server throttling, does not count
            print("  server throttling; waiting 15s")
            time.sleep(15)
            continue

        image_bytes = base64.b64decode(cj["captchaByteData"])
        value = solver.solve_bytes(image_bytes)
        if not value:
            no_segment += 1
            done += 1
            print(f"[{done}/{tries}] (could not segment into 5 digits)  X")
            time.sleep(DELAY)
            continue

        payload = {"loginName": FAKE_USER, "password": FAKE_PASS,
                   "captcha": {"salt": cj["salt"],
                               "hash": cj["hashedCaptcha"], "value": value}}
        try:
            body = session.post(LOGIN_URL, json=payload,
                                headers=HEADERS, timeout=10).json()
        except Exception as exc:
            print(f"  login post error: {exc}; waiting 5s")
            time.sleep(5)
            continue

        done += 1
        if body.get("errorCode") == CAPTCHA_WRONG_CODE:
            wrong += 1
            print(f"[{done}/{tries}] read {value}  ->  WRONG   X")
        else:
            correct += 1
            print(f"[{done}/{tries}] read {value}  ->  correct  OK")
        time.sleep(DELAY)

    print("\n---------------------------------------------")
    print(f"{correct} / {tries} correct  ({100.0 * correct / tries:.1f}%)")
    print(f"   wrong reads : {wrong}")
    print(f"   no-segment  : {no_segment}")
    graded = correct + wrong
    if graded:
        print(f"read-accuracy (of the ones submitted): "
              f"{100.0 * correct / graded:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test the CAPTCHA model win rate.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Path to the .keras model to test.")
    parser.add_argument("--tries", type=int, default=20,
                        help="How many CAPTCHAs to try.")
    args = parser.parse_args()
    run_test(args.model, args.tries)


if __name__ == "__main__":
    main()
