"""Automated CAPTCHA harvester - self-labeling, no real account needed.

WHY THIS IS SAFE (no ban risk)
------------------------------
The probe (probe_validation_order.py) proved the OMS checks the CAPTCHA
BEFORE the username/password. So we send a FAKE username + FAKE password
together with the CAPTCHA our model just read:

    errorCode == -1000  ("wrong security code")  -> our reading was WRONG
    any other errorCode (e.g. 3000 "wrong user") -> our reading was CORRECT

Because the credentials are fake and belong to no real account, there is
nothing to ban. Whenever the server complains about the USER instead of the
CAPTCHA, we have a CAPTCHA image whose TRUE label is known (what the model
read). We save that (image, label) pair as server-verified training data.

Over time this builds a clean, verified dataset for free. autotrain.py then
retrains on it and swaps in a better model, which this script auto-detects.

OUTPUT (all git-ignored - everything under data/ is ignored)
    data/verified/captchas/verified_XXXXX.png   the raw CAPTCHA images
    data/verified/labels.json                   {filename: "12345"} (train.py format)

Place this file at the repo root and run:  python harvest.py
Stop anytime with Ctrl+C - labels are saved after every hit.
"""
import base64
import json
import os
import pathlib
import sys
import time
from collections import Counter

import requests
import urllib3

# Make the captcha_reader package importable no matter the current directory.
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from captcha_reader import CaptchaSolver  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Target OMS (ephoenix identity flavor) ---
BROKER = "bbi"
CAPTCHA_URL = f"https://identity-{BROKER}.ephoenix.ir/api/Captcha/GetCaptcha"
LOGIN_URL = f"https://identity-{BROKER}.ephoenix.ir/api/v2/accounts/login"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0",
    "Referer": f"https://{BROKER}mobile.ephoenix.ir/auth/login",
}

# --- Safety: fake credentials, no real account is ever touched ---
FAKE_USER = "0000000000"
FAKE_PASS = "not_a_real_password"
CAPTCHA_WRONG_CODE = -1000  # server code meaning "the CAPTCHA was wrong"

# --- Where verified data is stored (data/ is git-ignored) ---
OUT_DIR = ROOT / "data" / "verified"
IMAGES_DIR = OUT_DIR / "captchas"
LABELS_PATH = OUT_DIR / "labels.json"

# --- Which model to read with. If autotrain produced an improved model, use
#     it automatically; otherwise fall back to the shipped sample model. ---
IMPROVED_MODEL = ROOT / "models" / "digit_cnn.keras"
SAMPLE_MODEL = ROOT / "models" / "digit_cnn_light.keras"

DELAY = 4.0        # polite pause between attempts (seconds)
RELOAD_EVERY = 50  # re-check for a newer model every N attempts
THROTTLE_BASE = 15  # base backoff (seconds) when the server starts throttling
THROTTLE_MAX = 90   # cap the backoff so we never sleep too long


def current_model_path():
    return IMPROVED_MODEL if IMPROVED_MODEL.exists() else SAMPLE_MODEL


def load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    saved = len(labels)
    print(f"Starting with {saved} verified samples already on disk.")

    model_path = current_model_path()
    print(f"Reading with model: {model_path}")
    solver = CaptchaSolver(str(model_path))
    loaded_mtime = model_path.stat().st_mtime

    session = requests.Session()
    stats = Counter()
    attempt = 0
    throttle_streak = 0  # how many throttle responses in a row

    try:
        while True:
            attempt += 1

            # Hot-swap to a newer/better model if autotrain produced one.
            if attempt % RELOAD_EVERY == 0:
                mp = current_model_path()
                mt = mp.stat().st_mtime
                if mp != model_path or mt != loaded_mtime:
                    print(f"\n>>> Reloading improved model from {mp}\n")
                    solver = CaptchaSolver(str(mp))
                    model_path, loaded_mtime = mp, mt

            # 1) Get one CAPTCHA (base64 PNG + salt + server-side hash).
            try:
                cj = session.get(CAPTCHA_URL, headers=HEADERS, timeout=10).json()
            except Exception as exc:
                print(f"[{attempt}] network error: {exc}; waiting 5s")
                time.sleep(5)
                continue

            # If the response has no image, the server is throttling us. Back
            # off with an increasing delay so we stop hammering it.
            if "captchaByteData" not in cj:
                throttle_streak += 1
                backoff = min(THROTTLE_MAX, THROTTLE_BASE * throttle_streak)
                stats["throttled"] += 1
                print(f"[{attempt}] server throttling (no captcha); waiting {backoff}s")
                time.sleep(backoff)
                continue
            throttle_streak = 0  # got a real captcha, reset the backoff
            image_bytes = base64.b64decode(cj["captchaByteData"])

            # 2) Read it with the model.
            value = solver.solve_bytes(image_bytes)
            if not value:
                stats["no_segment"] += 1  # did not split into 5 clean digits
                time.sleep(DELAY)
                continue

            # 3) Submit with FAKE creds so the server tells us if the read was
            #    right. Correct read -> it moves past the captcha to the user
            #    check (errorCode != -1000). Wrong read -> errorCode == -1000.
            payload = {
                "loginName": FAKE_USER,
                "password": FAKE_PASS,
                "captcha": {
                    "salt": cj["salt"],
                    "hash": cj["hashedCaptcha"],
                    "value": value,
                },
            }
            try:
                body = session.post(LOGIN_URL, json=payload,
                                    headers=HEADERS, timeout=10).json()
            except Exception as exc:
                print(f"[{attempt}] login post error: {exc}; waiting 5s")
                time.sleep(5)
                continue

            code = body.get("errorCode")
            if code == CAPTCHA_WRONG_CODE:
                stats["captcha_wrong"] += 1
            else:
                # CORRECT reading -> save as verified training data.
                fname = f"verified_{saved:05d}.png"
                with open(IMAGES_DIR / fname, "wb") as f:
                    f.write(image_bytes)
                labels[fname] = value
                save_labels(labels)
                saved += 1
                stats["verified"] += 1
                print(f"[{attempt}] VERIFIED {value}   (total {saved})")

            # Live read-accuracy over CAPTCHAs that at least segmented.
            graded = stats["verified"] + stats["captcha_wrong"]
            if graded and attempt % 10 == 0:
                acc = 100.0 * stats["verified"] / graded
                print(f"     ... read-accuracy: {acc:.1f}%  "
                      f"(verified {stats['verified']}, wrong {stats['captcha_wrong']}, "
                      f"no-segment {stats['no_segment']})")

            time.sleep(DELAY)

    except KeyboardInterrupt:
        pass
    finally:
        save_labels(labels)
        print(f"\nStopped. {saved} verified samples in {IMAGES_DIR}")
        print(f"Session stats: {dict(stats)}")


if __name__ == "__main__":
    main()
