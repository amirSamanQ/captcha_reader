"""Automated retrainer with an UNBIASED online evaluation gate.

Run this ALONGSIDE harvest.py (in a second terminal). It watches the pool of
server-verified CAPTCHAs. When the pool has grown enough, it trains a NEW model
and only promotes it if it truly reads better than the model in use.

WHY THE OLD "HOLDOUT" GATE WAS WRONG
------------------------------------
harvest.py only saves CAPTCHAs the current model already reads correctly. Any
holdout carved out of that pool is therefore made entirely of "easy" CAPTCHAs
the champion is guaranteed to get right, so the champion always scored ~100% and
no challenger could ever win. That gate could never promote anything.

THE FIX: evaluate on FRESH CAPTCHAs from the server
---------------------------------------------------
The only unbiased test is the real one. To score a model we fetch fresh
CAPTCHAs, read them with that model, and submit each with FAKE credentials:

    errorCode == -1000  -> the read was WRONG
    any other errorCode -> the read was CORRECT (server confirmed)

The score is verified / (verified + wrong) over N fresh CAPTCHAs - exactly the
"read-accuracy" you watch in harvest, and it includes the hard CAPTCHAs too.
(no-segment cases are excluded: segmentation is shared, model-independent code,
so it affects both models equally.)

SAFETY GUARANTEES (unchanged)
-----------------------------
1) Every trained model is saved to its own numbered file
   models/digit_cnn_vNNN.keras. Nothing is ever overwritten or deleted, and the
   shipped sample models/digit_cnn_light.keras is never touched.
2) The challenger is promoted (copied to models/digit_cnn.keras, the file
   harvest.py reads) ONLY if it beats the champion by at least EVAL_MARGIN on
   fresh CAPTCHAs. Worst case: we keep the old model. We never lose a good one.

Place this at the repo root and run:  python autotrain.py   (Ctrl+C to stop)
"""
import base64
import json
import pathlib
import shutil
import subprocess
import sys
import time

import requests
import urllib3

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from captcha_reader import CaptchaSolver  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Target OMS (same endpoints harvest.py uses) ---
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

# --- Paths ---
VERIFIED_DIR = ROOT / "data" / "verified"
IMAGES_DIR = VERIFIED_DIR / "captchas"
LABELS_PATH = VERIFIED_DIR / "labels.json"
MARKER_PATH = VERIFIED_DIR / "last_trained.json"
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
MODELS_DIR = ROOT / "models"
CHAMPION_PATH = MODELS_DIR / "digit_cnn.keras"       # what harvest.py reads
SAMPLE_PATH = MODELS_DIR / "digit_cnn_light.keras"   # shipped baseline

# --- Retrain trigger ---
CHECK_EVERY = 120     # seconds between checks
MIN_NEW = 20          # need at least this many new samples since last train
MIN_GROWTH = 0.10     # ...and the pool must grow by at least 10%
EPOCHS = 40

# --- Online evaluation gate ---
EVAL_PROBES = 150     # fresh CAPTCHAs to score each model on
EVAL_MARGIN = 0.03    # challenger must win by >= 3 percentage points to promote
EVAL_DELAY = 1.5      # seconds between probes (be polite to the server)


def load_labels():
    if not LABELS_PATH.exists():
        return {}
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def last_trained():
    if MARKER_PATH.exists():
        with open(MARKER_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("count", 0)
    return 0


def set_last_trained(count):
    with open(MARKER_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": count}, f)


def should_retrain(current, last):
    if current < MIN_NEW or current - last < MIN_NEW:
        return False
    if last > 0 and (current - last) / last < MIN_GROWTH:
        return False
    return True


def next_version_path():
    n = 0
    for p in MODELS_DIR.glob("digit_cnn_v*.keras"):
        try:
            n = max(n, int(p.stem.split("_v")[-1]))
        except ValueError:
            continue
    return MODELS_DIR / f"digit_cnn_v{n + 1:03d}.keras"


def online_read_accuracy(model_path, n_probes):
    """Score a model on FRESH server CAPTCHAs: verified / (verified + wrong)."""
    solver = CaptchaSolver(str(model_path))
    session = requests.Session()
    verified = wrong = graded = fetched = 0
    max_fetches = n_probes * 5  # safety cap so a bad network can't loop forever

    while graded < n_probes and fetched < max_fetches:
        fetched += 1
        try:
            cj = session.get(CAPTCHA_URL, headers=HEADERS, timeout=10).json()
            image_bytes = base64.b64decode(cj["captchaByteData"])
        except Exception:
            time.sleep(3)
            continue

        value = solver.solve_bytes(image_bytes)
        if not value:
            continue  # no-segment: model-independent, excluded from the score

        payload = {"loginName": FAKE_USER, "password": FAKE_PASS,
                   "captcha": {"salt": cj["salt"],
                               "hash": cj["hashedCaptcha"], "value": value}}
        try:
            body = session.post(LOGIN_URL, json=payload,
                                headers=HEADERS, timeout=10).json()
        except Exception:
            time.sleep(3)
            continue

        if body.get("errorCode") == CAPTCHA_WRONG_CODE:
            wrong += 1
        else:
            verified += 1
        graded += 1
        time.sleep(EVAL_DELAY)

    return (verified / graded if graded else 0.0), graded


def retrain_and_maybe_promote(current):
    # 1) Train a challenger on ALL verified data -> its own numbered file.
    challenger_path = next_version_path()
    print(f"\n=== Training challenger -> {challenger_path.name} "
          f"on {current} samples ===")
    result = subprocess.run([
        sys.executable, str(TRAIN_SCRIPT),
        "--images", str(IMAGES_DIR),
        "--labels", str(LABELS_PATH),
        "--out", str(challenger_path),
        "--epochs", str(EPOCHS),
    ])
    if result.returncode != 0 or not challenger_path.exists():
        print("=== Training failed; keeping current model ===\n")
        return

    # 2) Fair contest on FRESH server CAPTCHAs (unbiased, includes hard ones).
    champion_path = CHAMPION_PATH if CHAMPION_PATH.exists() else SAMPLE_PATH
    print(f"\n=== Scoring on {EVAL_PROBES} fresh CAPTCHAs each "
          f"(~{2 * EVAL_PROBES * EVAL_DELAY / 60:.0f} min) ===")
    print("Scoring challenger...")
    chal_acc, n_chal = online_read_accuracy(challenger_path, EVAL_PROBES)
    print("Scoring champion...")
    champ_acc, n_champ = online_read_accuracy(champion_path, EVAL_PROBES)

    print(f"\nFresh-CAPTCHA read-accuracy:")
    print(f"   champion  ({champion_path.name}): {champ_acc*100:.1f}%  (n={n_champ})")
    print(f"   challenger ({challenger_path.name}): {chal_acc*100:.1f}%  (n={n_chal})")

    # 3) Promote only if the challenger clearly wins.
    if chal_acc >= champ_acc + EVAL_MARGIN:
        shutil.copyfile(challenger_path, CHAMPION_PATH)
        print(f">>> PROMOTED {challenger_path.name} -> {CHAMPION_PATH.name} "
              f"(+{(chal_acc - champ_acc)*100:.1f} pts). "
              f"harvest.py will pick it up automatically.\n")
    else:
        print(f">>> Challenger did not beat champion by >= "
              f"{EVAL_MARGIN*100:.0f} pts; keeping current model. "
              f"({challenger_path.name} kept on disk.)\n")


def main():
    print("Watching the verified pool. Ctrl+C to stop.")
    try:
        while True:
            current = len(load_labels())
            last = last_trained()
            print(f"pool={current}  last_trained={last}")
            if should_retrain(current, last):
                retrain_and_maybe_promote(current)
                set_last_trained(current)  # count the attempt regardless of outcome
            time.sleep(CHECK_EVERY)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
