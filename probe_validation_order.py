"""Probe: does the OMS validate the CAPTCHA before the credentials?

This is the one-off experiment that makes safe, automated data harvesting
possible (see harvest.py). It answers a single question: when a login request
is wrong on BOTH the CAPTCHA and the username, which error does the server
report first?

Decision rule:
  - Send a FAKE username + a DELIBERATELY WRONG captcha ("").
      * If the server returns the CAPTCHA error  -> it checks the captcha FIRST.
        => harvesting with fake credentials can reveal captcha correctness. GOOD.
      * If it returns a user/credential error     -> it checks the user first.
        => fake credentials cannot reveal captcha validity; use another approach.
  - Then send a FAKE username + a model-read (likely CORRECT) captcha a few
    times. If a NEW error code appears, that confirms correct captchas are
    distinguishable from wrong ones.

For identity-bbi.ephoenix.ir the observed result is:
    wrong captcha   -> errorCode -1000 ("wrong security code")
    correct captcha -> errorCode  3000 ("invalid username or password")
so the captcha is checked first and harvest.py can label data for free.

No real account is used, so there is no ban risk.
"""
import base64
import pathlib
import sys
import time
from collections import Counter

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

FAKE_USER = "0000000000"  # a username that does NOT exist
FAKE_PASS = "not_a_real_password"


def submit(session, captcha_json, value):
    payload = {"loginName": FAKE_USER, "password": FAKE_PASS,
               "captcha": {"salt": captcha_json["salt"],
                           "hash": captcha_json["hashedCaptcha"], "value": value}}
    body = session.post(LOGIN_URL, json=payload, headers=HEADERS, timeout=10).json()
    return body.get("errorCode"), body.get("errorMessage")


def main():
    solver = CaptchaSolver()
    session = requests.Session()
    outcomes = Counter()

    # Part A: force a WRONG captcha with an empty value.
    for _ in range(3):
        cj = session.get(CAPTCHA_URL, headers=HEADERS, timeout=10).json()
        code, msg = submit(session, cj, "")
        outcomes[("empty_captcha", code)] += 1
        print(f"[empty ] code={code}  msg={msg}")
        time.sleep(2)

    # Part B: send model-read (hopefully correct) captchas.
    for _ in range(15):
        cj = session.get(CAPTCHA_URL, headers=HEADERS, timeout=10).json()
        value = solver.solve_bytes(base64.b64decode(cj["captchaByteData"]))
        if not value:
            continue
        code, msg = submit(session, cj, value)
        outcomes[("model_captcha", code)] += 1
        print(f"[model ] value={value}  code={code}  msg={msg}")
        time.sleep(2)

    print("\nSummary:", dict(outcomes))
    print("If you see a code OTHER than the empty-captcha code in the "
          "'model_captcha' rows, harvesting with fake credentials works.")


if __name__ == "__main__":
    main()
