"""Login SDK for the Sahra OMS (ephoenix identity) trading platform.

WHAT THIS IS
------------
Several Iranian brokerage OMS (Order Management System) web apps are built on
the "Sahra OMS" / ephoenix platform. Their login endpoint asks for a 5 digit
numeric CAPTCHA. This module logs in for you: it downloads the CAPTCHA, reads
it with the bundled CNN model (``CaptchaSolver``), and posts the login form.
Because the CAPTCHA reader is not perfect, it retries until it gets in (with
the sample model, about 1 success every ~10 tries).

The sample model shipped in this repo was trained for:

    https://identity-bbi.ephoenix.ir      (Sahra OMS)

TWO API FLAVORS
---------------
There are two request formats in the wild. Use the one that matches your OMS:

* ``login()``         -> ephoenix identity ("identity-<broker>.ephoenix.ir").
                         JSON body; the CAPTCHA image comes back under
                         "captchaByteData" and is sent back as a nested
                         {salt, hash, value} object. This is the flavor the
                         sample model was trained for.

* ``login_futures()`` -> exphoenixfuture ("bbi.exphoenixfuture.ir").
                         Form-urlencoded body; the CAPTCHA image comes back
                         under "img". The sample model was NOT trained for this
                         site, so train your own model first (see scripts/).

IMPORTANT
---------
You must have a real account on the target broker, and only log in to your own
account. Pass your own credentials at call time; never hard-code them in the
source or commit them to git.
"""

import base64
import time

import requests
import urllib3

from .solver import CaptchaSolver

# The OMS uses self-signed / mismatched certificates on some hosts; the
# original scripts disabled TLS verification. We keep that behavior but hide
# the noisy warning. Remove verify=False below if your host has a valid cert.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# One shared solver so the CNN model is loaded only once per process.
_shared_solver = None


def _get_solver():
    global _shared_solver
    if _shared_solver is None:
        _shared_solver = CaptchaSolver()
    return _shared_solver


def login(username, password, broker="bbi", max_attempts=15,
          captcha_url=None, login_url=None, solver=None):
    """Log in to a Sahra OMS (ephoenix identity) account.

    Args:
        username: Your login name (usually the brokerage / "bourse" code).
        password: Your account password.
        broker: Broker sub-domain used to build the default URLs
            ("identity-<broker>.ephoenix.ir"). Ignored if you pass
            captcha_url / login_url yourself.
        max_attempts: How many CAPTCHAs to try before giving up.
        captcha_url: Override the CAPTCHA endpoint.
        login_url: Override the login endpoint.
        solver: A CaptchaSolver instance. A shared default one is used if None.

    Returns:
        dict with "token", "session_id", "cookie" and the raw "response".

    Raises:
        RuntimeError: if login did not succeed within max_attempts.
    """
    if captcha_url is None:
        captcha_url = f"https://identity-{broker}.ephoenix.ir/api/Captcha/GetCaptcha"
    if login_url is None:
        login_url = f"https://identity-{broker}.ephoenix.ir/api/v2/accounts/login"

    solver = solver or _get_solver()
    session = requests.Session()
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://{broker}mobile.ephoenix.ir/auth/login",
    }

    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}/{max_attempts}")

        # 1) Download one CAPTCHA (base64 PNG + salt + a server-side hash).
        try:
            captcha = session.get(captcha_url, headers=headers, timeout=10).json()
        except Exception as exc:
            print(f"  could not fetch CAPTCHA: {exc}")
            time.sleep(1)
            continue

        # 2) Read the CAPTCHA with the CNN model.
        value = solver.solve_bytes(base64.b64decode(captcha["captchaByteData"]))
        if not value:
            print("  CAPTCHA did not segment into 5 digits, retrying...")
            continue
        print(f"  CAPTCHA read as: {value}")

        # 3) Send the login request. The answer must be returned together with
        #    the exact salt and hash the server gave us for this CAPTCHA.
        payload = {
            "loginName": username,
            "password": password,
            "captcha": {
                "salt": captcha["salt"],
                "hash": captcha["hashedCaptcha"],
                "value": value,
            },
        }
        try:
            resp = session.post(login_url, json=payload, headers=headers, timeout=10)
            body = resp.json()
        except Exception as exc:
            print(f"  login request failed: {exc}")
            time.sleep(1)
            continue

        if body.get("isSuccess") or body.get("token"):
            print("Login OK")
            return {
                "token": body.get("token"),
                "session_id": body.get("sessionId"),
                "cookie": "; ".join(f"{c.name}={c.value}" for c in session.cookies),
                "response": body,
            }
        print(f"  login failed: {body.get('errorMessage')}")
        time.sleep(1)

    raise RuntimeError(f"Login failed after {max_attempts} attempts")


def login_futures(username, password, max_attempts=15,
                  captcha_url="https://bbi.exphoenixfuture.ir:8080/api/v5/user/captcha",
                  login_url="https://bbi.exphoenixfuture.ir:8080/api/v5/user/login",
                  solver=None):
    """Log in to the exphoenixfuture OMS (a different request format).

    This OMS returns the CAPTCHA image under "img" (a data-URI) and expects a
    form-urlencoded login body with flattened "captcha.*" fields. The bundled
    sample model was NOT trained for this site, so you will likely need to
    train your own model on its CAPTCHAs first (see scripts/).

    Returns the same dict shape as ``login`` plus the live requests session
    under "session".
    """
    solver = solver or _get_solver()
    session = requests.Session()
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://bbi.exphoenixfuture.ir",
        "Referer": "https://bbi.exphoenixfuture.ir/",
    }

    # Visit the site once to pick up the initial cookie the login expects.
    try:
        session.get("https://bbi.exphoenixfuture.ir", headers=headers,
                    timeout=10, verify=False)
    except Exception as exc:
        print(f"  initial visit failed: {exc}")

    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}/{max_attempts}")

        # 1) Download one CAPTCHA. Here the image is under "img" as a data-URI.
        try:
            captcha = session.get(captcha_url, headers=headers,
                                  timeout=10, verify=False).json()
        except Exception as exc:
            print(f"  could not fetch CAPTCHA: {exc}")
            time.sleep(1)
            continue

        img_field = captcha["img"]
        if img_field.startswith("data:image"):
            img_field = img_field.split(",", 1)[1]
        value = solver.solve_bytes(base64.b64decode(img_field))
        if not value:
            print("  CAPTCHA did not segment into 5 digits, retrying...")
            continue
        print(f"  CAPTCHA read as: {value}")

        # 2) Send the login request as a form (not JSON) with flattened fields.
        payload = {
            "username": username,
            "password": password,
            "otp": "",
            "captcha.salt": captcha["salt"],
            "captcha.captcha": value,
            "captcha.hashedCaptcha": captcha["hashedCaptcha"],
        }
        try:
            resp = session.post(
                login_url,
                data=payload,
                headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
                verify=False,
            )
            body = resp.json()
        except Exception as exc:
            print(f"  login request failed: {exc}")
            time.sleep(1)
            continue

        if body.get("isSuccess") or body.get("token") or body.get("userId"):
            print("Login OK")
            return {
                "token": body.get("token"),
                "session_id": body.get("sessionId"),
                "cookie": "; ".join(f"{c.name}={c.value}" for c in session.cookies),
                "response": body,
                "session": session,
            }
        print(f"  login failed: {body.get('errorMessage', body)}")
        time.sleep(1)

    raise RuntimeError(f"Login failed after {max_attempts} attempts")
