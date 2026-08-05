"""Step 1 - Collect CAPTCHA images to build a training dataset.

This downloads CAPTCHA images from the OMS CAPTCHA endpoint and saves them as
PNG files. Run it, wait, and you get a folder full of raw CAPTCHAs ready to be
labeled (see label_captchas.py).

The two known OMS flavors return the image under different JSON keys, so this
script accepts either "captchaByteData" (ephoenix identity) or "img"
(exphoenixfuture). It backs off automatically when the server rate-limits it
(HTTP 429).

Example:
    python scripts/collect_captchas.py --broker bbi --count 200 --out data/captchas
"""

import argparse
import base64
import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0",
}


def extract_image_bytes(data):
    """Pull the CAPTCHA image bytes out of an OMS JSON response."""
    b64 = data.get("captchaByteData") or data.get("img")
    if not b64:
        return None
    if b64.startswith("data:image"):  # strip a data-URI prefix if present
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def collect(url, out_dir, count, delay):
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    saved = 0

    i = 0
    while saved < count:
        i += 1
        try:
            resp = session.get(url, headers=HEADERS, timeout=10, verify=False)
            data = resp.json()
        except Exception as exc:
            print(f"[{i}] request error: {exc}; waiting 5s")
            time.sleep(5)
            continue

        image_bytes = extract_image_bytes(data)
        if image_bytes is None:
            # Most often this is a rate-limit message; wait a bit longer.
            print(f"[{i}] no image in response ({data}); waiting 10s")
            time.sleep(10)
            continue

        filename = os.path.join(out_dir, f"captcha_{saved:04d}.png")
        with open(filename, "wb") as f:
            f.write(image_bytes)
        saved += 1
        print(f"[{saved}/{count}] saved {filename}")
        time.sleep(delay)

    print(f"Done. Saved {saved} CAPTCHAs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download CAPTCHAs to build a dataset.")
    parser.add_argument("--broker", default="bbi",
                        help="Broker sub-domain for the default ephoenix URL.")
    parser.add_argument("--url", default=None,
                        help="Full CAPTCHA endpoint URL (overrides --broker).")
    parser.add_argument("--out", default="data/captchas",
                        help="Output folder for the PNG files.")
    parser.add_argument("--count", type=int, default=100,
                        help="How many CAPTCHAs to save.")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between requests (avoid rate limits).")
    args = parser.parse_args()

    url = args.url or f"https://identity-{args.broker}.ephoenix.ir/api/Captcha/GetCaptcha"
    print(f"CAPTCHA endpoint: {url}")
    collect(url, args.out, args.count, args.delay)


if __name__ == "__main__":
    main()
