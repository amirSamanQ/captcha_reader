# captcha_reader

Read a **5 digit numeric CAPTCHA** with a small CNN, and use it to log in to a
**Sahra OMS** (ephoenix identity) brokerage account.

Many Iranian brokerage OMS (Order Management System) web apps are built on the
"Sahra OMS" / ephoenix platform and ask for a 5 digit numeric CAPTCHA at login.
This project trains a CNN to read that CAPTCHA and ships a tiny login SDK that
downloads the CAPTCHA, reads it, and logs you in.

A pre-trained sample model is included. It was trained for:

> https://identity-bbi.ephoenix.ir  (Sahra OMS)

The CAPTCHA reader is not perfect. With the sample model you get roughly **1
successful login per ~10 tries**, so the SDK simply retries with a fresh
CAPTCHA until it gets in.

> **Use it only on your own account.** You need a real account on the target
> broker. Never hard-code or commit your credentials.

---

## Project layout

```
captcha_reader/
├── captcha_reader/          # the installable package (SDK + shared code)
│   ├── segmentation.py      # split a CAPTCHA image into 5 digit crops
│   ├── solver.py            # CaptchaSolver: image -> "12345" using the CNN
│   └── login.py             # login SDK for the Sahra OMS
├── scripts/                 # the data pipeline, one job per script
│   ├── collect_captchas.py  # step 1: download CAPTCHAs to build a dataset
│   ├── label_captchas.py    # step 2: label them by hand
│   └── train.py             # step 3: segment digits + train the CNN
├── examples/
│   └── login_example.py     # minimal SDK usage
├── models/
│   └── digit_cnn_light.keras  # pre-trained sample model (Sahra OMS)
└── data/                    # your datasets go here (git-ignored)
```

The three `scripts/` are deliberately independent so you can run just the step
you need. The `captcha_reader/` package holds the shared segmentation logic and
the SDK.

---

## Install

```bash
pip install -r requirements.txt
# or, to install the package itself (SDK only):
pip install -e .
# add the training extras (matplotlib, scikit-learn) if you will train:
pip install -e ".[train]"
```

Python 3.9+ is recommended.

---

## Quick start: log in with the sample model

```python
from captcha_reader import login

result = login("your_bourse_code", "your_password", broker="bbi")
print(result["token"])
print(result["cookie"])
```

Or run the example:

```bash
export OMS_USERNAME="your_bourse_code"
export OMS_PASSWORD="your_password"
python examples/login_example.py
```

`login(...)` returns a dict:

| key          | meaning                                        |
|--------------|------------------------------------------------|
| `token`      | JWT bearer token for API calls                 |
| `session_id` | OMS session id                                 |
| `cookie`     | cookie header string for the logged-in session |
| `response`   | the raw login response JSON                     |

### Which login function?

There are two OMS request formats. Pick the one that matches your site:

- **`login()`** — ephoenix identity (`identity-<broker>.ephoenix.ir`).
  JSON body; CAPTCHA sent back as a nested `{salt, hash, value}` object.
  **This is what the sample model was trained for.**
- **`login_futures()`** — exphoenixfuture (`bbi.exphoenixfuture.ir`).
  Form-urlencoded body; CAPTCHA image comes back under `img`. The sample model
  was **not** trained for this site — train your own model first.

---

## Train your own model

Useful if the sample model is not accurate enough, or you target a different
OMS. Run the three steps in order:

```bash
# 1) Collect raw CAPTCHAs into data/captchas/
python scripts/collect_captchas.py --broker bbi --count 200 --out data/captchas

# 2) Label them by hand -> data/captcha_labels.json
python scripts/label_captchas.py --images data/captchas --labels data/captcha_labels.json

# 3) Segment digits and train -> models/digit_cnn.keras
python scripts/train.py --images data/captchas --labels data/captcha_labels.json --out models/digit_cnn.keras
```

Then point the solver at your new model:

```python
from captcha_reader import CaptchaSolver, login

solver = CaptchaSolver("models/digit_cnn.keras")
result = login("user", "pass", broker="bbi", solver=solver)
```

### How training works

You label each CAPTCHA with the 5 digits you see. `train.py` then binarizes
every CAPTCHA, splits it into 5 digit crops (left to right), and pairs digit *i*
with character *i* of your label — turning whole-CAPTCHA labels into a
single-digit training set. CAPTCHAs that do not cleanly split into exactly 5
digits are skipped. The digit preprocessing matches the solver exactly, so
training and inference stay consistent.

---

## How it works (pipeline)

1. **Get CAPTCHA** — the OMS returns a base64 PNG plus a `salt` and a
   server-side `hash`.
2. **Binarize** — Otsu threshold to white digits on black (`segmentation.py`).
3. **Segment** — find contours, drop noise, keep exactly 5 boxes, sort left to
   right. If it is not exactly 5, the attempt is abandoned and retried.
4. **Read** — the CNN predicts each digit; the 5 predictions are joined into a
   string (`solver.py`).
5. **Log in** — post `loginName`, `password`, and the CAPTCHA answer together
   with the original `salt` and `hash`. Retry until success (`login.py`).

The main weak spot is **step 3**: when two digits touch or noise is picked up,
segmentation returns the wrong number of boxes and the whole CAPTCHA is
discarded. That is why several attempts are usually needed per login.

---

## Notes

- TLS verification is disabled for the OMS hosts (some use self-signed or
  mismatched certificates), matching the original scripts. Remove `verify=False`
  in `login.py` if your host has a valid certificate.
- `data/`, `cookies.txt`, `.env` and token files are git-ignored so you do not
  accidentally commit CAPTCHAs, credentials, or session tokens.
