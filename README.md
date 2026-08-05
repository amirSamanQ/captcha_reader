# captcha_reader

Read a **5 digit numeric CAPTCHA** with a small CNN, and use it to log in to a
**Sahra OMS** (ephoenix identity) brokerage account.

Many Iranian brokerage OMS (Order Management System) web apps are built on the
"Sahra OMS" / ephoenix platform and ask for a 5 digit numeric CAPTCHA at login.
This project trains a CNN to read that CAPTCHA and ships a tiny login SDK that
downloads the CAPTCHA, reads it, and logs you in.

The bundled model was trained for:

> https://identity-bbi.ephoenix.ir  (Sahra OMS)

> **Use it only on your own account.** You need a real account on the target
> broker. Never hard-code or commit your credentials, cookies, or tokens.

---

## Win rate

The win rate is the chance a single CAPTCHA is read completely correctly (all 5
digits), which is roughly how often one login attempt gets past the CAPTCHA.

| Model                          | Win rate (per attempt) |
|--------------------------------|------------------------|
| First trained model            | ~18%  (about 1 in 6)   |
| **Bundled model** (this repo)  | **~84%** (about 5 in 6)|

The bundled model was **not** hand-labeled up to 84%. It got there on its own,
using the self-improving loop described below: the login server itself tells us
which CAPTCHA readings were correct, so the model can collect verified training
data automatically and keep retraining until it beats its previous best.

---

## Project layout

```
captcha_reader/
├── captcha_reader/            # the installable package (SDK + shared code)
│   ├── segmentation.py        # split a CAPTCHA image into 5 digit crops
│   ├── solver.py              # CaptchaSolver: image -> "12345" using the CNN
│   └── login.py               # login SDK for the Sahra OMS
├── scripts/                   # build a dataset by hand, one job per script
│   ├── collect_captchas.py    # download CAPTCHAs to a folder
│   ├── label_captchas.py      # label them by hand
│   └── train.py               # segment digits + train the CNN
├── probe_validation_order.py  # self-training step 0: is the CAPTCHA checked first?
├── harvest.py                 # self-training step 1: auto-collect verified CAPTCHAs
├── autotrain.py               # self-training step 2: retrain + promote a better model
├── test_model.py              # measure a model's win rate on fresh CAPTCHAs
├── examples/
│   └── login_example.py       # minimal SDK usage
├── models/
│   └── digit_cnn_light.keras  # the bundled model (default used by the SDK)
└── data/                      # your datasets go here (git-ignored)
```

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

## Quick start: log in with the bundled model

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
  **This is what the bundled model was trained for.**
- **`login_futures()`** — exphoenixfuture (`bbi.exphoenixfuture.ir`).
  Form-urlencoded body; CAPTCHA image comes back under `img`. The bundled model
  was **not** trained for this site — train your own model first.

### Using a specific model

The SDK uses `models/digit_cnn_light.keras` by default. To use a different
model, pass your own solver:

```python
from captcha_reader import CaptchaSolver, login

solver = CaptchaSolver("models/my_model.keras")
result = login("user", "pass", broker="bbi", solver=solver)
```

---

## Test a model's win rate

Measure how often a model reads a fresh CAPTCHA correctly. It uses **fake**
credentials, so no real account is touched (see the safety note below):

```bash
python test_model.py --tries 20
# test a different model, e.g. to compare:
python test_model.py --model models/digit_cnn_light.keras --tries 50
```

Example output:

```
[1/20] read 84195  ->  correct  OK
...
17 / 20 correct  (85.0%)
```

---

## Improve the model automatically (self-training loop)

This is the interesting part. The login server checks the CAPTCHA **before** the
username/password, and reports a distinct error for a wrong CAPTCHA. That lets us
send a CAPTCHA our model just read, together with **fake credentials**, and learn
whether the reading was right — without ever touching a real account:

```
errorCode == -1000  ("wrong security code")     -> our reading was WRONG
any other errorCode ("invalid username/pass")   -> our reading was CORRECT
```

Every correct reading is a CAPTCHA whose true label we now know for free. We save
those as training data, retrain, and keep the new model only if it truly reads
better. Run it in two terminals:

```bash
# Terminal 0 (once): confirm the server checks the CAPTCHA first.
python probe_validation_order.py

# Terminal 1: harvest server-verified CAPTCHAs into data/verified/ (runs forever)
python harvest.py

# Terminal 2: retrain when enough new data arrives, promote the winner (runs forever)
python autotrain.py
```

How the loop closes:

```
harvest.py  ──►  data/verified/  ──►  autotrain.py
    ▲                                      │  trains a challenger, then scores
    │                                      │  champion vs challenger on FRESH
    │                                      ▼  CAPTCHAs; promotes only if better
    └──────  models/digit_cnn.keras  ◄─────┘
         (harvest reloads the improved model automatically, no restart)
```

- **`harvest.py`** reads fresh CAPTCHAs and saves only the ones the server
  confirms were read correctly, to `data/verified/` (git-ignored).
- **`autotrain.py`** retrains once the verified pool grows enough, saves each
  attempt to a numbered file `models/digit_cnn_v###.keras`, and promotes it to
  `models/digit_cnn.keras` **only if** it beats the current model on a batch of
  fresh CAPTCHAs. The old model is never lost.
- **`harvest.py`** notices the promoted model and switches to it on the fly.

When you are happy with `models/digit_cnn.keras`, copy it over the bundled model
so all your code picks it up with no changes:

```bash
cp models/digit_cnn.keras models/digit_cnn_light.keras
```

### Safety notes for the loop

- **No ban risk:** harvesting and testing use fake credentials
  (`0000000000` / a fake password), so no real account is ever logged into or
  locked out.
- **Be polite:** the server rate-limits if you request too fast. `harvest.py`
  backs off automatically when that happens; raise its `DELAY` if you see a lot
  of throttling.
- **Nothing sensitive is committed:** `data/` and the numbered/iteration models
  are git-ignored.

---

## Train your own model from scratch

Useful if you target a different OMS and cannot use the self-training loop. Run
the three steps in order:

```bash
# 1) Collect raw CAPTCHAs into data/captchas/
python scripts/collect_captchas.py --broker bbi --count 200 --out data/captchas

# 2) Label them by hand -> data/captcha_labels.json
python scripts/label_captchas.py --images data/captchas --labels data/captcha_labels.json

# 3) Segment digits and train -> models/digit_cnn.keras
python scripts/train.py --images data/captchas --labels data/captcha_labels.json --out models/digit_cnn.keras
```

`train.py` binarizes every CAPTCHA, splits it into 5 digit crops (left to right),
and pairs digit *i* with character *i* of your label. CAPTCHAs that do not cleanly
split into exactly 5 digits are skipped. The digit preprocessing matches the
solver exactly, so training and inference stay consistent.

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

The remaining weak spot is **step 3**: when two digits touch or noise is picked
up, segmentation returns the wrong number of boxes and that CAPTCHA is discarded
(shown as `no-segment`). This is a segmentation limit, not a CNN limit, so a
better reader does not fix it — the SDK simply fetches a fresh CAPTCHA.

---

## Notes

- TLS verification is disabled for the OMS hosts (some use self-signed or
  mismatched certificates), matching the original scripts. Remove `verify=False`
  in `login.py` if your host has a valid certificate.
- `data/`, `cookies.txt`, `.env` and token files are git-ignored so you do not
  accidentally commit CAPTCHAs, credentials, or session tokens.
