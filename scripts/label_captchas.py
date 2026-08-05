"""Step 2 - Label the collected CAPTCHAs by hand.

Each CAPTCHA image is shown to you; type the 5 digits you see and press Enter.
Labels are saved to a JSON file ({filename: "12345"}) after every entry, so
you can stop at any time and continue later (already labeled images are
skipped).

This tool needs a screen (it opens a small window to show each image).

Example:
    python scripts/label_captchas.py --images data/captchas --labels data/captcha_labels.json
"""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt

EXPECTED_LENGTH = 5


def load_labels(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def label(images_dir, labels_path):
    labels = load_labels(labels_path)
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))

    plt.ion()  # non-blocking display so we can read input() from the terminal
    figure = plt.figure(figsize=(4, 2))

    try:
        for fname in files:
            if fname in labels:  # already labeled -> skip
                continue

            img = cv2.imread(os.path.join(images_dir, fname))
            if img is None:
                continue

            figure.clear()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(fname)
            plt.axis("off")
            plt.pause(0.1)  # render the window

            answer = input(f"Digits for {fname} (Enter to skip, q to quit): ").strip()
            if answer.lower() == "q":
                break
            if not answer:
                continue
            if len(answer) != EXPECTED_LENGTH or not answer.isdigit():
                print(f"  '{answer}' is not {EXPECTED_LENGTH} digits, skipping.")
                continue

            labels[fname] = answer
            save_labels(labels_path, labels)
    finally:
        plt.close(figure)

    print(f"Saved {len(labels)} labels to {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Manually label CAPTCHA images.")
    parser.add_argument("--images", default="data/captchas",
                        help="Folder with the CAPTCHA PNG files.")
    parser.add_argument("--labels", default="data/captcha_labels.json",
                        help="JSON file to store the labels in.")
    args = parser.parse_args()
    label(args.images, args.labels)


if __name__ == "__main__":
    main()
