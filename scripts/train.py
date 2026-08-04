"""Step 3 - Segment digits and train the CNN.

This takes the labeled CAPTCHAs (from label_captchas.py), splits each one into
its 5 digits, pairs every digit image with the matching character of its label,
and trains a small CNN to recognize single digits (0-9). The trained model is
saved so the solver / login SDK can use it.

Only CAPTCHAs that cleanly segment into exactly 5 digits are used for training;
the rest are counted and reported so you know how much data was skipped.

The digit preprocessing here (resize to 28x28, scale to [0, 1]) is kept
identical to captcha_reader/solver.py, so training and inference match.

Example:
    python scripts/train.py --images data/captchas \
        --labels data/captcha_labels.json --out models/digit_cnn.keras
"""

import argparse
import json
import os
import pathlib
import sys

import cv2
import numpy as np

# Make the captcha_reader package importable when run as a plain script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from captcha_reader.segmentation import binarize, split_digits  # noqa: E402

DIGIT_SIZE = 28


def build_dataset(images_dir, labels_path):
    """Turn labeled CAPTCHAs into (digit_image, digit_label) training pairs."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    x, y = [], []
    used, skipped = 0, 0
    for fname, code in labels.items():
        code = str(code).strip()
        if len(code) != 5 or not code.isdigit():
            skipped += 1
            continue

        img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        digits = split_digits(binarize(img))
        if digits is None:  # not exactly 5 clean digits -> cannot align labels
            skipped += 1
            continue

        # digits are left to right, so digit i matches character i of the code.
        for digit_img, digit_char in zip(digits, code):
            resized = cv2.resize(digit_img, (DIGIT_SIZE, DIGIT_SIZE),
                                 interpolation=cv2.INTER_AREA)
            x.append(resized.astype("float32") / 255.0)
            y.append(int(digit_char))
        used += 1

    print(f"Used {used} CAPTCHAs, skipped {skipped}.")
    x = np.array(x).reshape(-1, DIGIT_SIZE, DIGIT_SIZE, 1)
    y = np.array(y)
    print(f"Digit samples: {x.shape[0]}")
    return x, y


def build_model():
    """A small CNN, matching the shipped digit_cnn_light.keras design."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    augment = models.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    model = models.Sequential([
        layers.Input(shape=(DIGIT_SIZE, DIGIT_SIZE, 1)),
        augment,  # only active during training
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train(images_dir, labels_path, out_path, epochs):
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    x, y = build_dataset(images_dir, labels_path)
    if len(x) == 0:
        raise SystemExit("No training data. Collect and label some CAPTCHAs first.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]
    model.fit(x_train, y_train, validation_split=0.15,
              epochs=epochs, batch_size=32, callbacks=callbacks)

    _, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    print(f"Saved model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the digit CNN from labeled CAPTCHAs.")
    parser.add_argument("--images", default="data/captchas",
                        help="Folder with the CAPTCHA PNG files.")
    parser.add_argument("--labels", default="data/captcha_labels.json",
                        help="JSON file with {filename: '12345'} labels.")
    parser.add_argument("--out", default="models/digit_cnn.keras",
                        help="Where to save the trained model. Note: the shipped "
                             "sample is models/digit_cnn_light.keras.")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train(args.images, args.labels, args.out, args.epochs)


if __name__ == "__main__":
    main()
