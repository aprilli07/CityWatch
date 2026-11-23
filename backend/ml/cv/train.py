"""Train the 10-class infrastructure-issue classifier (ImageNet MobileNetV2).

Two-stage transfer learning:
  1. train only the new classification head on the frozen backbone
  2. fine-tune the top of the backbone at a low learning rate
     (BatchNorm layers stay in inference mode so their statistics are kept)

Pixel scaling ([-1, 1]) and augmentation are layers inside the model, so the
saved model takes uint8 RGB 224x224 images straight from ml/cv/preprocess.py.

Usage: python -m ml.cv.train [--data ml/data/cv] [--head-epochs 5] [--ft-epochs 8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.config import ARTIFACTS_DIR, BACKEND_DIR, get_settings
from ml.cv.preprocess import IMG_SIZE

BATCH = 32


def build_model(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
    ], name="augment")
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # == mobilenet_v2.preprocess_input
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def class_weights(ds, n: int) -> dict[int, float]:
    counts = np.zeros(n)
    for _, y in ds.unbatch():
        counts[int(y)] += 1
    return {i: float(counts.sum() / (n * max(c, 1))) for i, c in enumerate(counts)}


def main() -> None:
    import os

    import certifi  # python.org macOS builds ship without CA certs; needed to fetch ImageNet weights

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=BACKEND_DIR / "ml" / "data" / "cv")
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=8)
    ap.add_argument("--unfreeze-from", type=int, default=100, help="backbone layer index to unfreeze from")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)

    def load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            args.data / split, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
            label_mode="int", shuffle=shuffle, seed=42,
        )

    train_ds, val_ds, test_ds = load("train", True), load("val", False), load("test", False)
    classes = train_ds.class_names
    n = len(classes)
    cw = class_weights(train_ds, n)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds, val_ds, test_ds = (d.prefetch(AUTOTUNE) for d in (train_ds, val_ds, test_ds))

    model, base = build_model(n)
    ckpt = ARTIFACTS_DIR / "cv_model.best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1),
    ]
    t0 = time.perf_counter()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=cw,
                   callbacks=callbacks)
    head_val_acc = max(h1.history["val_accuracy"])

    base.trainable = True
    for layer in base.layers[: args.unfreeze_from]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs, class_weight=cw,
                   callbacks=callbacks)
    train_s = time.perf_counter() - t0

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(1)
    top2 = float(np.mean([t in np.argsort(p)[-2:] for t, p in zip(y_true, probs, strict=False)]))
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    s = get_settings()
    model.save(s.cv_model_path)
    s.cv_labels_path.write_text(json.dumps({"classes": classes}, indent=2))
    ckpt.unlink(missing_ok=True)
    metrics = {
        "classes": classes,
        "test_images": int(len(y_true)),
        "test_accuracy": float(report["accuracy"]),
        "test_top2_accuracy": top2,
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "val_accuracy_head_only": float(head_val_acc),
        "val_accuracy_after_finetune": float(max(h2.history["val_accuracy"])),
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
        "confusion_matrix": cm.tolist(),
        "train_seconds": round(train_s, 1),
        "epochs": {"head": len(h1.history["loss"]), "finetune": len(h2.history["loss"])},
        "unfreeze_from_layer": args.unfreeze_from,
    }
    (ARTIFACTS_DIR / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(cm, classes, ARTIFACTS_DIR / "cv_confusion_matrix.png")
    print(f"Test accuracy {metrics['test_accuracy']:.1%}  macro-F1 {metrics['test_macro_f1']:.3f}  "
          f"top-2 {top2:.1%}  ({len(y_true)} test images)")


def _plot_confusion(cm, classes, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CityWatch classifier - test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
