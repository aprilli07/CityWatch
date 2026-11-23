"""Turn the raw Urban Issues Dataset (Kaggle: akinduhiman/urban-issues-dataset)
into a clean, de-duplicated, 224x224 classification dataset using OpenCV.

Steps (all via ml/cv/preprocess.py, the same code the API uses at inference):
  1. decode + reject unreadable / tiny / extremely blurry images
  2. centre-crop to square and resize (INTER_AREA)
  3. drop near-duplicate images (64-bit perceptual hash, Hamming distance <= 6, checked
     against all 8 flip/rotation variants); the source
     dataset has many augmented copies, and a copy of a training image in the
     test split would inflate accuracy
  4. re-split per class into train / val / test (75 / 10 / 15)

Class comes from the YOLO label file (one folder mixes potholes and road cracks).
Images with no label, several classes, or a label foreign to their folder are skipped.

Usage: python -m ml.cv.prepare_dataset RAW_DIR [RAW_DIR ...] [--out ml/data/cv] [--max-per-class 3000]

The optional second dataset (Kaggle: atulyakumar98/pothole-detection-dataset)
adds a `no_issue` class of undamaged roads so the model can say "nothing wrong".
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.config import ARTIFACTS_DIR, BACKEND_DIR
from ml.cv.preprocess import (
    BLUR_THRESHOLD,
    ImageRejected,
    average_hash,
    blur_score,
    decode,
    is_near_duplicate,
    prepare,
    variant_hashes,
)

# YOLO class ids of the Urban Issues Dataset (config.yaml) -> CityWatch class names
ID_TO_CLASS = {
    0: "damaged_roads", 1: "potholes", 2: "illegal_parking", 3: "broken_road_signs", 4: "fallen_trees",
    5: "garbage", 6: "graffiti", 7: "dead_animals", 8: "damaged_concrete", 9: "damaged_electric_poles",
}
# Which label ids are trusted inside each top-level folder (lower-cased prefix). Ids outside the
# set are labelling noise (e.g. "pothole" boxes inside the Graffiti folder) and are dropped.
YOLO_FOLDERS = {
    "potholes and roadcracks": {0, 1},   # one folder holds both classes; the label decides
    "damaged concrete": {8},
    "damagedelectrical": {9},
    "damagedroadsigns": {3},
    "deadanimals": {7},
    "fallentrees": {4},
    "garbage": {5},
    "graffit": {6},
    "illegalparking": {2},
}
# Plain folder-per-class datasets (no label files); matched against any directory in the path.
FOLDER_CLASSES = {
    # atulyakumar98/pothole-detection-dataset
    "normal": "no_issue",
    "potholes": "potholes",
    # programmerrdai/road-issues-detection-dataset ("Damaged Road issues" mixes in speed bumps and
    # "Mixed Issues" has several classes per image, so both are left out)
    "illegal parking issues": "illegal_parking",
    "pothole issues": "potholes",
    "broken road sign issues": "broken_road_signs",
    "vandalism issues": "graffiti",
    "littering garbage": "garbage",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = (("train", 0.75), ("val", 0.10), ("test", 0.15))


def class_for(path: Path, root: Path, skipped: Counter) -> str | None:
    top = path.relative_to(root).parts[0].lower()
    for prefix, allowed in YOLO_FOLDERS.items():
        if top.startswith(prefix):
            label = path.parent.parent / "labels" / (path.stem + ".txt")
            ids = set()
            if label.exists():
                ids = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
            if len(ids) != 1:
                skipped["unlabelled_or_mixed"] += 1
                return None
            (cid,) = ids
            if cid not in allowed:
                skipped["foreign_label"] += 1
                return None
            return ID_TO_CLASS[cid]
    for part in path.relative_to(root).parts[:-1]:
        for prefix, cls in FOLDER_CLASSES.items():
            if part.lower().startswith(prefix):
                return cls
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dirs", type=Path, nargs="+", help="one or more raw dataset roots")
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "ml" / "data" / "cv")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files: dict[str, list[Path]] = defaultdict(list)
    skipped: Counter = Counter()
    for root in args.raw_dirs:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in IMG_EXT and "labels" not in p.parts:
                cls = class_for(p, root, skipped)
                if cls:
                    files[cls].append(p)
    print("Skipped:", dict(skipped))

    stats = {"raw": Counter(), "rejected": Counter(), "duplicates": Counter(), "kept": {},
             "skipped": dict(skipped)}
    for cls in sorted(files):
        paths = files[cls]
        rng.shuffle(paths)
        hashes = np.empty(0, dtype=np.uint64)
        kept: list[tuple[int, object]] = []
        for p in paths:
            stats["raw"][cls] += 1
            if len(kept) >= args.max_per_class:
                break
            try:
                img = decode(p.read_bytes())
                if blur_score(img) < BLUR_THRESHOLD:
                    raise ImageRejected("blurry")
                img = prepare(img)
            except ImageRejected:
                stats["rejected"][cls] += 1
                continue
            h = average_hash(img)
            if any(is_near_duplicate(v, hashes) for v in variant_hashes(img)):
                stats["duplicates"][cls] += 1
                continue
            hashes = np.append(hashes, np.uint64(h))
            kept.append((h, img))

        n = len(kept)
        bounds, start = [], 0
        for name, frac in SPLITS:
            end = n if name == "test" else start + int(round(frac * n))
            bounds.append((name, start, end))
            start = end
        counts = {}
        for name, a, b in bounds:
            d = args.out / name / cls
            d.mkdir(parents=True, exist_ok=True)
            for h, img in kept[a:b]:
                cv2.imwrite(str(d / f"{h:016x}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[name] = b - a
        stats["kept"][cls] = counts
        print(f"{cls:24s} raw={stats['raw'][cls]:5d} kept={n:5d} "
              f"rejected={stats['rejected'][cls]:4d} dupes={stats['duplicates'][cls]:4d} {counts}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "cv_dataset.json").write_text(json.dumps(stats, indent=2))
    print(f"{len(stats['kept'])} classes written to {args.out}")


if __name__ == "__main__":
    main()
