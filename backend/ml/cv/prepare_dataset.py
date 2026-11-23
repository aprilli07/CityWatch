"""Turn the raw Urban Issues Dataset (Kaggle: akinduhiman/urban-issues-dataset)
into a clean, de-duplicated, 224x224 classification dataset using OpenCV.

Steps (all via ml/cv/preprocess.py, the same code the API uses at inference):
  1. decode + reject unreadable / tiny / extremely blurry images
  2. centre-crop to square and resize (INTER_AREA)
  3. drop near-duplicate images (64-bit perceptual hash, Hamming distance <= 6); the source
     dataset has many augmented copies, and a copy of a training image in the
     test split would inflate accuracy
  4. re-split per class into train / val / test (75 / 10 / 15) by hash group

Usage: python -m ml.cv.prepare_dataset RAW_DIR [--out data/cv] [--max-per-class 3000]
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
)

# Raw folder name (lower-cased prefix) -> class id used everywhere in CityWatch
CLASS_MAP = {
    "damaged road": "damaged_roads",
    "pothole": "potholes",
    "illegal parking": "illegal_parking",
    "broken road sign": "broken_road_signs",
    "fallen tree": "fallen_trees",
    "garbage": "garbage",
    "litter": "garbage",
    "vandalism": "graffiti",
    "graffit": "graffiti",
    "dead animal": "dead_animals",
    "damaged concrete": "damaged_concrete",
    "damaged electric": "damaged_electric_poles",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = (("train", 0.75), ("val", 0.10), ("test", 0.15))


def class_for(path: Path, root: Path) -> str | None:
    top = path.relative_to(root).parts[0].lower()
    for prefix, cls in CLASS_MAP.items():
        if top.startswith(prefix):
            return cls
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "ml" / "data" / "cv")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files: dict[str, list[Path]] = defaultdict(list)
    for p in args.raw_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT and "labels" not in p.parts:
            cls = class_for(p, args.raw_dir)
            if cls:
                files[cls].append(p)

    stats = {"raw": Counter(), "rejected": Counter(), "duplicates": Counter(), "kept": {}}
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
            if is_near_duplicate(h, hashes):
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
