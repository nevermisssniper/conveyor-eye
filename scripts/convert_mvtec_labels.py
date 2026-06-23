"""Convert MVTec AD Tile crack masks → YOLO bounding box labels.

One-off script for fine-tuning data prep (Step 1).

MVTec masks are binary PNGs (0 / 255). Each defect region becomes one YOLO
label line: <class_id> <cx> <cy> <w> <h> (all normalized 0–1).

Class id 1 = crack (matches conveyoreye CLASS_NAMES order).

Usage (from repo root):
    python scripts/convert_mvtec_labels.py

Outputs:
    data/real/images/crack/        symlinked/copied images (source for finetune dataset)
    data/real/labels/crack/<stem>.txt   one label file per image
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import shutil

import cv2
import numpy as np

CRACK_CLASS_ID = 1          # conveyoreye CLASS_NAMES[1] == "crack"
MIN_AREA_PX    = 100        # ignore components smaller than this (noise)

MASK_DIR   = _Path("tile/ground_truth/crack")
IMAGE_DIR  = _Path("tile/test/crack")
LABEL_DIR  = _Path("data/real/labels/crack")
IMAGE_OUT  = _Path("data/real/images/crack")   # copy images here for dataset.yaml


def mask_to_boxes(mask_path: _Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Return list of (cx, cy, w, h) normalized boxes from a binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # connectedComponentsWithStats: label 0 is background, 1..n-1 are regions
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for label in range(1, n):          # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_AREA_PX:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        boxes.append((cx, cy, wn, hn))

    return boxes


def main() -> None:
    if not MASK_DIR.exists():
        sys.exit(f"[error] mask dir not found: {MASK_DIR}. Run from repo root.")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)

    masks = sorted(MASK_DIR.glob("*_mask.png"))
    print(f"[convert] {len(masks)} masks → {LABEL_DIR}")
    print(f"[copy]    images   → {IMAGE_OUT}\n")

    total_boxes = 0
    for mask_path in masks:
        stem = mask_path.stem.replace("_mask", "")           # e.g. 000
        img_path = IMAGE_DIR / f"{stem}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] image not found for {mask_path.name}, skipping")
            continue
        img_h, img_w = img.shape[:2]

        boxes = mask_to_boxes(mask_path, img_w, img_h)

        # write label
        label_path = LABEL_DIR / f"{stem}.txt"
        with open(label_path, "w") as f:
            for cx, cy, w, h in boxes:
                f.write(f"{CRACK_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # copy image alongside label (tile/ is read-only)
        shutil.copy2(img_path, IMAGE_OUT / f"{stem}.png")

        total_boxes += len(boxes)
        box_str = "  ".join(f"[{cx:.3f} {cy:.3f} {w:.3f} {h:.3f}]" for cx, cy, w, h in boxes)
        print(f"  {stem}.png  →  {len(boxes)} box(es)   {box_str}")

    print(f"\n[done] {total_boxes} boxes / {len(masks)} images")
    print(f"       labels → {LABEL_DIR}")
    print(f"       images → {IMAGE_OUT}")


if __name__ == "__main__":
    main()
