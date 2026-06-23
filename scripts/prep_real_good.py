"""Copy MVTec good-tile images and write empty YOLO label files for them.

Empty label file = background image in Ultralytics training (no objects to detect).
Adding 230 real "good" images teaches the model what clean tile looks like,
suppressing false-positive crack predictions on real-domain textures.

Usage (from repo root):
    python scripts/prep_real_good.py

Outputs:
    data/real/images/good/    230 copied images
    data/real/labels/good/    230 empty .txt files
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path as _Path

SOURCE_DIR = _Path("tile/train/good")
IMAGE_OUT  = _Path("data/real/images/good")
LABEL_OUT  = _Path("data/real/labels/good")


def main() -> None:
    if not SOURCE_DIR.exists():
        sys.exit(f"[error] source dir not found: {SOURCE_DIR}. Run from repo root.")

    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    LABEL_OUT.mkdir(parents=True, exist_ok=True)

    images = sorted(SOURCE_DIR.glob("*.png"))
    print(f"[prep_real_good] {len(images)} images → {IMAGE_OUT}")
    print(f"                  empty labels  → {LABEL_OUT}\n")

    for img_path in images:
        shutil.copy2(img_path, IMAGE_OUT / img_path.name)
        (LABEL_OUT / img_path.with_suffix(".txt").name).write_text("")

    print(f"[done] {len(images)} background images prepared.")
    print("       Next: add data/real/images/good to finetune.yaml train list.")


if __name__ == "__main__":
    main()
