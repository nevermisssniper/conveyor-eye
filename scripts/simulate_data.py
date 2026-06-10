"""Generate a synthetic YOLO dataset + dataset.yaml for Ultralytics.

Run order: this is step 2 (after `pip install -e ".[dev]"`).

    python scripts/simulate_data.py --n-train 2000 --n-val 400

Writes the standard Ultralytics layout::

    data/raw/
      images/{train,val}/*.png
      labels/{train,val}/*.txt
      dataset.yaml

WHY a script, not a notebook? Data generation must be reproducible and
parameterizable from the command line so a CI job or a teammate reproduces the
exact set with `--seed`. The simulator stays a pure library; this script owns IO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from conveyoreye import CLASS_NAMES
from conveyoreye.data.simulator import ConveyorSimulator


def _write_split(out_root: Path, split: str, n: int, img_size: int, seed: int) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    sim = ConveyorSimulator(img_size=img_size, seed=seed)
    for i in range(n):
        sample = sim.generate()
        stem = f"{split}_{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), sample.image)
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(b.to_yolo_line() for b in sample.boxes)
        )
    print(f"  {split}: wrote {n} images -> {img_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic ConveyorEye dataset.")
    ap.add_argument("--out", default="data/raw", help="Output root.")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    print(f"Generating synthetic dataset under {out_root.resolve()}")
    # Different seeds per split so train/val never share an identical RNG stream.
    _write_split(out_root, "train", args.n_train, args.img_size, seed=args.seed)
    _write_split(out_root, "val", args.n_val, args.img_size, seed=args.seed + 10_000)

    # dataset.yaml is what Ultralytics' trainer consumes (paths + class names).
    dataset_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(out_root / "dataset.yaml", "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)
    print(f"Wrote {out_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
