"""Batch inference on MVTec AD Tile — Option A evaluation (Step 1).

Split logic (Option A — OOD-as-false-positive):
  tile/test/crack/        → split "crack"   real positives; only class in our taxonomy
  tile/test/good/         → split "good"    true negatives; model should fire nothing
  tile/test/glue_strip/   → split "ood"     OOD defects; model should ideally fire nothing
  tile/test/gray_stroke/  →   "              but will reveal what the model hallucinates
  tile/test/oil/          →   "
  tile/test/rough/        →   "

Outputs:
  results/mvtec_eval.csv          per-image row with all predictions
  results/mvtec_summary.txt       printed to stdout and written to file

Usage:
    python scripts/eval_mvtec.py \
        --weights runs/conveyoreye/weights/best.pt \
        --calibrator calibrator.pkl \
        --thresholds configs/thresholds.generated.yaml \
        --tile-dir tile/test
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

CRACK_CATEGORY = "crack"
GOOD_CATEGORY = "good"
OOD_CATEGORIES = {"glue_strip", "gray_stroke", "oil", "rough"}


def category_to_split(cat: str) -> str:
    if cat == GOOD_CATEGORY:
        return "good"
    if cat == CRACK_CATEGORY:
        return "crack"
    return "ood"


def load_detector(weights: str, calibrator_path: str | None, thresholds_path: str, device: str):
    """Build a Detector with calibrator and per-class thresholds from generated yaml."""
    from conveyoreye.model.detector import Detector

    calib = None
    if calibrator_path and Path(calibrator_path).exists():
        with open(calibrator_path, "rb") as f:
            calib = pickle.load(f)
        print(f"[setup] Loaded calibrator: {calibrator_path}")
    else:
        print(f"[setup] No calibrator found at {calibrator_path!r} — running uncalibrated")

    thr_cfg = {}
    if Path(thresholds_path).exists():
        raw = yaml.safe_load(Path(thresholds_path).read_text())
        thr_cfg = {name: cls["threshold"] for name, cls in raw.get("classes", {}).items()}
        print(f"[setup] Thresholds: {thr_cfg}")
    else:
        print(f"[setup] No thresholds at {thresholds_path!r} — using default 0.25")

    det = Detector(
        weights=weights,
        device=device,
        iou_threshold=0.50,
        class_thresholds=thr_cfg,
        default_threshold=0.25,
        calibrator=calib,
        warmup=True,
    )
    print(f"[setup] Detector ready (device={device})\n")
    return det


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="MVTec AD Tile — Option A batch eval")
    ap.add_argument("--weights", default="runs/conveyoreye/weights/best.pt")
    ap.add_argument("--calibrator", default="calibrator.pkl")
    ap.add_argument("--thresholds", default="configs/thresholds.generated.yaml")
    ap.add_argument("--tile-dir", default="tile/test",
                    help="Path to tile/test/ directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="results",
                    help="Directory for CSV and summary output")
    args = ap.parse_args()

    tile_root = Path(args.tile_dir)
    if not tile_root.exists():
        sys.exit(f"[error] tile-dir not found: {tile_root}. Run from repo root.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "mvtec_eval.csv"
    summary_path = out_dir / "mvtec_summary.txt"

    # ------------------------------------------------------------------ setup
    detector = load_detector(
        weights=args.weights,
        calibrator_path=args.calibrator,
        thresholds_path=args.thresholds,
        device=args.device,
    )

    # ------------------------------------------------------------------ collect images
    # Each subdirectory of tile/test/ is a defect category.
    images: list[tuple[str, str, Path]] = []  # (category, split, image_path)
    for cat_dir in sorted(tile_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        split = category_to_split(cat)
        for img_path in sorted(cat_dir.glob("*.png")):
            images.append((cat, split, img_path))

    print(f"[run] {len(images)} images across {len({c for c,_,_ in images})} categories\n")

    # ------------------------------------------------------------------ inference loop
    rows: list[dict] = []
    t_start = time.perf_counter()

    for i, (cat, split, img_path) in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[warn] could not read {img_path}, skipping")
            continue

        result = detector.predict(frame, frame_id=img_path.stem)
        dets = result.detections

        # Summarise per detection
        pred_classes = [d.class_name for d in dets]
        pred_confs   = [round(d.confidence, 4) for d in dets]
        pred_boxes   = [tuple(round(v, 1) for v in d.xyxy) for d in dets]

        rows.append({
            "filename":      img_path.name,
            "category":      cat,
            "split":         split,
            "num_dets":      len(dets),
            "pred_classes":  "|".join(pred_classes) if pred_classes else "",
            "pred_confs":    "|".join(str(c) for c in pred_confs) if pred_confs else "",
            "pred_boxes":    "|".join(str(b) for b in pred_boxes) if pred_boxes else "",
            "latency_ms":    round(result.latency_ms, 1),
        })

        # progress tick every 20 images
        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            elapsed = time.perf_counter() - t_start
            print(f"  {i+1}/{len(images)} images — {elapsed:.1f}s elapsed")

    # ------------------------------------------------------------------ write CSV
    if rows:
        fieldnames = ["filename", "category", "split", "num_dets",
                      "pred_classes", "pred_confs", "pred_boxes", "latency_ms"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[out] CSV -> {csv_path}  ({len(rows)} rows)")

    # ------------------------------------------------------------------ summary
    lines: list[str] = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("\n" + "="*60)
    p("MVTec AD Tile — Option A Evaluation Summary")
    p("="*60)

    # -- per split breakdown --
    for split_name in ("crack", "good", "ood"):
        split_rows = [r for r in rows if r["split"] == split_name]
        if not split_rows:
            continue

        fired = [r for r in split_rows if r["num_dets"] > 0]
        total = len(split_rows)
        fire_rate = len(fired) / total if total else 0.0

        p(f"\n[{split_name.upper()}]  {total} images")
        p(f"  Detection rate:  {len(fired)}/{total}  ({fire_rate:.1%})")

        if split_name == "good":
            p(f"  FPR:             {fire_rate:.1%}  (target <10%)")

        if split_name == "crack":
            p(f"  Crack detection rate (proxy recall): {fire_rate:.1%}")
            # What does it predict on crack images?
            crack_correct = [r for r in fired if "crack" in r["pred_classes"]]
            p(f"  Predicted 'crack' on crack images:   {len(crack_correct)}/{total}")

        # class distribution across all fired predictions
        class_counts: dict[str, int] = defaultdict(int)
        for r in split_rows:
            for cls in r["pred_classes"].split("|"):
                if cls:
                    class_counts[cls] += 1
        if class_counts:
            p("  Predicted classes:")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                p(f"    {cls:<16} {cnt:>4} detections")

    # -- OOD breakdown by category --
    ood_rows = [r for r in rows if r["split"] == "ood"]
    if ood_rows:
        p("\n[OOD] Per-category false positive breakdown:")
        for cat in sorted({r["category"] for r in ood_rows}):
            cat_rows = [r for r in ood_rows if r["category"] == cat]
            fired = [r for r in cat_rows if r["num_dets"] > 0]
            class_counts: dict[str, int] = defaultdict(int)
            for r in cat_rows:
                for cls in r["pred_classes"].split("|"):
                    if cls:
                        class_counts[cls] += 1
            cls_str = ", ".join(f"{k}×{v}" for k, v in sorted(class_counts.items(), key=lambda x: -x[1]))
            p(f"  {cat:<14}  {len(fired)}/{len(cat_rows)} fired    {cls_str or '(none)'}")

    # -- latency --
    lats = [r["latency_ms"] for r in rows if r["latency_ms"] > 0]
    if lats:
        p(f"\n[LATENCY]  mean={sum(lats)/len(lats):.0f}ms  "
          f"min={min(lats):.0f}ms  max={max(lats):.0f}ms")

    p("\n" + "="*60)

    # write summary to file
    summary_path.write_text("\n".join(lines))
    print(f"[out] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
