"""Full evaluation pipeline: metrics + calibration fit + threshold export.

Run order: step 4.

    python scripts/evaluate.py --weights runs/conveyoreye/weights/best.pt \
        --data data/raw/dataset.yaml

What this does, end to end:
  1. Run the detector over the val split (collecting raw, low-threshold preds).
  2. Greedy-match preds to ground truth in the DetectionEvaluator -> EvalReport.
  3. Categorize every error in the FailureTaxonomy -> actionable recommendations.
  4. Fit the ConfidenceCalibrator on the matched (conf, tp) pairs and report the
     ECE before/after, then pickle it for the serving layer.
  5. Sweep PR curves and export recommended per-class thresholds back to YAML.

This script is the project's "did it actually work, and at what operating point
should we run it" answer — it produces the two artifacts serving needs
(calibrator.pkl, thresholds.yaml) plus a human-readable report.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import yaml

from conveyoreye import CLASS_NAMES, class_id
from conveyoreye.eval.failure_taxonomy import FailureTaxonomy
from conveyoreye.eval.metrics import DetectionEvaluator
from conveyoreye.eval.threshold import ThresholdSweeper
from conveyoreye.model.calibration import ConfidenceCalibrator


def _read_gt(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file -> (xyxy pixel boxes, class ids)."""
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    boxes, classes = [], []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = (float(v) for v in line.split())
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2]); classes.append(int(c))
    return np.array(boxes).reshape(-1, 4), np.array(classes, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ConveyorEye + fit calibration + export thresholds.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset.yaml (uses its val split).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--calib-method", default="temperature",
                    choices=["temperature", "platt", "isotonic"])
    ap.add_argument("--calibrator-out", default="calibrator.pkl")
    ap.add_argument("--thresholds-out", default="configs/thresholds.generated.yaml")
    args = ap.parse_args()

    from conveyoreye.model.detector import Detector

    data_cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(data_cfg["path"])
    val_images = sorted((root / data_cfg["val"]).glob("*.png"))
    labels_dir = root / "labels" / "val"

    # Run the detector permissively (low threshold) so the evaluator sees the full
    # PR curve, not just boxes above the deployed operating point.
    detector = Detector(
        weights=args.weights, device=args.device, iou_threshold=args.iou,
        class_thresholds={}, default_threshold=0.001, warmup=False,
    )

    evaluator = DetectionEvaluator(iou_threshold=args.iou)
    taxonomy = FailureTaxonomy(iou_threshold=args.iou)

    print(f"Evaluating {len(val_images)} val frames...")
    for img_path in val_images:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        gt_boxes, gt_classes = _read_gt(labels_dir / f"{img_path.stem}.txt", w, h)
        result = detector.predict(image, frame_id=img_path.stem)
        evaluator.add_frame(gt_boxes, gt_classes, result)
        taxonomy.add_frame(gt_boxes, gt_classes, result, image=image)

    # --- 1. Metrics report at the *current* thresholds.yaml operating points ----
    cfg = yaml.safe_load(Path("configs/thresholds.yaml").read_text()) if Path(
        "configs/thresholds.yaml").exists() else {"classes": {}}
    cur_thresholds = {n: c.get("threshold", 0.5) for n, c in cfg.get("classes", {}).items()}
    report = evaluator.compute(class_thresholds=cur_thresholds)
    report.summary()

    # --- 2. Failure taxonomy + recommendations ---------------------------------
    print("\nFailure taxonomy:", taxonomy.summarize())
    print("\nActionable recommendations:")
    for i, rec in enumerate(taxonomy.actionable_recommendations(), 1):
        print(f"  {i}. {rec}")

    # --- 3. Fit + serialize the calibrator -------------------------------------
    confs, tps, cids = evaluator.labeled_pairs()
    calibrator = ConfidenceCalibrator(method=args.calib_method)
    if confs.size:
        pre = calibrator.reliability_diagram(confs, tps).ece
        calibrator.fit(confs, tps, cids)
        post = calibrator.reliability_diagram(
            calibrator.calibrate_array(cids, confs), tps
        ).ece
        print(f"\nCalibration ({args.calib_method}): ECE {pre:.4f} -> {post:.4f}")
        with open(args.calibrator_out, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"Wrote calibrator -> {args.calibrator_out}")

    # --- 4. Sweep + export recommended thresholds ------------------------------
    # Strategy mirrors thresholds.yaml's business rationale: crack=min_recall,
    # missing_part=min_precision, the rest=max_f1.
    strategy = {
        "scratch": ("max_f1", None),
        "crack": ("min_recall", 0.97),
        "dent": ("max_f1", None),
        "discoloration": ("min_precision", 0.80),
        "missing_part": ("min_precision", 0.95),
    }
    sweeper = ThresholdSweeper.from_evaluator(evaluator)
    recs = sweeper.recommend(strategy)
    print("\nRecommended thresholds:")
    for name in CLASS_NAMES:
        r = recs.get(name)
        if r:
            flag = "" if r.target_met else "  [TARGET NOT MET]"
            print(f"  {name:<14} thr={r.threshold:.3f} P={r.achieved_precision:.3f} "
                  f"R={r.achieved_recall:.3f}{flag}")
    sweeper.export_yaml(recs, args.thresholds_out, iou_threshold=args.iou,
                        active_learning=cfg.get("active_learning"))
    print(f"Wrote thresholds -> {args.thresholds_out}")


if __name__ == "__main__":
    main()
