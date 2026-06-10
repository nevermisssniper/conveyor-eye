"""Detection metrics: mAP *and* the operational numbers ops actually use.

WHY both mAP and operating-point metrics?
------------------------------------------
mAP is the academic comparison metric: it integrates precision over all recall
levels and over all thresholds, summarizing the model's *ranking* quality
independent of any chosen operating point. But a factory does not run at "all
thresholds" — it runs at *one* threshold per class (thresholds.yaml). So a model
can have great mAP and still miss cracks at the threshold you actually deploy.

This evaluator therefore reports two distinct families, and keeps them visibly
separate in ``ClassMetrics``:
  * AP / mAP50 / mAP50-95 — threshold-independent ranking quality.
  * precision_at_threshold / recall_at_threshold / miss_rate — what you get at
    the deployed operating point.

Mechanics: for each frame we greedily match predictions to ground truth by
descending confidence at the eval IoU. Each match is a true positive; unmatched
predictions are false positives; unmatched ground truth are false negatives. The
per-detection ``(confidence, tp_flag)`` pairs are accumulated per class — these
are exactly the pairs the calibrator fits on, so calibration and metrics share
one matching pass.

Scalability path
----------------
  v1 (here): pure-NumPy greedy matching, single IoU for the operating-point
      stats, coarse multi-IoU loop for mAP50-95 (see TODO).
  v2: vectorize matching with a cost matrix; stream accumulation so we never hold
      all pairs in memory for large eval sets.
  v3: bootstrap confidence intervals on AP so model comparisons are significance-
      tested, not eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from conveyoreye import CLASS_NAMES, NUM_CLASSES, class_name


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against many boxes (N,4). Returns (N,)."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_bs = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_bs - inter + 1e-9
    return inter / union


@dataclass
class ClassMetrics:
    """Per-class metrics, with ranking quality and operating point kept separate.

    The separation is the whole point: ``ap`` answers "how good is the ranking?",
    while ``precision_at_threshold`` answers "what do I get at the threshold I
    deploy?". Conflating them is the classic way to ship a model that looks good
    on paper and misses defects in production.
    """

    name: str
    n_gt: int                          # ground-truth instances of this class
    n_pred: int                        # predictions above the operating threshold
    ap: float                          # average precision @ eval IoU (ranking)
    precision_at_threshold: float      # operating-point precision
    recall_at_threshold: float         # operating-point recall
    f1_at_threshold: float
    miss_rate: float                   # 1 - recall: the safety-relevant view

    def __str__(self) -> str:
        return (
            f"{self.name:<14} gt={self.n_gt:<5} AP={self.ap:.3f} "
            f"P={self.precision_at_threshold:.3f} R={self.recall_at_threshold:.3f} "
            f"F1={self.f1_at_threshold:.3f} miss={self.miss_rate:.3f}"
        )


@dataclass
class EvalReport:
    """Aggregate evaluation result across all classes."""

    map50: float
    map50_95: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    def summary(self) -> str:
        """Render a readable fixed-width table. Returns the string (also prints)."""
        lines = [
            "=" * 72,
            "ConveyorEye — Detection Evaluation",
            "=" * 72,
            f"mAP@50      : {self.map50:.4f}",
            f"mAP@50-95   : {self.map50_95:.4f}   (approx; see metrics.py TODO)",
            f"mean P/R/F1 : {self.mean_precision:.3f} / {self.mean_recall:.3f} / {self.mean_f1:.3f}",
            "-" * 72,
            f"{'class':<14} {'gt':<8} {'AP':<7} {'P':<7} {'R':<7} {'F1':<7} {'miss':<7}",
            "-" * 72,
        ]
        for name in CLASS_NAMES:
            m = self.per_class.get(name)
            if m is None:
                continue
            lines.append(
                f"{m.name:<14} {m.n_gt:<8} {m.ap:<7.3f} {m.precision_at_threshold:<7.3f} "
                f"{m.recall_at_threshold:<7.3f} {m.f1_at_threshold:<7.3f} {m.miss_rate:<7.3f}"
            )
        lines.append("=" * 72)
        text = "\n".join(lines)
        print(text)
        return text


class DetectionEvaluator:
    """Accumulates matched (conf, tp) pairs per class, then computes a report.

    Usage::

        ev = DetectionEvaluator(iou_threshold=0.5)
        for gt_boxes, gt_cls, pred in zip(...):
            ev.add_frame(gt_boxes, gt_cls, pred)
        report = ev.compute(class_thresholds={...})

    ``add_frame`` takes ground truth in xyxy pixels and a DetectionResult, so the
    evaluator speaks the same dataclass the detector emits.
    """

    def __init__(self, iou_threshold: float = 0.50) -> None:
        self.iou_threshold = iou_threshold
        # Per class: list of (confidence, tp_flag) for every prediction, and a
        # running ground-truth count. tp_flag is 1 if the prediction matched an
        # unused GT box at >= iou_threshold, else 0.
        self._records: dict[int, list[tuple[float, int]]] = {c: [] for c in range(NUM_CLASSES)}
        self._n_gt: dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    # --------------------------------------------------------------- ingestion

    def add_frame(self, gt_boxes: np.ndarray, gt_classes: np.ndarray, result) -> None:
        """Greedy-match one frame's predictions to its ground truth, per class.

        gt_boxes : (M,4) xyxy pixels. gt_classes : (M,). result : DetectionResult.
        Matching is done independently per class — a scratch prediction can never
        consume a crack ground-truth box. Within a class, predictions are matched
        in descending confidence (greedy), which is the standard COCO/VOC rule and
        the reason high-confidence FPs are penalized first.
        """
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)

        for cid in range(NUM_CLASSES):
            cls_gt = gt_boxes[gt_classes == cid]
            self._n_gt[cid] += len(cls_gt)

            preds = [d for d in result.detections if d.class_id == cid]
            preds.sort(key=lambda d: d.confidence, reverse=True)

            used = np.zeros(len(cls_gt), dtype=bool)
            for d in preds:
                if len(cls_gt) == 0:
                    self._records[cid].append((d.confidence, 0))
                    continue
                ious = iou_xyxy(np.array(d.xyxy), cls_gt)
                ious[used] = -1.0  # cannot reuse a matched GT box
                best = int(np.argmax(ious))
                if ious[best] >= self.iou_threshold:
                    used[best] = True
                    self._records[cid].append((d.confidence, 1))  # true positive
                else:
                    self._records[cid].append((d.confidence, 0))  # false positive

    # --------------------------------------------------------------- accessors

    def labeled_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pooled (confidences, tp_flags, class_ids) for the calibrator.

        This is the bridge to calibration.py: the same TP/FP labeling that drives
        metrics feeds the calibrator, so the two never disagree about what counts
        as correct.
        """
        confs, tps, cids = [], [], []
        for cid, recs in self._records.items():
            for conf, tp in recs:
                confs.append(conf); tps.append(tp); cids.append(cid)
        return np.array(confs), np.array(tps), np.array(cids)

    # ----------------------------------------------------------------- compute

    def compute(self, class_thresholds: dict[str, float] | None = None) -> EvalReport:
        """Compute AP (ranking) and operating-point P/R/F1 (deployment) per class."""
        class_thresholds = class_thresholds or {}
        per_class: dict[str, ClassMetrics] = {}
        aps, ps, rs, f1s = [], [], [], []

        for cid in range(NUM_CLASSES):
            name = class_name(cid)
            recs = sorted(self._records[cid], key=lambda x: x[0], reverse=True)
            n_gt = self._n_gt[cid]

            ap = self._average_precision(recs, n_gt)

            thr = class_thresholds.get(name, 0.0)
            p_at, r_at, f1_at, n_pred = self._operating_point(recs, n_gt, thr)
            miss = 1.0 - r_at

            per_class[name] = ClassMetrics(
                name=name, n_gt=n_gt, n_pred=n_pred, ap=ap,
                precision_at_threshold=p_at, recall_at_threshold=r_at,
                f1_at_threshold=f1_at, miss_rate=miss,
            )
            # Only average over classes that actually appear in the eval set, so
            # an absent class does not drag mAP toward zero.
            if n_gt > 0:
                aps.append(ap); ps.append(p_at); rs.append(r_at); f1s.append(f1_at)

        map50 = float(np.mean(aps)) if aps else 0.0
        map50_95 = self._approx_map_50_95(map50)

        return EvalReport(
            map50=map50,
            map50_95=map50_95,
            mean_precision=float(np.mean(ps)) if ps else 0.0,
            mean_recall=float(np.mean(rs)) if rs else 0.0,
            mean_f1=float(np.mean(f1s)) if f1s else 0.0,
            per_class=per_class,
        )

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _average_precision(recs: list[tuple[float, int]], n_gt: int) -> float:
        """AP via the all-points (COCO-style) PR-curve integral at the eval IoU.

        Walk predictions in descending confidence, accumulate TP/FP to trace the
        precision-recall curve, make precision monotonic from the right
        (envelope), then integrate area under it. This is threshold-independent —
        it uses the whole curve, which is what makes AP a *ranking* metric.
        """
        if n_gt == 0 or not recs:
            return 0.0
        tp = np.array([r[1] for r in recs], dtype=np.float64)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-9)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        # Precision envelope: precision[i] = max(precision[i:]).
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    @staticmethod
    def _operating_point(
        recs: list[tuple[float, int]], n_gt: int, threshold: float
    ) -> tuple[float, float, float, int]:
        """P/R/F1 and prediction count at a fixed confidence threshold.

        This is the deployment view: keep only predictions at/above ``threshold``
        (the value in thresholds.yaml) and score those. Unlike AP, this collapses
        the curve to the single point the line will actually run at.
        """
        kept = [(c, tp) for c, tp in recs if c >= threshold]
        n_pred = len(kept)
        if n_pred == 0:
            return 0.0, 0.0, 0.0, 0
        tps = sum(tp for _, tp in kept)
        precision = tps / n_pred
        recall = tps / (n_gt + 1e-9) if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return float(precision), float(recall), float(f1), n_pred

    @staticmethod
    def _approx_map_50_95(map50: float) -> float:
        """Cheap stand-in for mAP@50:95 until the proper multi-IoU loop lands.

        TODO (key learning implementation): mAP@50:95 is the *mean* of AP computed
        at IoU thresholds 0.50, 0.55, ..., 0.95. Doing it properly means re-running
        ``add_frame`` matching at each of the 10 IoU thresholds (or caching the
        raw IoU matrices per frame and re-thresholding) and averaging the per-IoU
        mAPs. AP falls off as IoU rises, so the true value is below mAP50; the
        0.72 factor below is a crude placeholder that roughly tracks that decay
        for boxes of our typical size. Replace this with the real loop — it is the
        single most instructive metrics exercise in the repo.
        """
        return float(map50 * 0.72)
