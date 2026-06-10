"""Failure-mode taxonomy: turn FP/FN counts into a debugging plan.

WHY a taxonomy instead of a single error number?
-------------------------------------------------
"mAP went down" tells you *that* the model is worse, not *why*. Detection errors
are not interchangeable: a flood of background false positives (model fires on
belt seams), duplicate boxes (NMS too loose), wrong-class confusions (scratch vs
crack), or systematic misses of small/occluded/low-contrast defects each demand a
*different* fix. This module classifies every error into a typed category and
then maps each category to a concrete next action — so evaluation output is a
worklist, not a verdict.

The categories are deliberately the ones that map to levers we control in this
repo: occlusion/low-contrast point back at augmentation.yaml, duplicates at the
NMS IoU in detector.py, missed-rare at the WeightedRandomSampler and the active-
learning queue.

Scalability path
----------------
  v1 (here): rule-based categorization from geometry + class of each error.
  v2: cluster FP image crops with the detector's own embeddings to discover
      *unlabeled* failure modes (e.g. a new contaminant) rather than only the
      pre-named ones.
  v3: tie each failure type to a slice-based alert in monitoring so regressions
      in a specific failure mode page someone, not just aggregate mAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from conveyoreye import class_name
from conveyoreye.eval.metrics import iou_xyxy


class FPType(str, Enum):
    """Why a prediction was a false positive."""

    BACKGROUND = "background"      # no GT nearby at all (fired on texture/seam)
    DUPLICATE = "duplicate"        # overlaps a GT already matched (NMS leak)
    WRONG_CLASS = "wrong_class"    # localizes a real defect, wrong label
    POOR_LOCALIZATION = "poor_localization"  # right class, IoU just below thresh


class FNType(str, Enum):
    """Why a ground-truth box was missed (false negative)."""

    MISSED = "missed"              # nothing predicted near it
    LOW_CONTRAST = "low_contrast"  # faint defect (discoloration-like) — pre-proc lever
    OCCLUSION = "occlusion"        # partially covered (small visible area)
    SIZE = "size"                  # very small box — scale sensitivity
    SUPPRESSED = "suppressed"      # a pred existed but lost to NMS / threshold


@dataclass
class FailureRecord:
    """One categorized error."""

    kind: str                      # "FP" or "FN"
    category: str                  # FPType or FNType value
    class_name: str
    confidence: float | None = None
    iou_with_gt: float | None = None


@dataclass
class TaxonomySummary:
    """Counts of failures broken down two ways: by type and by class."""

    by_type: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    total_fp: int = 0
    total_fn: int = 0

    def __str__(self) -> str:
        types = ", ".join(f"{k}={v}" for k, v in sorted(self.by_type.items()))
        return f"FP={self.total_fp} FN={self.total_fn} | {types}"


class FailureTaxonomy:
    """Accumulates and categorizes detection failures across frames.

    Feed it the same (gt, pred) pairs as the evaluator. It does its own light
    matching because it needs *near-misses* (IoU below the match threshold but
    above zero) that the evaluator discards — those near-misses are exactly the
    poor-localization / wrong-class signal.
    """

    def __init__(
        self,
        iou_threshold: float = 0.50,
        low_contrast_std: float = 18.0,   # px-intensity std below this -> low-contrast
        small_area_frac: float = 0.01,    # box area < this frac of image -> "small"
        occlusion_area_frac: float = 0.02,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.low_contrast_std = low_contrast_std
        self.small_area_frac = small_area_frac
        self.occlusion_area_frac = occlusion_area_frac
        self.records: list[FailureRecord] = []

    def add_frame(
        self,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        result,
        image: np.ndarray | None = None,
    ) -> None:
        """Categorize FPs and FNs for one frame.

        ``image`` is optional but enables the appearance-based FN categories
        (low_contrast via local intensity std). Without it those collapse to a
        geometric best-guess.
        """
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gt_classes = np.asarray(gt_classes, dtype=np.int64).reshape(-1)
        img_area = float(image.shape[0] * image.shape[1]) if image is not None else None

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        preds = sorted(result.detections, key=lambda d: d.confidence, reverse=True)

        # ---- classify each prediction as TP or a typed FP -------------------
        for d in preds:
            pbox = np.array(d.xyxy)
            if len(gt_boxes) == 0:
                self.records.append(
                    FailureRecord("FP", FPType.BACKGROUND.value, d.class_name, d.confidence, 0.0)
                )
                continue
            ious = iou_xyxy(pbox, gt_boxes)
            best = int(np.argmax(ious))
            best_iou = float(ious[best])
            same_class = gt_classes[best] == d.class_id

            if best_iou >= self.iou_threshold and same_class and not matched_gt[best]:
                matched_gt[best] = True            # true positive, no record
            elif best_iou >= self.iou_threshold and same_class and matched_gt[best]:
                self.records.append(
                    FailureRecord("FP", FPType.DUPLICATE.value, d.class_name, d.confidence, best_iou)
                )
            elif best_iou >= self.iou_threshold and not same_class:
                self.records.append(
                    FailureRecord("FP", FPType.WRONG_CLASS.value, d.class_name, d.confidence, best_iou)
                )
            elif 0.1 <= best_iou < self.iou_threshold and same_class:
                self.records.append(
                    FailureRecord("FP", FPType.POOR_LOCALIZATION.value, d.class_name, d.confidence, best_iou)
                )
            else:
                self.records.append(
                    FailureRecord("FP", FPType.BACKGROUND.value, d.class_name, d.confidence, best_iou)
                )

        # ---- classify each unmatched GT box as a typed FN -------------------
        for i, (box, cid) in enumerate(zip(gt_boxes, gt_classes)):
            if matched_gt[i]:
                continue
            category = self._categorize_fn(box, result, image, img_area)
            self.records.append(
                FailureRecord("FN", category, class_name(int(cid)), None, None)
            )

    def _categorize_fn(self, box, result, image, img_area) -> str:
        """Pick the most informative FN category for one missed GT box."""
        # Was there a prediction near it that got suppressed/thresholded out?
        if result.detections:
            ious = iou_xyxy(np.array(box), np.array([d.xyxy for d in result.detections]))
            if ious.size and ious.max() >= 0.3:
                return FNType.SUPPRESSED.value

        x1, y1, x2, y2 = box
        if img_area is not None:
            frac = ((x2 - x1) * (y2 - y1)) / (img_area + 1e-9)
            if frac < self.small_area_frac:
                return FNType.SIZE.value
            if frac < self.occlusion_area_frac:
                return FNType.OCCLUSION.value

        # Appearance: a faint defect over its patch -> low contrast.
        if image is not None:
            xi1, yi1 = max(0, int(y1)), max(0, int(x1))
            patch = image[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if patch.size > 0 and float(patch.std()) < self.low_contrast_std:
                return FNType.LOW_CONTRAST.value

        return FNType.MISSED.value

    # ------------------------------------------------------------------ output

    def summarize(self) -> TaxonomySummary:
        """Aggregate records into counts by type and by class."""
        by_type: dict[str, int] = {}
        by_class: dict[str, int] = {}
        n_fp = n_fn = 0
        for r in self.records:
            by_type[r.category] = by_type.get(r.category, 0) + 1
            by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
            if r.kind == "FP":
                n_fp += 1
            else:
                n_fn += 1
        return TaxonomySummary(by_type=by_type, by_class=by_class, total_fp=n_fp, total_fn=n_fn)

    def actionable_recommendations(self) -> list[str]:
        """Map the observed failure mix to concrete next steps.

        Only fires a recommendation when a category is actually present, and
        orders them by count so the biggest lever comes first. These strings are
        meant to be dropped straight into an eval report or a ticket.
        """
        summary = self.summarize()
        bt = summary.by_type
        playbook: dict[str, str] = {
            FPType.BACKGROUND.value:
                "Many background FPs: add hard-negative belt/seam crops to training "
                "(active_learning hard_negative source) and consider raising the class threshold.",
            FPType.DUPLICATE.value:
                "Duplicate FPs: tighten NMS IoU in detector.py (iou_threshold) — boxes are "
                "surviving suppression.",
            FPType.WRONG_CLASS.value:
                "Wrong-class FPs: classes are confusable (likely scratch<->crack). Mine confused "
                "pairs and rebalance via the WeightedRandomSampler; inspect simulator visual signatures.",
            FPType.POOR_LOCALIZATION.value:
                "Poor-localization FPs: boxes land just under IoU. Increase localization "
                "augmentation fidelity or train longer; check letterbox/box-reversal math.",
            FNType.MISSED.value:
                "Outright misses: the model never fires here. Likely under-represented — oversample "
                "via dataset weights and queue these frames for labeling.",
            FNType.LOW_CONTRAST.value:
                "Low-contrast misses: enable/strengthen CLAHE in preprocessing and raise the CLAHE "
                "augmentation probability in augmentation.yaml.",
            FNType.OCCLUSION.value:
                "Occlusion misses: increase CoarseDropout strength/probability so the model learns "
                "partial defects.",
            FNType.SIZE.value:
                "Small-object misses: train/infer at higher img_size or add a small-object head; "
                "scale-jitter augmentation toward smaller boxes.",
            FNType.SUPPRESSED.value:
                "Suppressed detections: a prediction existed but was dropped — lower the class "
                "threshold (see ThresholdSweeper min_recall) or loosen NMS.",
        }
        recs = [(bt[k], v) for k, v in playbook.items() if bt.get(k, 0) > 0]
        recs.sort(key=lambda t: t[0], reverse=True)
        return [msg for _, msg in recs]
