"""Threshold selection: turn a PR sweep into per-class operating points.

WHY a separate sweeper from the evaluator?
------------------------------------------
The evaluator scores a model at thresholds you already chose. The *sweeper*
answers the prior question: given a business goal per class, what threshold
should we choose? Those are different jobs and different lifecycles — you sweep
once after training to *set* thresholds.yaml, then evaluate continuously against
them.

The strategy is expressed per class as one of:
  * ("min_recall", target)    -> lowest threshold that still meets P-side sanity,
                                  picked to *guarantee* recall >= target. Use for
                                  crack (never miss a safety defect).
  * ("min_precision", target) -> highest-recall point with precision >= target.
                                  Use for missing_part (never falsely stop the line).
  * ("max_f1", None)          -> the balanced point. Use for cosmetic classes.

This mirrors thresholds.yaml's asymmetric business costs directly: the strategy
dict is the machine-readable form of that file's rationale comments.

Scalability path
----------------
  v1 (here): sweep a fixed grid of candidate thresholds over accumulated pairs.
  v2: sweep on a held-out *operational* log (real traffic) not just val, so the
      operating point reflects the deployed distribution.
  v3: multi-objective / cost-curve selection that optimizes expected dollar cost
      using per-class FP/FN costs instead of a single P or R target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from conveyoreye import CLASS_NAMES, NUM_CLASSES, class_id, class_name

StrategyKind = Literal["min_recall", "min_precision", "max_f1"]


@dataclass
class PRPoint:
    """One point on a class's precision-recall sweep."""

    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass
class ThresholdRecommendation:
    """The chosen threshold for one class plus why it was chosen.

    Carries the achieved P/R/F1 and whether the target was actually met, so
    export_yaml can write an honest rationale and ops can see when a goal was
    infeasible (e.g. recall target unreachable at any threshold).
    """

    name: str
    strategy: StrategyKind
    target: float | None
    threshold: float
    achieved_precision: float
    achieved_recall: float
    achieved_f1: float
    target_met: bool


class ThresholdSweeper:
    """Sweeps PR curves from accumulated (conf, tp) pairs and recommends cutoffs.

    Consumes the same per-class records the DetectionEvaluator builds; construct
    it directly from an evaluator via ``from_evaluator`` so there is no second
    matching pass.
    """

    def __init__(
        self,
        records: dict[int, list[tuple[float, int]]],
        n_gt: dict[int, int],
        grid: np.ndarray | None = None,
    ) -> None:
        self._records = records
        self._n_gt = n_gt
        # Candidate thresholds. 0.01..0.99 is plenty granular for an operating
        # point; finer grids buy nothing given label noise.
        self.grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)

    @classmethod
    def from_evaluator(cls, evaluator) -> "ThresholdSweeper":
        """Build a sweeper from a DetectionEvaluator's accumulated state."""
        return cls(records=evaluator._records, n_gt=evaluator._n_gt)

    # ------------------------------------------------------------------- sweep

    def sweep_class(self, cid: int) -> list[PRPoint]:
        """Compute the full PR sweep for one class over the threshold grid."""
        recs = self._records.get(cid, [])
        n_gt = self._n_gt.get(cid, 0)
        points: list[PRPoint] = []
        confs = np.array([c for c, _ in recs])
        tps = np.array([t for _, t in recs])
        for thr in self.grid:
            keep = confs >= thr
            n_pred = int(keep.sum())
            if n_pred == 0:
                points.append(PRPoint(float(thr), 0.0, 0.0, 0.0))
                continue
            tp = int(tps[keep].sum())
            p = tp / n_pred
            r = tp / (n_gt + 1e-9) if n_gt > 0 else 0.0
            f1 = 2 * p * r / (p + r + 1e-9)
            points.append(PRPoint(float(thr), float(p), float(r), float(f1)))
        return points

    # --------------------------------------------------------------- recommend

    def recommend(
        self, strategy: dict[str, tuple[StrategyKind, float | None]]
    ) -> dict[str, ThresholdRecommendation]:
        """Recommend a threshold per class from a {name: (kind, target)} strategy.

        Classes absent from ``strategy`` are skipped (caller keeps their existing
        threshold). Each kind has a deliberate tie-break:
          * min_recall    : among points meeting recall>=target, take the one with
            highest precision (catch them all, but as cleanly as possible).
          * min_precision : among points meeting precision>=target, take the one
            with highest recall (be safe to act on, but catch as many as you can).
          * max_f1        : the single highest-F1 point.
        If no point meets the target, fall back to the closest feasible point and
        flag target_met=False so the infeasibility is visible, not silent.
        """
        out: dict[str, ThresholdRecommendation] = {}
        for name, (kind, target) in strategy.items():
            cid = class_id(name)
            pts = self.sweep_class(cid)
            rec = self._select(name, kind, target, pts)
            out[name] = rec
        return out

    def _select(
        self,
        name: str,
        kind: StrategyKind,
        target: float | None,
        pts: list[PRPoint],
    ) -> ThresholdRecommendation:
        if not pts:
            return ThresholdRecommendation(name, kind, target, 0.5, 0, 0, 0, False)

        if kind == "max_f1":
            best = max(pts, key=lambda p: p.f1)
            return ThresholdRecommendation(
                name, kind, target, best.threshold, best.precision,
                best.recall, best.f1, True
            )

        if kind == "min_recall":
            assert target is not None
            feasible = [p for p in pts if p.recall >= target]
            if feasible:
                best = max(feasible, key=lambda p: p.precision)
                met = True
            else:
                # Cannot reach the recall target -> take max-recall point we can.
                best = max(pts, key=lambda p: p.recall)
                met = False
            return ThresholdRecommendation(
                name, kind, target, best.threshold, best.precision,
                best.recall, best.f1, met
            )

        if kind == "min_precision":
            assert target is not None
            feasible = [p for p in pts if p.precision >= target]
            if feasible:
                best = max(feasible, key=lambda p: p.recall)
                met = True
            else:
                best = max(pts, key=lambda p: p.precision)
                met = False
            return ThresholdRecommendation(
                name, kind, target, best.threshold, best.precision,
                best.recall, best.f1, met
            )

        raise ValueError(f"Unknown strategy kind: {kind!r}")

    # ------------------------------------------------------------------ export

    def export_yaml(
        self,
        recommendations: dict[str, ThresholdRecommendation],
        path: str | Path,
        iou_threshold: float = 0.50,
        active_learning: dict | None = None,
    ) -> None:
        """Write recommendations back into a thresholds.yaml-compatible file.

        Round-trips the same schema the detector loads, so the loop is closed:
        sweep -> export -> the next deploy reads the new operating points. The
        achieved P/R and target-met flag are written as comments-in-data so the
        file stays self-documenting.
        """
        classes: dict[str, dict] = {}
        for name in CLASS_NAMES:
            rec = recommendations.get(name)
            if rec is None:
                continue
            entry: dict[str, object] = {"threshold": round(rec.threshold, 4)}
            if rec.strategy == "min_recall":
                entry["target_recall"] = rec.target
            elif rec.strategy == "min_precision":
                entry["target_precision"] = rec.target
            entry["achieved_precision"] = round(rec.achieved_precision, 4)
            entry["achieved_recall"] = round(rec.achieved_recall, 4)
            entry["target_met"] = rec.target_met
            entry["rationale"] = (
                f"Auto-selected via {rec.strategy}"
                + ("" if rec.target is None else f"={rec.target}")
                + ("" if rec.target_met else "  [TARGET NOT MET — review]")
            )
            classes[name] = entry

        doc: dict[str, object] = {
            "classes": classes,
            "default_threshold": 0.50,
            "iou_threshold": iou_threshold,
        }
        if active_learning is not None:
            doc["active_learning"] = active_learning

        with open(path, "w") as f:
            f.write("# Auto-generated by ThresholdSweeper.export_yaml — review before deploy.\n")
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
