"""ConveyorEye — industrial defect detection on a simulated conveyor belt.

Why this package exists
-----------------------
This is a *learning-oriented* reference stack for a real-world detection problem:
spotting surface/structural defects on parts moving past a fixed camera on a
conveyor. It is deliberately end-to-end — simulation, preprocessing, detection,
calibration, evaluation, monitoring, and active learning — because the
interesting engineering lessons live in the *seams between* these stages, not in
any single model.

Design stance
-------------
Every public interface is written as if it will be scaled later, even where the
v1 implementation is a toy. Concretely:
  * Core methods return dataclasses, never raw dicts, so call sites are typed.
  * Batch is the primary unit of work (``predict_batch``), not single frames —
    single-frame APIs are thin wrappers, matching how throughput-bound serving
    actually runs.
  * Config (thresholds, augmentation) lives in YAML, not code, so an ops change
    does not require a redeploy.

The single source of truth for class identity is ``CLASS_NAMES`` below; class
*ids* are list indices, matching YOLO's integer label convention.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Ordered by class id. Order is load-bearing: it defines the integer<->name map
# used by YOLO labels, the detector, the evaluator and every YAML config.
CLASS_NAMES: tuple[str, ...] = (
    "scratch",        # 0
    "crack",          # 1  (safety-critical — recall-biased thresholds)
    "dent",           # 2
    "discoloration",  # 3
    "missing_part",   # 4  (rare — precision-biased thresholds)
)

# Designed class priors. Imbalanced on purpose: this is what makes per-class
# metrics, calibration and active learning *matter*. A model that only ever sees
# this distribution at train time will be over-confident on scratch and starved
# of missing_part — exactly the failure modes the eval/monitoring layers surface.
CLASS_PRIORS: dict[str, float] = {
    "scratch": 0.40,
    "crack": 0.22,
    "dent": 0.18,
    "discoloration": 0.12,
    "missing_part": 0.08,
}

NUM_CLASSES: int = len(CLASS_NAMES)


def class_id(name: str) -> int:
    """Map a class name to its integer id. Raises on unknown names by design."""
    return CLASS_NAMES.index(name)


def class_name(idx: int) -> str:
    """Map an integer class id back to its name."""
    return CLASS_NAMES[idx]
