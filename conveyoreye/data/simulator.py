"""Procedural synthetic conveyor-belt frame generator.

WHY simulate instead of collecting real data?
---------------------------------------------
A learning project should not be gated on access to a factory line. More
importantly, simulation gives us *ground-truth control* over the two things that
make industrial detection hard and that this repo is built to study:

  1. Class imbalance — we dial the priors (scratch 40%, missing_part 8%) exactly,
     so the eval/active-learning layers have a known-imbalanced distribution to
     work against.
  2. Nuisance factors — vignette lighting, belt seams, texture — are rendered
     explicitly, so the preprocessing/drift layers have real signal to react to
     rather than synthetic-clean images.

Each defect class has its *own* drawing method (``_draw_scratch`` etc.) so the
visual vocabulary of each class is distinct and a detector can actually learn to
separate them. The generator emits an image plus YOLO-format boxes; it never
writes files itself — that is the script layer's job (scripts/simulate_data.py),
keeping this module a pure, testable function of its RNG seed.

Scalability path
----------------
  v1 (here): pure NumPy/OpenCV procedural rendering, single process.
  v2: domain randomization on real belt photos as backgrounds (paste defects
      onto captured textures) to close the sim-to-real gap.
  v3: a small GAN/diffusion model fine-tuned on a few hundred real defect crops
      for photorealistic synthesis; this module's ``Sample`` contract stays put.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from conveyoreye import CLASS_NAMES, CLASS_PRIORS, class_id


@dataclass
class BBox:
    """A single ground-truth box in YOLO-normalized coordinates.

    Stored normalized (0..1) so the box is resolution-independent — the same
    Sample can be rendered at any size without rescaling labels.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        """Serialize to a single YOLO label-file line: ``cls cx cy w h``."""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert to absolute pixel corners for drawing/diagnostics."""
        x1 = (self.cx - self.w / 2) * img_w
        y1 = (self.cy - self.h / 2) * img_h
        x2 = (self.cx + self.w / 2) * img_w
        y2 = (self.cy + self.h / 2) * img_h
        return int(x1), int(y1), int(x2), int(y2)


@dataclass
class Sample:
    """One rendered frame with its ground-truth boxes."""

    image: np.ndarray              # HxWx3 uint8, BGR (OpenCV convention)
    boxes: list[BBox] = field(default_factory=list)

    @property
    def class_ids(self) -> list[int]:
        return [b.class_id for b in self.boxes]


class ConveyorSimulator:
    """Generates textured belt frames with procedurally drawn defects.

    The simulator is deterministic given its seed: ``ConveyorSimulator(seed=0)``
    and a fixed call sequence reproduce identical frames, which matters for
    regression-testing the downstream metrics.
    """

    def __init__(
        self,
        img_size: int = 640,
        priors: dict[str, float] | None = None,
        seed: int | None = None,
        max_defects_per_frame: int = 3,
        empty_frame_prob: float = 0.15,
    ) -> None:
        self.img_size = img_size
        self.priors = priors or dict(CLASS_PRIORS)
        self.max_defects_per_frame = max_defects_per_frame
        self.empty_frame_prob = empty_frame_prob
        self.rng = np.random.default_rng(seed)

        # Normalize priors into a sampling vector aligned to CLASS_NAMES order.
        self._classes = list(CLASS_NAMES)
        p = np.array([self.priors.get(c, 0.0) for c in self._classes], dtype=np.float64)
        if p.sum() <= 0:
            raise ValueError("Class priors must sum to a positive value.")
        self._prior_vec = p / p.sum()

        # Dispatch table: class name -> drawing method. Adding a defect type is a
        # one-line registration, not a branch edit elsewhere.
        self._drawers = {
            "scratch": self._draw_scratch,
            "crack": self._draw_crack,
            "dent": self._draw_dent,
            "discoloration": self._draw_discoloration,
            "missing_part": self._draw_missing_part,
        }

    # ----------------------------------------------------------------- public

    def generate(self) -> Sample:
        """Render a single frame: belt background + lighting + 0..N defects."""
        img = self._make_belt_background()
        boxes: list[BBox] = []

        if self.rng.random() >= self.empty_frame_prob:
            n = int(self.rng.integers(1, self.max_defects_per_frame + 1))
            for _ in range(n):
                cls = self._sample_class()
                box = self._drawers[cls](img)
                if box is not None:
                    boxes.append(box)

        img = self._apply_vignette(img)
        return Sample(image=img, boxes=boxes)

    def generate_batch(self, n: int) -> list[Sample]:
        """Convenience batch generator. Kept separate so callers can stream."""
        return [self.generate() for _ in range(n)]

    # ------------------------------------------------------------- background

    def _make_belt_background(self) -> np.ndarray:
        """Build a textured rubber-belt background with horizontal seams.

        The texture is intentionally non-uniform: a flat gray background would
        let a detector cheat by treating any deviation as a defect. Belt seams
        are recurring horizontal lines that a naive model loves to mistake for
        cracks — a deliberate hard-negative built into the data.
        """
        s = self.img_size
        base_gray = int(self.rng.integers(90, 130))
        img = np.full((s, s, 3), base_gray, dtype=np.uint8)

        # Fine rubber grain.
        noise = self.rng.normal(0, 8, (s, s)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)

        # Horizontal belt seams (the canonical crack-confuser).
        n_seams = int(self.rng.integers(1, 4))
        for _ in range(n_seams):
            y = int(self.rng.integers(0, s))
            thickness = int(self.rng.integers(1, 3))
            shade = int(np.clip(base_gray - self.rng.integers(20, 45), 0, 255))
            cv2.line(img, (0, y), (s, y), (shade, shade, shade), thickness)

        return img

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        """Radial brightness falloff modeling a single overhead lamp.

        Vignetting shifts the global brightness/contrast statistics frame to
        frame, which is exactly the signal ImageStatsDriftDetector watches.
        """
        s = self.img_size
        cx = self.rng.uniform(0.35, 0.65) * s
        cy = self.rng.uniform(0.35, 0.65) * s
        yy, xx = np.mgrid[0:s, 0:s]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = math.sqrt(2) * s / 2
        strength = self.rng.uniform(0.3, 0.6)
        mask = 1.0 - strength * (dist / max_dist)
        mask = np.clip(mask, 0.3, 1.0).astype(np.float32)
        out = (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)
        return out

    # ----------------------------------------------------------- class choice

    def _sample_class(self) -> str:
        """Draw a defect class according to the (imbalanced) prior vector."""
        idx = int(self.rng.choice(len(self._classes), p=self._prior_vec))
        return self._classes[idx]

    def _rand_region(self, min_frac: float, max_frac: float) -> tuple[int, int, int, int]:
        """Pick a random axis-aligned region (x1,y1,x2,y2) sized as a frame fraction."""
        s = self.img_size
        w = int(self.rng.uniform(min_frac, max_frac) * s)
        h = int(self.rng.uniform(min_frac, max_frac) * s)
        x1 = int(self.rng.integers(0, max(1, s - w)))
        y1 = int(self.rng.integers(0, max(1, s - h)))
        return x1, y1, x1 + w, y1 + h

    def _box_from_xyxy(self, cls: str, x1: int, y1: int, x2: int, y2: int) -> BBox:
        """Convert an absolute pixel region into a normalized YOLO BBox, clamped."""
        s = self.img_size
        x1, x2 = sorted((max(0, x1), min(s, x2)))
        y1, y2 = sorted((max(0, y1), min(s, y2)))
        cx = (x1 + x2) / 2 / s
        cy = (y1 + y2) / 2 / s
        w = (x2 - x1) / s
        h = (y2 - y1) / s
        return BBox(class_id(cls), cx, cy, max(w, 1e-3), max(h, 1e-3))

    # --------------------------------------------------------- defect drawers
    # Each drawer mutates `img` in place and returns the BBox it occupied (or
    # None if it declined to draw). Visual signatures are intentionally distinct.

    def _draw_scratch(self, img: np.ndarray) -> BBox | None:
        """Thin, bright, near-straight line — a tool gouge catching light."""
        s = self.img_size
        x1 = int(self.rng.integers(0, s))
        y1 = int(self.rng.integers(0, s))
        length = int(self.rng.uniform(0.08, 0.30) * s)
        angle = self.rng.uniform(0, math.pi)
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        bright = int(self.rng.integers(180, 240))
        thickness = int(self.rng.integers(1, 3))
        cv2.line(img, (x1, y1), (x2, y2), (bright, bright, bright), thickness, cv2.LINE_AA)
        pad = 4
        return self._box_from_xyxy("scratch", min(x1, x2) - pad, min(y1, y2) - pad,
                                   max(x1, x2) + pad, max(y1, y2) + pad)

    def _draw_crack(self, img: np.ndarray) -> BBox | None:
        """Dark, jagged polyline — a fracture. Distinguished from scratch by being
        dark + branching rather than bright + straight."""
        s = self.img_size
        x = int(self.rng.integers(s // 6, 5 * s // 6))
        y = int(self.rng.integers(s // 6, 5 * s // 6))
        pts = [(x, y)]
        n_seg = int(self.rng.integers(4, 8))
        step = int(self.rng.uniform(0.03, 0.07) * s)
        heading = self.rng.uniform(0, 2 * math.pi)
        for _ in range(n_seg):
            heading += self.rng.uniform(-0.9, 0.9)  # jaggedness
            x = int(x + step * math.cos(heading))
            y = int(y + step * math.sin(heading))
            pts.append((x, y))
        dark = int(self.rng.integers(20, 60))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (dark, dark, dark), 1, cv2.LINE_AA)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 5
        return self._box_from_xyxy("crack", min(xs) - pad, min(ys) - pad,
                                   max(xs) + pad, max(ys) + pad)

    def _draw_dent(self, img: np.ndarray) -> BBox | None:
        """Soft elliptical shadow with a highlight rim — a depression in the part."""
        x1, y1, x2, y2 = self._rand_region(0.06, 0.18)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, (170, 170, 170), 2)
        alpha = self.rng.uniform(0.4, 0.7)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return self._box_from_xyxy("dent", x1, y1, x2, y2)

    def _draw_discoloration(self, img: np.ndarray) -> BBox | None:
        """Low-contrast colored blob — staining/oxidation. Hard to see post-JPEG,
        which is the point: it stresses the contrast-recovery preprocessing."""
        x1, y1, x2, y2 = self._rand_region(0.10, 0.25)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)
        tint = self.rng.integers(0, 60, size=3).tolist()  # subtle BGR shift
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, tuple(int(t) for t in tint), -1)
        alpha = self.rng.uniform(0.2, 0.4)  # deliberately faint
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return self._box_from_xyxy("discoloration", x1, y1, x2, y2)

    def _draw_missing_part(self, img: np.ndarray) -> BBox | None:
        """A rectangular cutout where a component should be — flat dark patch with
        a crisp border. Rare by prior, visually unambiguous, so the challenge is
        *frequency* (few examples) not appearance."""
        x1, y1, x2, y2 = self._rand_region(0.12, 0.28)
        hole = int(self.rng.integers(15, 45))
        cv2.rectangle(img, (x1, y1), (x2, y2), (hole, hole, hole), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)  # machined edge
        return self._box_from_xyxy("missing_part", x1, y1, x2, y2)
