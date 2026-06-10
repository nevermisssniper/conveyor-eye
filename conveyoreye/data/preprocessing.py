"""Preprocessing: training augmentation + inference-time frame conditioning.

WHY split train vs inference transforms?
----------------------------------------
They have opposite goals. *Training* augmentation deliberately corrupts images to
model the nuisance factors of a real line (motion blur, compression, occlusion)
so the model generalizes. *Inference* preprocessing does the opposite — it
normalizes geometry (letterbox) and optionally recovers contrast (CLAHE) so the
model sees inputs as close as possible to its training distribution, and returns
the scale factor needed to map predicted boxes back to original pixels.

Both paths share one principle: the augmentation *policy* lives in YAML
(configs/augmentation.yaml), not here, because it is a line-specific operational
knob (see that file's header). This module is the typed plumbing that turns that
config into an ``A.Compose`` and the letterbox math that serving relies on.

Scalability path
----------------
  v1 (here): Albumentations on CPU, per-frame letterbox in NumPy.
  v2: move letterbox/normalize onto the GPU (torch) to remove the host->device
      copy bottleneck under high QPS.
  v3: fuse preprocessing into the model graph (export with embedded letterbox)
      so the serving layer ships raw frames straight to the accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml


@dataclass
class LetterboxResult:
    """Output of inference preprocessing — carries everything needed to invert it.

    A raw resized tensor is useless to the caller without the scale/pad it was
    produced with: predicted boxes are in letterboxed space and must be mapped
    back. Returning a dataclass forces that information to travel with the image.
    """

    image: np.ndarray          # CHW float32, normalized 0..1 (model-ready)
    scale: float               # original -> resized scale factor (uniform)
    pad_x: int                 # left padding in resized pixels
    pad_y: int                 # top padding in resized pixels
    orig_h: int
    orig_w: int

    def reverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map xyxy boxes from letterboxed space back to original-image pixels.

        Inverse of the letterbox: subtract padding, divide by scale, clamp to the
        original frame. Vectorized over an (N,4) array.
        """
        if boxes_xyxy.size == 0:
            return boxes_xyxy
        out = boxes_xyxy.astype(np.float32).copy()
        out[:, [0, 2]] -= self.pad_x
        out[:, [1, 3]] -= self.pad_y
        out /= self.scale
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.orig_w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.orig_h)
        return out


# ----------------------------------------------------------------- transforms

def build_transforms(config_path: str | Path, split: str = "train") -> A.Compose:
    """Load augmentation.yaml and build an ``A.Compose`` for the given split.

    bbox_params is attached here (not in the YAML) so the config stays a pure
    transform list and the YOLO-format contract is enforced in one place.

    label_fields=["class_labels"] means callers pass boxes *and* a parallel list
    of class ids; Albumentations keeps them in sync when it drops/clips a box.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    transform_specs = cfg.get(split) or []
    transforms = [_build_one(spec) for spec in transform_specs]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            # Drop boxes that augmentation has shrunk below usefulness — a box
            # with <10% area or tiny min-visibility is noise, not a label.
            min_visibility=0.10,
            min_area=1.0,
        ),
    )


def _build_one(spec: dict[str, Any]) -> A.BasicTransform:
    """Instantiate a single Albumentations transform from a {name, params} spec."""
    name = spec["name"]
    params = spec.get("params", {}) or {}
    cls = getattr(A, name, None)
    if cls is None:
        raise ValueError(f"Unknown Albumentations transform: {name!r}")
    # Albumentations expects tuples for several range params; YAML gives lists.
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
    return cls(**params)


# -------------------------------------------------------------- inference path

def preprocess_frame(
    frame: np.ndarray,
    img_size: int = 640,
    pad_value: int = 114,
    normalize: bool = True,
) -> LetterboxResult:
    """Letterbox a single BGR frame to a square ``img_size`` and normalize.

    Letterbox = resize preserving aspect ratio, then pad the short side. This is
    YOLO's expected input geometry; squashing the aspect ratio instead would
    distort defect shapes and hurt localization. pad_value=114 matches the gray
    YOLO uses so padding does not look like a dark defect.
    """
    orig_h, orig_w = frame.shape[:2]
    scale = min(img_size / orig_h, img_size / orig_w)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    canvas = np.full((img_size, img_size, 3), pad_value, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    img = canvas[:, :, ::-1]  # BGR -> RGB (model convention)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW
    out = img.astype(np.float32) / 255.0 if normalize else img.astype(np.float32)

    return LetterboxResult(
        image=out, scale=scale, pad_x=pad_x, pad_y=pad_y, orig_h=orig_h, orig_w=orig_w
    )


def batch_preprocess(
    frames: list[np.ndarray],
    img_size: int = 640,
    pad_value: int = 114,
    normalize: bool = True,
) -> tuple[np.ndarray, list[LetterboxResult]]:
    """Letterbox a list of frames into one stacked NCHW batch.

    Returns the stacked tensor *and* the per-frame LetterboxResult list, because
    each frame may have had a different scale/pad and the caller needs every one
    of them to reverse its own boxes.
    """
    results = [preprocess_frame(f, img_size, pad_value, normalize) for f in frames]
    batch = np.stack([r.image for r in results], axis=0) if results else np.empty(
        (0, 3, img_size, img_size), dtype=np.float32
    )
    return batch, results


# ---------------------------------------------------------- physical utilities

def estimate_motion_blur_kernel(
    belt_speed_m_s: float,
    exposure_s: float,
    pixels_per_meter: float,
) -> int:
    """Derive a motion-blur kernel size (odd px) from line physics.

    smear_px = belt_speed * exposure_time * pixels_per_meter

    This is the bridge between the augmentation.yaml comment and reality: when
    the line changes belt speed or exposure, call this to get the new MotionBlur
    blur_limit instead of guessing. Always rounds up to the next odd integer
    because OpenCV/Albumentations motion kernels must be odd-sized.
    """
    smear_px = belt_speed_m_s * exposure_s * pixels_per_meter
    k = int(np.ceil(smear_px))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def clahe_equalize(
    frame: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel.

    Applied to the luminance channel only (in LAB space) so we boost local
    contrast without shifting hue — important for discoloration, whose color is
    the signal. Used both as a (probabilistic) train augmentation and as an
    optional deterministic inference step to recover faint, compression-crushed
    defects.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
