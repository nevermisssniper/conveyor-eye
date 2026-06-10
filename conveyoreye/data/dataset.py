"""YOLO-format dataset with class-balanced sampling.

WHY a custom Dataset when Ultralytics ships its own?
----------------------------------------------------
Two reasons specific to this project's learning goals:

  1. Imbalance handling. Our priors are skewed by design (missing_part is 8%).
     A uniform sampler trains the model to ignore the rare class. This Dataset
     exposes a ``WeightedRandomSampler`` whose weights are the inverse of each
     frame's rarest-class frequency, so rare-class frames are oversampled — the
     standard lever for imbalanced detection, made explicit and inspectable.
  2. Transparency. Building the loader ourselves keeps the YOLO label-file
     contract (one .txt per image, ``cls cx cy w h`` lines) visible and lets the
     preprocessing/augmentation seam be the same one used at inference.

NOTE: ``scripts/train.py`` uses Ultralytics' own trainer (which wants a
dataset.yaml, not a torch Dataset). This class is the loader you would reach for
when writing a *custom* training loop or for offline analysis/embedding
extraction — it is the v2 path when the project outgrows the Ultralytics CLI.

Scalability path
----------------
  v1 (here): eager file listing, decode per __getitem__, CPU augmentation.
  v2: cache decoded images / use an LMDB or webdataset shard for IO-bound scale.
  v3: replace WeightedRandomSampler with a class-balanced *batch* sampler so
      every batch is guaranteed mixed — smoother gradients for the rare class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler

from conveyoreye import NUM_CLASSES


@dataclass
class DatasetItem:
    """One training example after augmentation.

    boxes is (N,4) YOLO-normalized; class_labels is (N,) int. Kept as a dataclass
    so a custom training loop destructures typed fields, not dict keys.
    """

    image: np.ndarray            # CHW float32 (post-transform)
    boxes: np.ndarray            # (N, 4) float32, YOLO cxcywh normalized
    class_labels: np.ndarray     # (N,) int64
    image_path: str


class YoloDetectionDataset(Dataset):
    """Reads an Ultralytics-style split directory: ``images/`` + ``labels/``.

    Expects the standard layout produced by scripts/simulate_data.py::

        <root>/images/*.png
        <root>/labels/*.txt   # same stem, YOLO lines

    The optional ``transform`` is an ``A.Compose`` from preprocessing.build_transforms.
    """

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        self.transform = transform

        self.image_paths: list[Path] = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _read_labels(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse a YOLO label file -> (boxes (N,4), class_ids (N,))."""
        lp = self._label_path(image_path)
        if not lp.exists():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        boxes, classes = [], []
        for line in lp.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = line.split()
            classes.append(int(cls))
            boxes.append([float(cx), float(cy), float(w), float(h)])
        return (
            np.array(boxes, np.float32).reshape(-1, 4),
            np.array(classes, np.int64),
        )

    def __getitem__(self, idx: int) -> DatasetItem:
        image_path = self.image_paths[idx]
        img = cv2.imread(str(image_path))  # BGR HWC uint8
        boxes, class_ids = self._read_labels(image_path)

        if self.transform is not None:
            aug = self.transform(
                image=img,
                bboxes=boxes.tolist(),
                class_labels=class_ids.tolist(),
            )
            img = aug["image"]
            boxes = np.array(aug["bboxes"], np.float32).reshape(-1, 4)
            class_ids = np.array(aug["class_labels"], np.int64)
        else:
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

        return DatasetItem(
            image=np.ascontiguousarray(img),
            boxes=boxes,
            class_labels=class_ids,
            image_path=str(image_path),
        )

    # ------------------------------------------------------ imbalance handling

    def class_frequencies(self) -> np.ndarray:
        """Count label occurrences per class across the whole split.

        Reads only label files (cheap — no image decode), so it is safe to call
        once up front to build sampler weights.
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for p in self.image_paths:
            _, class_ids = self._read_labels(p)
            for c in class_ids:
                counts[c] += 1
        return counts

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler that oversamples rare-class frames.

        Per-frame weight = max over the frame's classes of (1 / class_frequency).
        Using the *rarest* class in each frame (rather than a sum) means a frame
        containing a missing_part is pulled up to missing_part's weight even if it
        also contains a common scratch — we want every rare example seen often.
        Empty frames get the mean weight so the model still sees true negatives.

        TODO (v3): replace with a class-balanced batch sampler that guarantees a
        minimum count of each class *per batch*, not just in expectation.
        """
        counts = self.class_frequencies()
        # Avoid divide-by-zero for classes absent in this split.
        inv_freq = np.where(counts > 0, 1.0 / counts, 0.0)

        weights: list[float] = []
        for p in self.image_paths:
            _, class_ids = self._read_labels(p)
            if class_ids.size == 0:
                weights.append(float(inv_freq[inv_freq > 0].mean()) if (inv_freq > 0).any() else 1.0)
            else:
                weights.append(float(inv_freq[class_ids].max()))

        return WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
