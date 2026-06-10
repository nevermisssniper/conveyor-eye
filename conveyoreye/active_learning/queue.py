"""Labeling queue: the durable bridge between "model is unsure" and "human labels".

WHY a stateful queue with persisted status?
-------------------------------------------
Active learning is only useful if the selected frames actually get labeled and
fed back. That is a multi-actor, multi-day workflow — the sampler enqueues, a
human reviews later, labels land, and the frame is merged into training. None of
that survives a process restart unless state is persisted, so the queue is
JSON-backed and every frame carries an explicit ``FrameStatus`` through its
lifecycle.

WHY prioritize hard_negative over uncertainty?
----------------------------------------------
A confirmed hard negative (the model fired confidently on a belt seam — a known
FP) is a *higher-value* label than a merely uncertain frame: it directly corrects
a systematic error. So ``get_pending`` surfaces hard_negative-sourced frames
first, then uncertainty-sourced, then by score within each group.

Scalability path
----------------
  v1 (here): single JSON file, whole-state rewrite on save, in-process.
  v2: SQLite-backed queue (reuse the monitoring DB) with row-level status updates
      and optimistic concurrency for multiple labelers.
  v3: a real labeling-service integration (Label Studio / internal tool) where
      this class becomes the client and status syncs via webhook.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class FrameStatus(str, Enum):
    """Lifecycle of a frame in the labeling pipeline."""

    PENDING = "pending"          # selected, awaiting a human
    IN_REVIEW = "in_review"      # claimed by a labeler
    LABELED = "labeled"          # labels submitted, ready to merge
    REJECTED = "rejected"        # not useful (e.g. truly empty / mis-selected)
    MERGED = "merged"            # copied into the training set — terminal


@dataclass
class QueueItem:
    """One frame tracked through the labeling lifecycle.

    ``source`` ("uncertainty" | "hard_negative") drives prioritization; ``score``
    orders within a source; ``label_path`` is filled once a human submits labels.
    """

    frame_id: str
    image_path: str
    source: str
    score: float
    status: str = FrameStatus.PENDING.value
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    label_path: str | None = None
    note: str | None = None


class LabelingQueue:
    """JSON-persisted queue of frames awaiting human labels.

    State is a dict frame_id -> QueueItem, flushed to ``state_path`` on every
    mutation so a crash never loses queue position. Construct it pointing at the
    same path across runs to resume.
    """

    # Source priority: lower number == surfaced first.
    _SOURCE_PRIORITY = {"hard_negative": 0, "uncertainty": 1}

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self._items: dict[str, QueueItem] = {}
        if self.state_path.exists():
            self._load()

    # ----------------------------------------------------------- persistence

    def _load(self) -> None:
        data = json.loads(self.state_path.read_text())
        self._items = {fid: QueueItem(**item) for fid, item in data.items()}

    def _save(self) -> None:
        """Atomic-ish write: dump to a temp file then replace, so a crash mid-write
        cannot corrupt the queue state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({fid: asdict(i) for fid, i in self._items.items()}, indent=2))
        tmp.replace(self.state_path)

    # ---------------------------------------------------------------- mutate

    def add(
        self, frame_id: str, image_path: str, source: str, score: float,
        note: str | None = None,
    ) -> bool:
        """Enqueue a frame. Returns False if it is already tracked (dedup).

        Idempotent on frame_id so re-running the sampler over overlapping batches
        never double-queues the same frame.
        """
        if frame_id in self._items:
            return False
        self._items[frame_id] = QueueItem(
            frame_id=frame_id, image_path=image_path, source=source, score=score, note=note
        )
        self._save()
        return True

    def add_batch(self, items: list[dict]) -> int:
        """Bulk-enqueue; returns how many were newly added (skips duplicates)."""
        added = 0
        for it in items:
            if it["frame_id"] not in self._items:
                self._items[it["frame_id"]] = QueueItem(
                    frame_id=it["frame_id"], image_path=it["image_path"],
                    source=it.get("source", "uncertainty"), score=it.get("score", 0.0),
                    note=it.get("note"),
                )
                added += 1
        if added:
            self._save()
        return added

    def set_status(
        self, frame_id: str, status: FrameStatus, label_path: str | None = None
    ) -> None:
        """Transition a frame's status (and optionally attach its label file)."""
        item = self._items[frame_id]
        item.status = status.value
        item.updated_ts = time.time()
        if label_path is not None:
            item.label_path = label_path
        self._save()

    # ------------------------------------------------------------------ read

    def get_pending(self, limit: int | None = None) -> list[QueueItem]:
        """Return pending frames, hard-negatives first, then by score desc.

        This ordering is the policy: confirmed systematic errors (hard negatives)
        outrank merely-uncertain frames, and within each group the highest score
        is labeled first.
        """
        pending = [i for i in self._items.values() if i.status == FrameStatus.PENDING.value]
        pending.sort(
            key=lambda i: (self._SOURCE_PRIORITY.get(i.source, 99), -i.score)
        )
        return pending[:limit] if limit is not None else pending

    def stats(self) -> dict[str, int]:
        """Count items in each status — what the /queue/stats endpoint returns."""
        counts = {s.value: 0 for s in FrameStatus}
        for i in self._items.values():
            counts[i.status] = counts.get(i.status, 0) + 1
        counts["total"] = len(self._items)
        return counts

    # ----------------------------------------------------------------- merge

    def merge_to_training(
        self, train_images_dir: str | Path, train_labels_dir: str | Path
    ) -> int:
        """Copy every LABELED frame into the training split and mark it MERGED.

        Copies image + label file into the YOLO train layout the dataset loader
        expects, then flips status to MERGED (terminal). Returns the count merged.
        A frame without a ``label_path`` is skipped — you cannot merge an
        unlabeled frame even if its status drifted.
        """
        img_dir = Path(train_images_dir)
        lbl_dir = Path(train_labels_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        merged = 0
        for item in self._items.values():
            if item.status != FrameStatus.LABELED.value or not item.label_path:
                continue
            src_img = Path(item.image_path)
            src_lbl = Path(item.label_path)
            if not src_img.exists() or not src_lbl.exists():
                continue
            shutil.copy2(src_img, img_dir / src_img.name)
            shutil.copy2(src_lbl, lbl_dir / f"{src_img.stem}.txt")
            item.status = FrameStatus.MERGED.value
            item.updated_ts = time.time()
            merged += 1

        if merged:
            self._save()
        return merged
