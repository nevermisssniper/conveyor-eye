"""YOLOv8 detector wrapper returning structured, typed results.

WHY wrap Ultralytics at all?
----------------------------
Ultralytics' ``Results`` object is convenient interactively but a liability in a
system: it couples every call site to a third-party class, mixes plotting with
inference, and returns tensors the rest of our stack would have to re-interpret.
This wrapper draws a hard boundary: callers see only ``DetectionResult`` /
``Detection`` dataclasses with plain Python/NumPy fields. The day we swap YOLOv8
for YOLOv10, RT-DETR, or a TensorRT engine, *only this file changes* — metrics,
serving, logging and active learning are untouched.

Three design choices worth calling out:
  * ``predict_batch`` is the primary interface. Throughput-bound serving lives or
    dies on batching; single-frame ``predict`` is a thin convenience wrapper.
  * ``predict_stream`` is a generator for async consumers (the serving layer and
    the inference logger want to start processing frame 0's result while frame 1
    is still on the GPU).
  * Per-class thresholds and an optional calibrator are applied *here*, after the
    model, so the raw detector and the operating point stay decoupled.

Scalability path
----------------
  v1 (here): Ultralytics ``YOLO.predict`` on a single device.
  v2: export to ONNX/TensorRT and back this wrapper with an inference runtime;
      the dataclass contract is unchanged.
  v3: a model server (Triton) behind ``predict_batch``, with this class as the
      thin client. ``predict_stream`` becomes a gRPC stream consumer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np

from conveyoreye import CLASS_NAMES, NUM_CLASSES, class_name

if TYPE_CHECKING:
    from conveyoreye.model.calibration import ConfidenceCalibrator


@dataclass
class Detection:
    """A single detected box.

    xyxy is in *original-image* pixel coordinates (the wrapper reverses the
    letterbox before constructing this), so downstream code never has to know
    about model input geometry. ``raw_confidence`` is preserved alongside the
    (possibly calibrated) ``confidence`` so the calibration layer is auditable.
    """

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    raw_confidence: float | None = None  # pre-calibration score, if calibrated

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionResult:
    """All detections for one frame, plus light provenance for monitoring.

    ``latency_ms`` and ``frame_id`` exist because this object is what the
    inference logger persists — the result *is* the monitoring record.
    """

    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)   # (h, w) of the original frame
    latency_ms: float = 0.0
    frame_id: str | None = None

    @property
    def max_confidence(self) -> float:
        """Highest detection confidence, or 0.0 for an empty frame.

        This is the frame-level uncertainty signal the active-learning samplers
        and the low-confidence drift queries key off of.
        """
        return max((d.confidence for d in self.detections), default=0.0)

    def confidences(self) -> list[float]:
        return [d.confidence for d in self.detections]

    def by_class(self) -> dict[str, list[Detection]]:
        out: dict[str, list[Detection]] = {c: [] for c in CLASS_NAMES}
        for d in self.detections:
            out[d.class_name].append(d)
        return out


class Detector:
    """Thin, typed facade over an Ultralytics YOLO model."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        img_size: int = 640,
        iou_threshold: float = 0.50,
        class_thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.25,
        calibrator: "ConfidenceCalibrator | None" = None,
        warmup: bool = True,
    ) -> None:
        # Imported lazily so importing the package (e.g. for the data layer) does
        # not drag in torch/ultralytics. Keeps cold-start cheap for non-serving
        # entrypoints.
        from ultralytics import YOLO

        self.device = device
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.default_threshold = default_threshold
        self.class_thresholds = class_thresholds or {}
        self.calibrator = calibrator

        # We run the model at a permissive global conf and apply our *real*
        # per-class thresholds ourselves afterward. This lets one set of weights
        # serve many operating points without re-running inference, and keeps the
        # business threshold logic in our code, not Ultralytics'.
        self._infer_conf = min(
            [default_threshold, *self.class_thresholds.values()] or [default_threshold]
        )
        self._infer_conf = max(0.01, min(self._infer_conf, 0.25))

        self.model = YOLO(str(weights))
        self.model.to(device)

        if warmup:
            self._warmup()

    # ------------------------------------------------------------------ warmup

    def _warmup(self) -> None:
        """Run one dummy forward pass to amortize lazy CUDA/cuDNN init.

        The first inference on a fresh model pays kernel-compilation and memory-
        allocation costs that would otherwise land on the first *real* request
        and blow its latency budget. We eat that cost at startup instead.
        """
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy, imgsz=self.img_size, device=self.device, verbose=False
            )
        except Exception:
            # Warmup is best-effort; a failure here must not block startup.
            pass

    # ------------------------------------------------------------- thresholding

    def _threshold_for(self, cls_name: str) -> float:
        return self.class_thresholds.get(cls_name, self.default_threshold)

    def _apply_calibration(self, class_id: int, raw_conf: float) -> float:
        if self.calibrator is None:
            return raw_conf
        return float(self.calibrator.calibrate(class_id, raw_conf))

    # --------------------------------------------------------------- inference

    def predict_batch(
        self, frames: list[np.ndarray], frame_ids: list[str] | None = None
    ) -> list[DetectionResult]:
        """Primary interface: run a batch of BGR frames -> list of results.

        Ultralytics handles letterboxing internally and returns boxes already in
        original-image coordinates, so we do not re-letterbox here — but
        preprocessing.preprocess_frame remains the contract for custom runtimes
        (v2+) where we own that step.
        """
        if not frames:
            return []
        if frame_ids is not None and len(frame_ids) != len(frames):
            raise ValueError("frame_ids length must match frames length")

        t0 = time.perf_counter()
        raw_results = self.model.predict(
            frames,
            imgsz=self.img_size,
            iou=self.iou_threshold,
            conf=self._infer_conf,
            device=self.device,
            verbose=False,
        )
        # One latency number for the whole batch; divide per-frame for the record.
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_ms = batch_latency_ms / len(frames)

        results: list[DetectionResult] = []
        for i, r in enumerate(raw_results):
            fid = frame_ids[i] if frame_ids else None
            results.append(self._convert(r, per_frame_ms, fid))
        return results

    def predict(self, frame: np.ndarray, frame_id: str | None = None) -> DetectionResult:
        """Single-frame convenience wrapper over ``predict_batch``."""
        return self.predict_batch([frame], [frame_id] if frame_id else None)[0]

    def predict_stream(
        self, frames: Iterable[np.ndarray], frame_ids: Iterable[str] | None = None
    ) -> Iterator[DetectionResult]:
        """Yield results one frame at a time for async/streaming consumers.

        A generator (not a list) so a consumer — e.g. the FastAPI endpoint or the
        async inference logger — can begin handling each result without waiting
        for the whole stream. v3 backs this with a true streaming model server.
        """
        ids = list(frame_ids) if frame_ids is not None else None
        for i, frame in enumerate(frames):
            fid = ids[i] if ids else None
            yield self.predict(frame, fid)

    # ---------------------------------------------------------------- internal

    def _convert(self, raw, latency_ms: float, frame_id: str | None) -> DetectionResult:
        """Translate one Ultralytics Results object into a DetectionResult.

        This is the *only* place that touches Ultralytics' result shape. It
        applies calibration first, then the per-class operating threshold.
        """
        h, w = (int(raw.orig_shape[0]), int(raw.orig_shape[1])) if raw.orig_shape is not None else (0, 0)
        dets: list[Detection] = []

        boxes = getattr(raw, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), raw_conf, cid in zip(xyxy, confs, clss):
                if cid < 0 or cid >= NUM_CLASSES:
                    continue
                cname = class_name(int(cid))
                cal_conf = self._apply_calibration(int(cid), float(raw_conf))
                if cal_conf < self._threshold_for(cname):
                    continue  # below this class's operating point — drop it
                dets.append(
                    Detection(
                        class_id=int(cid),
                        class_name=cname,
                        confidence=cal_conf,
                        xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        raw_confidence=float(raw_conf) if self.calibrator else None,
                    )
                )

        return DetectionResult(
            detections=dets, image_shape=(h, w), latency_ms=latency_ms, frame_id=frame_id
        )
