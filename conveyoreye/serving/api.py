"""FastAPI serving app: inference + monitoring + active-learning queueing.

WHY this app looks the way it does
----------------------------------
The endpoint's only job on the hot path is inference + a fast response. Everything
that is *not* the answer to the client — persisting the result, checking the
low-confidence queue, computing image stats — is pushed off the request via
``BackgroundTasks`` and a periodic background loop. The client gets its boxes at
model latency; the system still gets its telemetry.

Config comes entirely from environment variables (MODEL_PATH, THRESHOLD_CONFIG,
DB_PATH, DEVICE, CALIBRATOR_PATH) so the same image runs in dev/stage/prod with
no code change — the twelve-factor pattern. Heavy, long-lived resources (the
model, the async logger, the drift detector, the queue) are created once in the
``lifespan`` context and shared, never per-request.

Endpoints:
  POST /infer/batch  — primary inference, optional annotated-image visualization
  GET  /health       — liveness/readiness
  GET  /drift        — current confidence-drift status over a recent window
  GET  /queue/stats  — labeling-queue counts

Scalability path
----------------
  v1 (here): one process, in-proc model + SQLite logger + JSON queue.
  v2: model behind a shared inference server; multiple API replicas; the logger
      points at a shared DB. The endpoints don't change.
  v3: the periodic drift task becomes a separate monitoring service consuming the
      log; the API stays purely request/response.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException

from conveyoreye import __version__, class_id
from conveyoreye.monitoring.drift import ConfidenceDriftDetector, Severity
from conveyoreye.monitoring.logger import InferenceLogger
from conveyoreye.active_learning.queue import LabelingQueue
from conveyoreye.serving.schema import (
    BatchInferResponse,
    ClassDriftOut,
    DetectionOut,
    DriftResponse,
    FrameResult,
    HealthResponse,
    QueueStatsResponse,
)

# ----------------------------------------------------------------- app state

class AppState:
    """Holds the long-lived resources shared across requests.

    A single instance is created in ``lifespan`` and stashed on ``app.state``.
    Bundling them in one object (vs scattered module globals) keeps the lifespan
    wiring readable and makes the dependencies explicit.
    """

    def __init__(self) -> None:
        self.detector = None
        self.logger: InferenceLogger | None = None
        self.drift: ConfidenceDriftDetector | None = None
        self.queue: LabelingQueue | None = None
        self.config: dict = {}
        self.device: str = "cpu"
        self.calibrator_path: str | None = None
        self.started_at: float = time.time()
        self._drift_task: asyncio.Task | None = None


def _load_config(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_detector(state: AppState):
    """Construct the Detector from env config. Imported lazily to keep import
    of this module (e.g. for schema/testing) free of torch."""
    from conveyoreye.model.detector import Detector

    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH env var is required to serve.")

    cfg = state.config
    classes_cfg = cfg.get("classes", {})
    class_thresholds = {name: c.get("threshold", cfg.get("default_threshold", 0.25))
                        for name, c in classes_cfg.items()}

    calibrator = None
    cal_path = os.environ.get("CALIBRATOR_PATH")
    if cal_path and os.path.exists(cal_path):
        import pickle
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        state.calibrator_path = cal_path

    return Detector(
        weights=model_path,
        device=state.device,
        iou_threshold=cfg.get("iou_threshold", 0.50),
        class_thresholds=class_thresholds,
        default_threshold=cfg.get("default_threshold", 0.25),
        calibrator=calibrator,
        warmup=True,
    )


# ----------------------------------------------------------------- lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create resources on startup, tear them down on shutdown.

    Using a lifespan context (not @app.on_event) guarantees the async logger's
    writer task is started *and* drained in the right order, and that the periodic
    drift task is cancelled cleanly so shutdown doesn't hang.
    """
    state = AppState()
    state.device = os.environ.get("DEVICE", "cpu")
    state.config = _load_config(os.environ.get("THRESHOLD_CONFIG"))

    # Inference logger (async writer).
    db_path = os.environ.get("DB_PATH", "conveyoreye_inference.db")
    state.logger = InferenceLogger(db_path)
    await state.logger.start()

    # Drift detector — reference is set lazily once enough traffic accrues, or by
    # an offline job; starts empty and simply reports OK until then.
    state.drift = ConfidenceDriftDetector()

    # Active-learning queue.
    queue_path = os.environ.get("QUEUE_PATH", "labeling_queue.json")
    state.queue = LabelingQueue(queue_path)

    # Detector last (heaviest; warms up).
    state.detector = _build_detector(state)

    # Periodic background drift check.
    state._drift_task = asyncio.create_task(_periodic_drift_check(state))

    app.state.ctx = state
    try:
        yield
    finally:
        if state._drift_task:
            state._drift_task.cancel()
            try:
                await state._drift_task
            except asyncio.CancelledError:
                pass
        if state.logger:
            await state.logger.stop()


async def _periodic_drift_check(state: AppState, interval_s: float = 300.0) -> None:
    """Every 5 minutes, pull the recent confidence window and evaluate drift.

    Runs out-of-band so a drift computation never sits in a request path. Logs the
    severity; v3 would publish to an alerting sink. Guarded so one failed check
    never kills the loop.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            if not (state.logger and state.drift):
                continue
            confs = await state.logger.query_confidence_window(window_s=interval_s)
            report = state.drift.check(np.array(confs))
            if report.drifted:
                # TODO(v2): emit to a real alert channel (PagerDuty/Slack). For now
                # a structured log line is the signal.
                print(f"[drift] {report.name} severity={report.severity.value} "
                      f"psi={report.psi:.3f} n={report.n_current}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let the monitor loop die silently
            print(f"[drift] check failed: {exc}")


app = FastAPI(title="ConveyorEye", version=__version__, lifespan=lifespan)


# ------------------------------------------------------------------ helpers

def _decode_image(b64: str) -> np.ndarray:
    """Decode a base64 (data-URL or raw) image into a BGR ndarray."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def _draw_and_encode(image: np.ndarray, result) -> str:
    """Draw boxes+labels on a copy and return a base64 JPEG.

    Visualization is opt-in (visualize=true) and done here, off to the side, so
    the cost is never paid by clients that only want the JSON boxes.
    """
    vis = image.copy()
    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    ok, enc = cv2.imencode(".jpg", vis)
    return base64.b64encode(enc.tobytes()).decode("ascii") if ok else ""


async def _post_inference_tasks(state: AppState, results: list, images: list[np.ndarray]) -> None:
    """Background work after a response is sent: log + enqueue low-confidence frames.

    Runs via BackgroundTasks so none of this is on the request's latency budget.
    Logging persists every frame; the active-learning hook enqueues only frames
    below the configured low_confidence_threshold, up to the per-batch budget.
    """
    from conveyoreye.monitoring.drift import ImageStatsDriftDetector

    al_cfg = state.config.get("active_learning", {})
    low_thr = al_cfg.get("low_confidence_threshold", 0.55)
    budget = al_cfg.get("queue_budget_per_batch", 20)

    enqueued = 0
    for result, image in zip(results, images):
        stats = ImageStatsDriftDetector.compute_stats(image)
        if state.logger:
            await state.logger.log(result, image_stats=stats)
        if (state.queue and result.frame_id and enqueued < budget
                and result.max_confidence < low_thr):
            added = state.queue.add(
                frame_id=result.frame_id,
                image_path=f"<live:{result.frame_id}>",  # v2: persist frame to blob store
                source="uncertainty",
                score=float(1.0 - result.max_confidence),
                note=f"max_conf={result.max_confidence:.3f} below {low_thr}",
            )
            if added:
                enqueued += 1


# ------------------------------------------------------------------ routes

@app.post("/infer/batch", response_model=BatchInferResponse)
async def infer_batch(payload: dict, background: BackgroundTasks) -> BatchInferResponse:
    """Run inference on a batch of base64 images.

    Request: {"images": [b64, ...], "frame_ids": [...]?, "visualize": bool?}
    The detector call is awaited (it is the answer); logging and AL queueing are
    deferred to background tasks so they don't extend the response.
    """
    state: AppState = app.state.ctx
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    images_b64 = payload.get("images")
    if not images_b64:
        raise HTTPException(status_code=400, detail="'images' (list of base64) required.")
    frame_ids = payload.get("frame_ids")
    visualize = bool(payload.get("visualize", False))

    images = [_decode_image(b) for b in images_b64]
    results = state.detector.predict_batch(images, frame_ids)

    out: list[FrameResult] = []
    for result, image in zip(results, images):
        out.append(FrameResult(
            frame_id=result.frame_id,
            detections=[
                DetectionOut(
                    class_id=d.class_id, class_name=d.class_name,
                    confidence=d.confidence, bbox=list(d.xyxy),
                    raw_confidence=d.raw_confidence,
                ) for d in result.detections
            ],
            max_confidence=result.max_confidence,
            latency_ms=result.latency_ms,
            image_b64=_draw_and_encode(image, result) if visualize else None,
        ))

    background.add_task(_post_inference_tasks, state, results, images)
    return BatchInferResponse(results=out, n_frames=len(out), model_version=__version__)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness. Reports whether the model and calibrator are loaded."""
    state: AppState = app.state.ctx
    return HealthResponse(
        status="ok" if state.detector is not None else "starting",
        model_loaded=state.detector is not None,
        device=state.device,
        calibrator=state.calibrator_path,
        uptime_s=time.time() - state.started_at,
    )


@app.get("/drift", response_model=DriftResponse)
async def drift(window_s: float = 3600.0) -> DriftResponse:
    """Confidence-drift status over the last ``window_s`` seconds, per class + overall."""
    state: AppState = app.state.ctx
    if not (state.logger and state.drift):
        raise HTTPException(status_code=503, detail="Monitoring not ready.")

    confs = await state.logger.query_confidence_window(window_s=window_s)
    # We do not have per-detection class ids in this lightweight query, so report
    # the overall confidence drift; per-class drift is available via check_all when
    # class ids are queried (v2 adds a class-tagged window query).
    report = state.drift.check(np.array(confs))
    out = [ClassDriftOut(
        key=report.name, severity=report.severity.value, psi=report.psi,
        kl=report.kl, js=report.js, n_reference=report.n_reference,
        n_current=report.n_current, detail=report.detail,
    )]
    return DriftResponse(
        window_s=window_s, reports=out, any_drift=report.severity is not Severity.OK
    )


@app.get("/queue/stats", response_model=QueueStatsResponse)
async def queue_stats() -> QueueStatsResponse:
    """Labeling-queue counts by status."""
    state: AppState = app.state.ctx
    if state.queue is None:
        raise HTTPException(status_code=503, detail="Queue not ready.")
    return QueueStatsResponse(counts=state.queue.stats())
