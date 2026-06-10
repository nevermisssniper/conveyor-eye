"""Pydantic request/response models for the serving API.

WHY a schema module separate from the app?
-------------------------------------------
The wire contract is an asset with its own lifecycle: clients depend on it, it is
versioned, and it should be documentable (FastAPI/OpenAPI reads these models)
without importing the model-loading machinery. Keeping the Pydantic models here
means the contract can be imported by tests, client SDKs, or a schema-export
script without spinning up torch.

These models are deliberately *not* the internal dataclasses (DetectionResult et
al.). The internal types optimize for the code; these optimize for the wire —
explicit field names, validation, JSON-native types. The api layer maps between
the two, which is the seam that lets either side evolve independently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from conveyoreye import CLASS_NAMES


class DetectionOut(BaseModel):
    """One detected box on the wire."""

    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="xyxy pixels")
    raw_confidence: float | None = Field(
        None, description="Pre-calibration score, present only when a calibrator is attached."
    )


class FrameResult(BaseModel):
    """Per-frame inference result returned to the client."""

    frame_id: str | None = None
    detections: list[DetectionOut] = Field(default_factory=list)
    max_confidence: float = 0.0
    latency_ms: float = 0.0
    image_b64: str | None = Field(
        None, description="Optional annotated JPEG (base64) when visualize=true."
    )


class BatchInferResponse(BaseModel):
    """Response for /infer/batch."""

    results: list[FrameResult]
    n_frames: int
    model_version: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload for /health."""

    status: str
    model_loaded: bool
    device: str
    calibrator: str | None
    classes: list[str] = Field(default_factory=lambda: list(CLASS_NAMES))
    uptime_s: float


class ClassDriftOut(BaseModel):
    """One class's (or overall) drift status for /drift."""

    key: str
    severity: str
    psi: float
    kl: float
    js: float
    n_reference: int
    n_current: int
    detail: str = ""


class DriftResponse(BaseModel):
    """Response for /drift — confidence drift per key plus the window used."""

    window_s: float
    reports: list[ClassDriftOut]
    any_drift: bool


class QueueStatsResponse(BaseModel):
    """Response for /queue/stats — counts by FrameStatus."""

    counts: dict[str, int]
