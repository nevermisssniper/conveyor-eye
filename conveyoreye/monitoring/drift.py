"""Drift detection on confidence distributions and image statistics.

WHY monitor drift at all?
-------------------------
A deployed detector silently degrades when the *input* world moves away from the
training world — a new part finish, a dirtier lens, a relit cell — long before
anyone files a quality complaint. We almost never have live ground-truth labels
to compute accuracy on, so we monitor *proxies* that move when the world moves:

  * Confidence distribution (ConfidenceDriftDetector): if the histogram of model
    confidences shifts vs a healthy reference, the model is seeing inputs it
    wasn't trained on, even if we can't say it's "wrong" yet. PSI/KL/JS quantify
    that shift.
  * Image statistics (ImageStatsDriftDetector): brightness/contrast/edge-density
    are cheap, label-free signals for lighting and focus changes upstream of the
    model entirely.

PSI is the headline because it has battle-tested operating thresholds (warn>0.1,
alert>0.2 — the credit-risk convention) that translate cleanly to "investigate"
vs "page someone". KL and JS are reported alongside for interpretability (JS is
symmetric and bounded, nice for dashboards).

Scalability path
----------------
  v1 (here): batch PSI/KL/JS over a window pulled from the inference log.
  v2: streaming/EWMA estimates so drift is detected continuously, not per poll.
  v3: multivariate drift on the detector's embedding space (detect *novel* defect
      types), not just scalar confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from conveyoreye import CLASS_NAMES, class_id

_EPS = 1e-6


class Severity(str, Enum):
    """Drift severity, mapped to an operational response."""

    OK = "ok"          # within tolerance — no action
    WARN = "warn"      # PSI > 0.1 — investigate, candidate for re-eval
    ALERT = "alert"    # PSI > 0.2 — likely material shift, consider rollback/retrain


@dataclass
class DriftReport:
    """Result of one drift check. Carries every metric, not just the verdict.

    We surface PSI, KL and JS together so a dashboard can show the divergence
    *and* the severity, and so the choice of headline metric (PSI) is auditable
    rather than buried.
    """

    name: str                 # what was checked (e.g. "confidence:crack")
    severity: Severity
    psi: float
    kl: float
    js: float
    n_reference: int
    n_current: int
    detail: str = ""

    @property
    def drifted(self) -> bool:
        return self.severity is not Severity.OK


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Normalized histogram with epsilon flooring (so KL/PSI never divide by 0)."""
    counts, _ = np.histogram(values, bins=edges)
    dist = counts.astype(np.float64)
    dist = dist / max(dist.sum(), 1.0)
    return np.clip(dist, _EPS, None)


def population_stability_index(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    """PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i) over shared bins.

    Symmetric-ish, unbounded, and with well-known thresholds (0.1 / 0.2). Both
    distributions are binned on the *same* edges (derived from the reference) so
    the comparison is apples-to-apples.
    """
    p_ref = _histogram(ref, edges)
    p_cur = _histogram(cur, edges)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def kl_divergence(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    """KL(cur || ref): how surprising current looks under the reference model."""
    p_ref = _histogram(ref, edges)
    p_cur = _histogram(cur, edges)
    return float(np.sum(p_cur * np.log(p_cur / p_ref)))


def js_divergence(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    """Jensen-Shannon divergence: symmetric, bounded [0, ln2]. Dashboard-friendly."""
    p_ref = _histogram(ref, edges)
    p_cur = _histogram(cur, edges)
    m = 0.5 * (p_ref + p_cur)
    return float(0.5 * np.sum(p_ref * np.log(p_ref / m)) + 0.5 * np.sum(p_cur * np.log(p_cur / m)))


class ConfidenceDriftDetector:
    """Tracks shift in the *confidence distribution*, overall and per class.

    Set a reference once (from a known-good period — typically the val set or the
    first stable day in production), then ``check`` recent windows against it.
    """

    def __init__(
        self,
        n_bins: int = 10,
        warn_psi: float = 0.10,
        alert_psi: float = 0.20,
    ) -> None:
        self.n_bins = n_bins
        self.warn_psi = warn_psi
        self.alert_psi = alert_psi
        # Bin edges fixed on [0,1] because confidences are probabilities — shared
        # edges across overall + every class keep all PSIs comparable.
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self._reference: dict[str, np.ndarray] = {}   # key "overall" or class name

    def set_reference(
        self, confidences: np.ndarray, class_ids: np.ndarray | None = None
    ) -> None:
        """Record the healthy baseline distribution, overall and per class.

        Storing the raw reference samples (not just the histogram) lets us re-bin
        later if we change resolution, and keeps per-class references available
        for ``check_all``.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        self._reference = {"overall": confidences}
        if class_ids is not None:
            class_ids = np.asarray(class_ids)
            for name in CLASS_NAMES:
                mask = class_ids == class_id(name)
                if mask.any():
                    self._reference[name] = confidences[mask]

    def check(self, current: np.ndarray, key: str = "overall") -> DriftReport:
        """Compare a current confidence window against the stored reference.

        Returns OK with a note if there is no reference or too little current data
        — we never raise a false alarm off a thin sample.
        """
        ref = self._reference.get(key)
        cur = np.asarray(current, dtype=np.float64)
        if ref is None or ref.size == 0:
            return DriftReport(f"confidence:{key}", Severity.OK, 0, 0, 0, 0, cur.size,
                               "no reference set")
        if cur.size < 20:
            return DriftReport(f"confidence:{key}", Severity.OK, 0, 0, 0, ref.size, cur.size,
                               "insufficient current samples")

        psi = population_stability_index(ref, cur, self.edges)
        kl = kl_divergence(ref, cur, self.edges)
        js = js_divergence(ref, cur, self.edges)
        severity = self._severity(psi)
        return DriftReport(f"confidence:{key}", severity, psi, kl, js, ref.size, cur.size)

    def check_all(
        self, current: np.ndarray, current_class_ids: np.ndarray | None = None
    ) -> dict[str, DriftReport]:
        """Run the overall check plus a per-class check for every class with a ref.

        Per-class drift is where the action usually is: aggregate confidence can
        look stable while one rare class quietly collapses. Returns a dict keyed
        by "overall" and class name.
        """
        reports = {"overall": self.check(current, "overall")}
        if current_class_ids is not None:
            current = np.asarray(current, dtype=np.float64)
            current_class_ids = np.asarray(current_class_ids)
            for name in CLASS_NAMES:
                if name not in self._reference:
                    continue
                mask = current_class_ids == class_id(name)
                reports[name] = self.check(current[mask], name)
        return reports

    def _severity(self, psi: float) -> Severity:
        if psi > self.alert_psi:
            return Severity.ALERT
        if psi > self.warn_psi:
            return Severity.WARN
        return Severity.OK


@dataclass
class ImageStatsReference:
    """Per-statistic mean/std baseline for z-score drift alerting."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)


class ImageStatsDriftDetector:
    """Label-free drift on cheap image statistics (brightness/contrast/edges).

    These proxies catch problems *upstream of the model* — a dimmed lamp, a
    smeared lens — that confidence drift would only see indirectly. Alerting is by
    z-score: how many reference standard deviations the current mean has moved.
    Computing the stats is intentionally trivial so it can run on every frame.
    """

    def __init__(self, z_warn: float = 2.0, z_alert: float = 3.0) -> None:
        self.z_warn = z_warn
        self.z_alert = z_alert
        self._ref: ImageStatsReference | None = None

    @staticmethod
    def compute_stats(image: np.ndarray) -> dict[str, float]:
        """Brightness (mean), contrast (std), edge-density (Sobel-magnitude mean).

        These three were chosen because each maps to a distinct upstream fault:
        brightness->lighting, contrast->exposure/fog, edge-density->focus/dirt.
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_density = float(np.sqrt(gx ** 2 + gy ** 2).mean())
        return {
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "edge_density": edge_density,
        }

    def set_reference(self, images: list[np.ndarray]) -> None:
        """Compute per-stat mean and std over a batch of healthy reference frames."""
        stats = [self.compute_stats(im) for im in images]
        keys = ("brightness", "contrast", "edge_density")
        means = {k: float(np.mean([s[k] for s in stats])) for k in keys}
        stds = {k: float(np.std([s[k] for s in stats]) + _EPS) for k in keys}
        self._ref = ImageStatsReference(means=means, stds=stds)

    def check(self, image: np.ndarray) -> DriftReport:
        """Z-score the current frame's stats against the reference baseline.

        Severity is driven by the single largest |z| across the three stats —
        any one statistic blowing out is enough to warrant a look.
        """
        if self._ref is None:
            return DriftReport("image_stats", Severity.OK, 0, 0, 0, 0, 0, "no reference set")
        stats = self.compute_stats(image)
        z = {
            k: abs(stats[k] - self._ref.means[k]) / self._ref.stds[k]
            for k in stats
        }
        worst_key = max(z, key=z.get)
        worst_z = z[worst_key]
        if worst_z > self.z_alert:
            sev = Severity.ALERT
        elif worst_z > self.z_warn:
            sev = Severity.WARN
        else:
            sev = Severity.OK
        detail = ", ".join(f"{k}:z={v:.2f}" for k, v in z.items())
        # We reuse DriftReport's psi slot to carry the dominant z-score so the
        # serving layer can treat both detectors' reports uniformly.
        return DriftReport("image_stats", sev, worst_z, 0.0, 0.0, 1, 1,
                           f"worst={worst_key}; {detail}")
