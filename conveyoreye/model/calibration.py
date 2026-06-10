"""Confidence calibration for detection scores.

WHY calibrate at all?
---------------------
A detector's raw confidence is an *ordering* signal, not a probability. YOLO
typically reports 0.9 for boxes that are correct far less than 90% of the time —
it is over-confident. That breaks everything that treats confidence as a
probability: per-class operating thresholds (thresholds.yaml), the
low-confidence active-learning queue, and confidence-distribution drift
monitoring all assume "0.7 means roughly 70% likely correct". Calibration makes
that assumption true.

Three methods, increasing flexibility, all per-class (because each defect class
has its own miscalibration profile — rare classes are usually the worst):

  * temperature : one scalar T per class; divides logits. Cannot change ranking,
    only sharpness. Minimal params -> robust with little validation data. We fit
    T by minimizing NLL with scipy ``minimize_scalar``.
  * platt       : 1-D logistic regression (a, b) on the score. Can shift *and*
    scale; more flexible than temperature, still 2 params.
  * isotonic    : non-parametric monotonic fit. Most flexible, can correct
    arbitrary monotonic miscalibration, but needs the most data and can overfit.

The class is pickle-serializable so ``evaluate.py`` fits it once on the val set
and the serving layer loads it via CALIBRATOR_PATH — fit offline, apply online.

Scalability path
----------------
  v1 (here): scalar/1-D calibration per class on confidence alone.
  v2: feature-conditioned calibration (box size, image brightness) since
      miscalibration correlates with object scale and lighting.
  v3: online recalibration that refits T on a rolling window when drift fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from conveyoreye import CLASS_NAMES, NUM_CLASSES

_EPS = 1e-7
_VALID_METHODS = ("temperature", "platt", "isotonic")


@dataclass
class ReliabilityResult:
    """Output of a reliability diagram: the curve plus a scalar ECE.

    Returned as a dataclass (not a bare float) so the caller gets the binned
    points to plot *and* the headline Expected Calibration Error in one object.
    """

    bin_centers: np.ndarray       # (B,) mean predicted confidence per bin
    bin_accuracy: np.ndarray      # (B,) empirical accuracy per bin
    bin_counts: np.ndarray        # (B,) sample count per bin
    ece: float                    # Expected Calibration Error (weighted)


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ConfidenceCalibrator:
    """Per-class confidence calibrator. Fit on (confidence, correct) pairs.

    A "correct" label is whether a detection at that confidence was a true
    positive (matched a ground-truth box at the eval IoU). The evaluator produces
    exactly these pairs, so calibration is fit from the same matching used for
    metrics — no separate bookkeeping.
    """

    def __init__(self, method: str = "temperature") -> None:
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.method = method
        self._fitted = False
        # Per-class parameters, keyed by class id.
        self._temperature: dict[int, float] = {}
        self._platt: dict[int, tuple[float, float]] = {}     # (a, b)
        self._isotonic: dict[int, object] = {}               # sklearn IsotonicRegression

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        class_ids: np.ndarray,
    ) -> "ConfidenceCalibrator":
        """Fit one calibrator per class from pooled (conf, correct, class) arrays.

        Classes with too few samples to fit reliably fall back to identity
        (temperature T=1), because a badly-fit calibrator is worse than none —
        this is the rare-class safety valve.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        correct = np.asarray(correct, dtype=np.float64)
        class_ids = np.asarray(class_ids, dtype=np.int64)

        for cid in range(NUM_CLASSES):
            mask = class_ids == cid
            c, y = confidences[mask], correct[mask]
            if c.size < 10 or y.sum() == 0 or y.sum() == y.size:
                # Not enough signal (or all-correct/all-wrong) -> identity.
                self._temperature[cid] = 1.0
                self._platt[cid] = (1.0, 0.0)
                self._isotonic[cid] = None
                continue
            if self.method == "temperature":
                self._temperature[cid] = self._fit_temperature(c, y)
            elif self.method == "platt":
                self._platt[cid] = self._fit_platt(c, y)
            else:
                self._isotonic[cid] = self._fit_isotonic(c, y)

        self._fitted = True
        return self

    def _fit_temperature(self, conf: np.ndarray, correct: np.ndarray) -> float:
        """Find T>0 minimizing NLL of correctness under sigmoid(logit/T).

        Temperature scaling on a 1-D score: we treat each detection's confidence
        as a probability, convert to a logit, and search for the single T that
        makes the rescaled probabilities best predict correctness in NLL terms.
        """
        logits = _to_logit(conf)

        def nll(t: float) -> float:
            t = max(t, _EPS)
            p = _sigmoid(logits / t)
            p = np.clip(p, _EPS, 1 - _EPS)
            return float(-(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean())

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        return float(res.x)

    def _fit_platt(self, conf: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
        """Platt scaling: 1-D logistic regression of correctness on the logit."""
        from sklearn.linear_model import LogisticRegression

        x = _to_logit(conf).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x, correct.astype(int))
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def _fit_isotonic(self, conf: np.ndarray, correct: np.ndarray):
        """Non-parametric monotonic calibration via sklearn IsotonicRegression."""
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(conf, correct)
        return iso

    # --------------------------------------------------------------- transform

    def calibrate(self, class_id: int, confidence: float) -> float:
        """Map one raw confidence to a calibrated probability for its class.

        Falls back to the identity if unfitted or the class was never fit, so the
        detector can always call this unconditionally.
        """
        if not self._fitted:
            return confidence
        if self.method == "temperature":
            t = self._temperature.get(class_id, 1.0)
            return float(_sigmoid(_to_logit(np.array([confidence]))[0] / max(t, _EPS)))
        if self.method == "platt":
            a, b = self._platt.get(class_id, (1.0, 0.0))
            return float(_sigmoid(a * _to_logit(np.array([confidence]))[0] + b))
        iso = self._isotonic.get(class_id)
        if iso is None:
            return confidence
        return float(iso.predict([confidence])[0])

    def calibrate_array(self, class_ids: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Vectorized calibration over parallel class-id / confidence arrays."""
        return np.array(
            [self.calibrate(int(c), float(p)) for c, p in zip(class_ids, confidences)]
        )

    # ------------------------------------------------------------- diagnostics

    def reliability_diagram(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10,
    ) -> ReliabilityResult:
        """Bin predictions by confidence and compare to empirical accuracy.

        ECE = sum_b (n_b/N) * |acc_b - conf_b|. A perfectly calibrated model lies
        on the diagonal (conf == accuracy) and has ECE 0. Call this on raw scores
        before fit and on ``calibrate_array`` output after to *measure* the
        improvement, not just assume it.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        correct = np.asarray(correct, dtype=np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers, accs, counts = [], [], []
        ece = 0.0
        n = max(1, confidences.size)
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Last bin is closed on the right so conf==1.0 is counted.
            in_bin = (confidences > lo) & (confidences <= hi) if hi < 1.0 else (
                (confidences > lo) & (confidences <= hi + _EPS)
            )
            cnt = int(in_bin.sum())
            if cnt == 0:
                centers.append((lo + hi) / 2); accs.append(0.0); counts.append(0)
                continue
            conf_mean = float(confidences[in_bin].mean())
            acc_mean = float(correct[in_bin].mean())
            centers.append(conf_mean); accs.append(acc_mean); counts.append(cnt)
            ece += (cnt / n) * abs(acc_mean - conf_mean)

        return ReliabilityResult(
            bin_centers=np.array(centers),
            bin_accuracy=np.array(accs),
            bin_counts=np.array(counts),
            ece=float(ece),
        )

    # ------------------------------------------------------------- persistence

    def __getstate__(self) -> dict:
        # Plain-Python state only -> safe, portable pickle. (sklearn estimators
        # in _isotonic are themselves picklable.)
        return {
            "method": self.method,
            "_fitted": self._fitted,
            "_temperature": self._temperature,
            "_platt": self._platt,
            "_isotonic": self._isotonic,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
