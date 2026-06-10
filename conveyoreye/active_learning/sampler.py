"""Active-learning samplers: spend a finite labeling budget where it helps most.

WHY active learning here?
-------------------------
Labels are the scarce resource, not frames — the line produces millions of
frames, a human can label a few hundred a day. The question is *which* frames.
Two complementary signals answer it:

  * Uncertainty (UncertaintySampler): frames the model is unsure about sit near
    the decision boundary; a label there moves the boundary the most. But pure
    uncertainty over-samples redundant near-duplicates (the line jams and emits
    500 similar blurry frames — all uncertain, all the same lesson).
  * Diversity (CoresetSampler): greedy k-center over embeddings picks a spread-out
    set that covers the feature space, killing that redundancy — but ignores how
    informative each frame is.

HybridSampler is the practical answer: uncertainty pre-filter to a candidate pool,
then coreset de-duplication within it. This is the standard "informative AND
diverse" recipe and the one most production AL loops converge on.

Scalability path
----------------
  v1 (here): exact greedy k-center, O(N*k) distance work, pure NumPy.
  v2: FAISS-backed nearest-neighbor for the k-center inner loop (see TODO) once
      N exceeds ~50k candidates and the O(N*k) scan dominates.
  v3: batch-aware acquisition (BADGE-style gradient embeddings) that folds
      uncertainty and diversity into one geometry instead of two stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

UncertaintyMethod = Literal["least_confident", "entropy", "margin"]


@dataclass
class ScoredFrame:
    """A frame index with its acquisition score and the reason it scored high.

    Returning the score and method (not just the index) makes the queue's
    prioritization auditable — you can see *why* a frame was picked.
    """

    index: int
    score: float                 # higher == more worth labeling
    method: str

    def __lt__(self, other: "ScoredFrame") -> bool:  # enables sorting
        return self.score < other.score


class UncertaintySampler:
    """Frame-level uncertainty from per-detection confidences.

    A frame has many detections; we must aggregate them to one frame score. We
    aggregate by the frame's *most confident* detection's uncertainty — i.e. how
    unsure the model is about the thing it was surest of. The intuition: if even
    the best box on the frame is shaky, the whole frame is worth a look. Empty
    frames are treated as maximally uncertain (the model found nothing — possibly
    a miss).
    """

    def __init__(self, method: UncertaintyMethod = "least_confident") -> None:
        self.method = method

    def _frame_uncertainty(self, confidences: list[float]) -> float:
        if not confidences:
            return 1.0  # nothing detected -> maximally worth checking for a miss
        c = np.asarray(confidences, dtype=np.float64)
        if self.method == "least_confident":
            # Uncertainty of the most-confident detection: 1 - max(conf).
            return float(1.0 - c.max())
        if self.method == "entropy":
            # Binary entropy of the top detection (correct vs not), averaged is
            # noisy; we use the max-confidence box as the frame representative.
            p = float(c.max())
            p = min(max(p, 1e-6), 1 - 1e-6)
            return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
        if self.method == "margin":
            # Margin = gap between the two most confident detections. Small gap ->
            # the model is torn between two calls -> uncertain. One detection -> 0.
            if c.size < 2:
                return float(1.0 - c.max())
            top2 = np.sort(c)[-2:]
            return float(1.0 - (top2[1] - top2[0]))
        raise ValueError(f"Unknown uncertainty method: {self.method!r}")

    def score(self, results: list) -> list[ScoredFrame]:
        """Score every frame; higher score == more uncertain == higher priority."""
        return [
            ScoredFrame(i, self._frame_uncertainty(r.confidences()), self.method)
            for i, r in enumerate(results)
        ]

    def select(self, results: list, k: int) -> list[int]:
        """Return indices of the top-k most uncertain frames."""
        scored = self.score(results)
        scored.sort(reverse=True)
        return [s.index for s in scored[:k]]


class CoresetSampler:
    """Greedy k-center selection on L2-normalized embeddings (diversity).

    k-center greedy: repeatedly pick the frame *farthest* from everything already
    selected. This minimizes the maximum distance from any point to its nearest
    selected center — i.e. it covers the embedding space as evenly as possible,
    which is exactly what kills redundant near-duplicates.

    Embeddings are L2-normalized so Euclidean distance behaves like cosine
    distance (direction, not magnitude) — appropriate for neural feature vectors.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _l2_normalize(emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norms, 1e-9, None)

    def select(
        self, embeddings: np.ndarray, k: int, initial: list[int] | None = None
    ) -> list[int]:
        """Pick k diverse indices via greedy k-center.

        ``initial`` seeds the selection with already-labeled points so we choose
        frames that complement the *existing* training set, not just each other.
        """
        emb = self._l2_normalize(np.asarray(embeddings, dtype=np.float64))
        n = emb.shape[0]
        if k >= n:
            return list(range(n))

        selected: list[int] = list(initial or [])
        if not selected:
            selected.append(int(self.rng.integers(0, n)))

        # min_dist[i] = distance from point i to its nearest selected center.
        min_dist = np.full(n, np.inf)
        for s in selected:
            min_dist = np.minimum(min_dist, np.linalg.norm(emb - emb[s], axis=1))

        # TODO (v2 / N>50k): the np.linalg.norm broadcast below is O(N*d) per pick,
        # O(N*k*d) overall. Replace the "distance to nearest center" update with a
        # FAISS index (IndexFlatL2): add selected centers, query all points for
        # their nearest-center distance in one batched call. Keeps the same greedy
        # selection, drops the asymptotic constant enough to handle 50k+ candidates.
        while len(selected) < k:
            nxt = int(np.argmax(min_dist))
            selected.append(nxt)
            min_dist = np.minimum(min_dist, np.linalg.norm(emb - emb[nxt], axis=1))

        # Drop any seed indices we were given — caller already has those labeled.
        return [i for i in selected if initial is None or i not in set(initial)][:k]


class HybridSampler:
    """Uncertainty pre-filter -> coreset de-duplication.

    The two-stage recipe: take the top ``uncertainty_pool_mult * k`` most
    *uncertain* frames (cheap, informative), then run coreset over *that pool* to
    pick ``k`` that are also mutually *diverse*. Weights let you tune the pool
    size — a bigger pool leans toward diversity, a smaller one toward raw
    uncertainty.
    """

    def __init__(
        self,
        uncertainty_method: UncertaintyMethod = "least_confident",
        uncertainty_pool_mult: float = 3.0,
        seed: int | None = None,
    ) -> None:
        self.uncertainty = UncertaintySampler(uncertainty_method)
        self.coreset = CoresetSampler(seed=seed)
        self.uncertainty_pool_mult = uncertainty_pool_mult

    def select(
        self,
        results: list,
        embeddings: np.ndarray,
        k: int,
        initial: list[int] | None = None,
    ) -> list[int]:
        """Select k frames that are both uncertain and diverse.

        embeddings must be row-aligned with results (one embedding per frame).
        """
        if len(results) != len(embeddings):
            raise ValueError("results and embeddings must be row-aligned")
        n = len(results)
        if k >= n:
            return list(range(n))

        pool_size = min(n, int(self.uncertainty_pool_mult * k))
        uncertain_idx = self.uncertainty.select(results, pool_size)

        # Coreset within the uncertain pool; map local indices back to global.
        pool_emb = embeddings[uncertain_idx]
        local = self.coreset.select(pool_emb, k, initial=None)
        return [uncertain_idx[i] for i in local]
