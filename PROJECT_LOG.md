# ConveyorEye — Project Log

Source of truth for this project. If a decision isn't here, it didn't happen.
Two sections: a **Daily Log** (Notion-mirror, what got done) and a **Decision Log**
(why, who proposed it, and whether it actually worked). Append, never rewrite history.

---

## Daily Log

Mirrors the Notion page. Format: `MM.DD`, terse bullets, meaningful contributions only.

### 06.10
- Drew up architecture (5-defect YOLOv8 detection → calibration → eval → monitoring → active learning)
- Scaffolded full package: data / model / eval / monitoring / active_learning / serving + configs + scripts
- Plugged in platforms: VS Code (author), Terminal/Mac (sim·eval·serve, CPU), Colab Pro (train, GPU), GitHub (sync)
- Verified every non-torch layer in sandbox (sim, aug round-trip, evaluator, 3 calibrators, sweeper, taxonomy, 2 drift detectors, 3 samplers, queue, async logger, FastAPI app)
- Set up project memory: PROJECT_LOG / SESSION_CONTEXT / COLLABORATION

### 06.11
- _(next session)_

---

## Decision Log

Newest at the bottom. Each entry: **what**, **why**, **proposed by**, **outcome**.

### 06.10 · Wrap Ultralytics behind typed dataclasses
- **What:** `detector.py` returns `DetectionResult`/`Detection`, never raw YOLO `Results`.
- **Why:** Decouple the whole stack from one library so v2 (ONNX/TensorRT/Triton) touches only one file.
- **Proposed by:** Claude · **Outcome:** ✅ implemented, imports verified.

### 06.10 · Deterministic data, regenerate per-platform instead of syncing images
- **What:** `simulate_data.py` is seeded; Mac and Colab both run `--seed 0` to get byte-identical sets. Only `best.pt` travels.
- **Why:** Avoids shipping 2,400 images across machines; removes a sync-drift failure mode.
- **Proposed by:** Claude · **Outcome:** ✅ adopted in runbook. Train seed 0 / val seed +10000 (no leakage).

### 06.10 · Albumentations 2.x param rename
- **What:** `pip` resolves to 2.0.x; spec's 1.4 names (`var_limit`, `quality_lower`, `max_holes`…) silently dropped as invalid kwargs. Rewrote `augmentation.yaml` to 2.x API, pinned `>=2.0`.
- **Why:** 1.4-style names → augmentation silently not applied (warnings only, no error). Dangerous.
- **Proposed by:** Claude · **Outcome:** ✅ aug round-trip clean, zero warnings after fix.

### 06.10 · mAP@50:95 left as approximation
- **What:** `metrics.py::_approx_map_50_95` returns `0.72 * map50` with a TODO for the real multi-IoU loop.
- **Why:** Real value needs re-matching at IoU 0.50→0.95; deferred as the headline learning exercise.
- **Proposed by:** Rayden (kept as exercise) · **Outcome:** 🔲 OPEN — see SESSION_CONTEXT open questions.

### 06.10 · Calibration: three methods, fit on eval's own TP/FP pairs
- **What:** `ConfidenceCalibrator` (temperature/platt/isotonic) fits on the same greedy-match pairs the evaluator produces.
- **Why:** One matching pass drives both metrics and calibration → they can't disagree about "correct".
- **Proposed by:** Claude · **Outcome:** ✅ harness smoke test (synthetic dets, NOT a real model): ECE 0.138 → temp 0.088 / platt 0.052 / isotonic 0.000. Re-measure on real val set.

<!-- TEMPLATE — copy for new entries
### MM.DD · <decision title>
- **What:**
- **Why:**
- **Proposed by:** Claude | Rayden · **Outcome:** ✅ worked | 🔲 open | ↩︎ reverted (reason)
-->
