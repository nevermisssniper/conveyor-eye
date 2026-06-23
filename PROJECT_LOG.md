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
- _(skipped — see 06.15)_

### 06.15
- Generated full local dataset (2000 train / 400 val, seed 0, deterministic)
- Trained YOLOv8n on Colab (RTX PRO 6000 Blackwell, 50 epochs, 3.4 min, batch 32) → mAP@50 0.989, mAP@50:95 0.924 (still via `_approx_map_50_95`, see open TODO)
- Ran `evaluate.py` on CPU: fit temperature calibrator, ECE 0.0540 → 0.0255 (−52%), exported `configs/thresholds.generated.yaml`
- Failure taxonomy on val: 183 FP (mostly localization/background), 9 FN (scratch/crack)
- Brought up FastAPI inference server locally (`/health`, `/infer`, `/infer/batch`), health check green on CPU, ~200ms/img
- Downloaded MVTec AD **Tile** category (`tile/`) for real-data validation: 117 test images across `crack/glue_strip/gray_stroke/oil/rough/good` + ground-truth masks, 230 `train/good`
- Framed project explicitly as a practice mock-up of an Everest Labs–style CV pipeline — see Decision Log for fiscal/labor/success/scalability framing
- Wrote `PROGRESS.md` as a full session log (metrics, commands, file structure, key learnings)

### 06.19

**MVTec eval — Option A (OOD-as-FP probe)**
- Ran `eval_mvtec.py` on original `best.pt`: 5/17 crack detected (29% recall), 97% FPR on good images — discoloration firing everywhere
- Root cause: `discoloration` threshold was 0.01; raised to 0.40 in `configs/thresholds.generated.yaml`
- Ran Option A eval via `scripts/eval_mvtec.py`: crack=TP, good=TN, OOD (glue/gray/oil/rough)=FP probes

**Real-data fine-tuning (round 1)**
- Wrote `scripts/convert_mvtec_labels.py`: converted 17 MVTec crack PNG masks → YOLO bounding boxes (`data/real/labels/crack/`); note most boxes are nearly full-tile (h≈1.0 or w≈1.0 — real cracks span the whole surface)
- Created `data/finetune.yaml`: mixed 2000 synthetic + 17 real crack images
- Ran fine-tune on Colab (L4 GPU, 20 epochs): `freeze=10`, `lr0=0.0005` (overridden to AdamW 0.001111 by `optimizer=auto` — LR did NOT stick, see Decision Log); EarlyStopping triggered at epoch 7, best checkpoint epoch 2
- Fine-tune result: mAP@50 0.985 (regression from 0.989, expected — 17 images is noise at this scale), no catastrophic forgetting on synthetic classes

**Inference floor bug + diagnosis**
- Fine-tuned model returned 0 detections on all 117 MVTec images; traced to `Detector._infer_conf = max(0.01, min(class_thresholds.values()))` = 0.04 (driven by `missing_part=0.04`); YOLO filtered everything below that
- Confirmed via direct YOLO inference at `conf=0.0001`: model fires crack class on crack images at 0.02–0.065 — correct class, just low raw confidence (sim-to-real domain gap)
- Fix: lowered `crack: threshold: 0.03` and `missing_part: threshold: 0.01` in thresholds yaml for eval only (production thresholds unchanged)

**Round 1 eval result: 100% crack recall, 100% FPR**
- At threshold 0.03 all 17 crack images detected; all 33 good images also fired crack
- CSV analysis: crack image max confidence ~0.068, good image max confidence ~0.091 (`good/016.png`) — distributions overlap completely, no separating threshold exists
- Root cause: 17 crack positives is 0.8% of training set; frozen backbone never learned to distinguish real crack from real non-crack tile texture; model outputs weak crack activation on all tile textures uniformly

**Plan for round 2 (next session)**
- Added `scripts/prep_real_good.py`: copies 230 `tile/train/good/` images → `data/real/images/good/` with empty YOLO labels (background negatives)
- Updated `data/finetune.yaml` to include `data/real/images/good` as third train source
- Round 2 hyperparams: `freeze=4` (unfreeze mid-level texture layers 4–9), `lr0=0.0002`, `epochs=30`

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

### 06.15 · Synthetic-only training validated end-to-end
- **What:** 50 epochs on the 2000-image synthetic set hit mAP@50 0.989 / per-class AP ≥0.972 (dent & missing_part = 1.000), with mAP@50:95 still the `0.72×map50` approximation.
- **Why:** Confirms the full pipeline (sim → train → calibrate → serve) is wired correctly before spending effort on real data.
- **Proposed by:** Claude (ran the steps) · **Outcome:** ✅ pipeline works. mAP@50:95 number is NOT trustworthy until the real multi-IoU loop (open TODO) lands — don't cite it externally.

### 06.15 · MVTec AD Tile downloaded as the real-data validation set
- **What:** Pulled the Tile category (117 test imgs across crack/glue_strip/gray_stroke/oil/rough/good, 230 train/good, pixel masks) into `tile/`.
- **Why:** Cheapest real-world sanity check for a synthetic-trained model — texture surface, small dataset, masks included.
- **Proposed by:** Rayden · **Outcome:** 🔲 OPEN — taxonomy mismatch: only `crack` overlaps our 5 classes (scratch/crack/dent/discoloration/missing_part). `glue_strip`/`gray_stroke`/`oil`/`rough` won't map to existing classes — decide whether to (a) treat as OOD/background test only, or (b) remap to nearest class for a rough transfer check.

### 06.19 · `optimizer=auto` overrides explicit `lr0` in Ultralytics 8.4.71
- **What:** `finetune.py` set `lr0=0.0005` but Ultralytics logged `optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005'` and used AdamW at lr=0.001111 instead.
- **Why:** `optimizer='auto'` (the default) selects optimizer AND its own LR schedule, ignoring `lr0`. The run still converged fine (early stopped at epoch 7), but we didn't get the conservative LR we wanted.
- **Proposed by:** Discovered during fine-tune · **Outcome:** 🔲 OPEN — for round 2, add `optimizer='SGD'` explicitly to make `lr0=0.0002` stick.

### 06.19 · Real-data fine-tuning fails to separate crack vs. non-crack at real-image confidence
- **What:** After fine-tuning on 17 real crack images, the model outputs crack class detections at 0.02–0.068 on BOTH crack and good tile images — distributions overlap completely. No threshold can separate TP from FP.
- **Why:** 17 positives = 0.8% of training data; `freeze=10` means backbone never adapted to real tile textures; the model learned "something crack-like fires weakly at real-domain scale" not "crack specifically looks like this."
- **Proposed by:** Diagnosed from 06.19 eval CSV · **Outcome:** 🔲 OPEN — Fix: add 230 real `good` tiles as background negatives + `freeze=4` to let mid-level texture layers adapt. See round 2 plan in 06.19 daily log.

### 06.15 · Project framed as an Everest Labs CV pipeline practice run, optimized for fiscal/labor/success scalability
- **What:** Treat ConveyorEye as a scaled-down stand-in for a production CV project: explicit cost ceiling (Colab Pro + local CPU only, no paid GPU beyond what's already on hand), time-boxed labor (core TODOs — mAP@50:95 loop, FAISS coreset — are fixed-effort exercises, not open-ended), and success criteria stated up front rather than discovered after the fact (target: real mAP@50 ≥0.90 on MVTec-overlapping classes within current compute budget). Scalability is already handled structurally (typed dataclasses, config-driven thresholds, detector wrapped behind one file per README) — the practice run's job is to *exercise* those seams with real data, not redesign them.
- **Why:** Rayden's incoming role (Everest Labs ML eng intern, Summer 2026) makes this project most valuable if it mirrors how a CV team actually scopes work — cost-aware, time-boxed, success-criteria-first — rather than open-ended tinkering.
- **Proposed by:** Rayden · **Outcome:** 🔲 OPEN — adopt as the lens for all future scoping decisions on this project; revisit success criteria once MVTec taxonomy question above is resolved.

<!-- TEMPLATE — copy for new entries
### MM.DD · <decision title>
- **What:**
- **Why:**
- **Proposed by:** Claude | Rayden · **Outcome:** ✅ worked | 🔲 open | ↩︎ reverted (reason)
-->
