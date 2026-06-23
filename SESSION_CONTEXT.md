# ConveyorEye — Session Context

**Paste this whole file at the start of a new chat.** It's a living snapshot —
overwrite the metrics/hyperparams/open-questions sections as they change; don't
append history here (that's PROJECT_LOG.md's job).

Last updated: **06.19**

---

## Non-obvious constraints (restate these — they're easy to forget)

- **`--device` on Mac:** `finetune.py` and `train.py` default to `--device cpu`. Colab GPU runs require `--device 0` explicitly — omitting it on Mac raises `ValueError: Invalid CUDA 'device=0'`.

- **Class priors are imbalanced by design:** scratch 40 / crack 22 / dent 18 / discoloration 12 / missing_part 8 (%). Don't "fix" the imbalance — it's the point.
- **Asymmetric thresholds:** crack = recall-biased (safety; miss is unacceptable). missing_part = precision-biased (a false line-stop is expensive). Cosmetic classes = balanced F1.
- **Data is deterministic by seed.** Train seed `0`, val seed `0 + 10000`. Same seed ⇒ identical frames. Never train and eval on the same seed (leakage).
- **`mAP@50:95` is currently a placeholder** (`0.72 × map50`). Don't trust it until the real loop lands.

---

## Architecture (text diagram + dims)

```
frame (H×W×3, BGR uint8)
  └─ preprocess_frame: letterbox → 640×640×3, /255 → CHW float32  [+ scale, pad for box-reversal]
       └─ Detector (YOLOv8n, COCO-pretrained, fine-tuned)
            outputs: boxes (N,4 xyxy px) · conf (N,) · cls (N,) ∈ {0..4}
              └─ ConfidenceCalibrator.calibrate(cls, conf)   [optional, per-class]
                   └─ per-class operating threshold (thresholds.yaml)
                        └─ DetectionResult{ Detection[], latency_ms, frame_id }
                             ├─ eval:   DetectionEvaluator (greedy IoU@0.5 match) → EvalReport
                             │           FailureTaxonomy → FP/FN types → recommendations
                             ├─ monitor: InferenceLogger (async SQLite) → drift (PSI/KL/JS, img-stats z)
                             └─ active:  Uncertainty→Coreset (Hybrid) → LabelingQueue → merge_to_train
```

5 classes (id order is load-bearing): `0 scratch · 1 crack · 2 dent · 3 discoloration · 4 missing_part`.

---

## Active hyperparameters

| knob | value | where | note |
|---|---|---|---|
| base weights | `yolov8n.pt` | train.py | transfer from COCO |
| img size | 640 | train/infer | letterboxed |
| epochs | 50 | train.py | `patience=15` early-stop |
| batch | 16 local / 32 Colab | train.py | bump on GPU VRAM |
| lr0 / lrf | 0.005 / 0.01 | train.py | gentle: adapt, don't clobber backbone |
| warmup_epochs | 3 | train.py | |
| close_mosaic | 10 | train.py | clean final epochs |
| IoU (NMS + match) | 0.50 | thresholds.yaml | one value, infer + metric agree |
| per-class thresh | scratch .45 / crack .25 / dent .40 / disc .50 / missing .60 | thresholds.yaml | asymmetric (see constraints) |
| AL low-conf | 0.55 | thresholds.yaml | queue trigger |
| AL budget/batch | 20 | thresholds.yaml | |

---

## Latest metrics & trends

| run | date | mAP@50 | mAP@50:95* | crack recall | missing_part P | ECE (post-cal) | notes |
|---|---|---|---|---|---|---|---|
| run1 | 06.15 | 0.989 | 0.924* | 0.959 | 1.000 | 0.0255 (from 0.0540) | yolov8n, 50ep/3.4min Colab Blackwell, seed0; 183 FP / 9 FN |
| ft1 | 06.19 | 0.985 | — | — | — | not recalibrated | finetune on 2000 synth + 17 real crack; freeze=10; AdamW 0.001111 (lr0 ignored by optimizer=auto); early stopped ep7, best ep2 |

\* approximated until the real loop lands.

**MVTec eval (ft1):** 100% crack recall, 100% FPR on good tiles — confidence distributions overlap completely (crack max ~0.068, good max ~0.091). Root cause: 17 positives too few; frozen backbone never learned real texture discrimination. Fix: add 230 real good tiles as background negatives + `freeze=4`.

**Next (round 2 fine-tune):**
1. `python scripts/prep_real_good.py` — copies 230 good tiles → `data/real/images/good/` with empty labels
2. Colab: `freeze=4, lr0=0.0002, optimizer='SGD', epochs=30, patience=8` → `conveyoreye_ft2/weights/best.pt`
3. `python scripts/eval_mvtec.py --weights runs/conveyoreye_ft2/weights/best.pt --calibrator none` — target: good FPR < 50%
4. If step 3 looks good: `python scripts/evaluate.py --weights ... --data data/raw/dataset.yaml --device cpu --calib-method temperature`

---

## Open questions / known issues

1. **mAP@50:95 real loop** — re-match at IoU 0.50→0.95 (10 steps), average. `metrics.py` TODO. Highest-value exercise.
2. **FAISS coreset** — `sampler.py` greedy k-center is O(N·k); swap inner loop for `IndexFlatL2` at N>50k.
3. **Live frame persistence for AL** — `api.py` queues `<live:id>` placeholders; need a blob store so queued frames are actually labelable.
4. **ultralytics version parity** — Colab may resolve newer than the local pin. Match versions before loading `best.pt` locally (`import ultralytics; ultralytics.__version__`).
5. **Calibration on real data** — synthetic ECE looks great; re-measure on ft2 model's val set after round 2.
6. **`configs/thresholds.generated.yaml` eval overrides** — crack: 0.03 (was 0.12), missing_part: 0.01 (was 0.04) — these are eval-only; production thresholds not changed yet. Update after ft2 recalibration.
7. **If ft2 FPR still bad** — 17 crack positives may be fundamentally insufficient. Next lever: add MVTec `tile/test/crack/` images to val, or source more real crack images.
