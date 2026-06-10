# ConveyorEye — Session Context

**Paste this whole file at the start of a new chat.** It's a living snapshot —
overwrite the metrics/hyperparams/open-questions sections as they change; don't
append history here (that's PROJECT_LOG.md's job).

Last updated: **06.10**

---

## Non-obvious constraints (restate these — they're easy to forget)

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

**No real model trained yet** — pending first Colab run. Fill in after step 4 (`evaluate.py`).

| run | date | mAP@50 | mAP@50:95* | crack recall | missing_part P | ECE (post-cal) | notes |
|---|---|---|---|---|---|---|---|
| _baseline_ | — | — | — | — | — | — | yolov8n, 50ep, seed0 |

\* approximated until the real loop lands.

---

## Open questions / known issues

1. **mAP@50:95 real loop** — re-match at IoU 0.50→0.95 (10 steps), average. `metrics.py` TODO. Highest-value exercise.
2. **FAISS coreset** — `sampler.py` greedy k-center is O(N·k); swap inner loop for `IndexFlatL2` at N>50k.
3. **Live frame persistence for AL** — `api.py` queues `<live:id>` placeholders; need a blob store so queued frames are actually labelable.
4. **ultralytics version parity** — Colab may resolve newer than the local pin. Match versions before loading `best.pt` locally (`import ultralytics; ultralytics.__version__`).
5. **Calibration on real data** — synthetic ECE looks great; re-measure on the trained model's val set, watch isotonic for overfit on rare classes.
