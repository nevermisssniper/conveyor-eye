# ConveyorEye

Industrial defect detection on a simulated conveyor belt, built on YOLOv8. This
is a **learning-oriented** reference stack — end-to-end from data simulation
through serving and active learning — written so every interface is shaped for
scale even where the v1 implementation is intentionally a toy.

The point of the project is the *seams between* stages: how calibration feeds
operating thresholds, how the same TP/FP matching drives both metrics and
calibration, how drift monitoring and active learning close the loop back to
training. Each module's top docstring explains *why* it's designed the way it is.

## Working docs

- **`PROJECT_LOG.md`** — source of truth: daily log (Notion-mirror) + decision log with reasoning and outcomes.
- **`SESSION_CONTEXT.md`** — paste at the start of every chat: architecture, hyperparameters, latest metrics, open questions.
- **`COLLABORATION.md`** — how Claude and Rayden work on this (verification habits, where to lean / not lean).

## Defect classes

Five classes, **imbalanced by design** so per-class metrics, calibration, and
active learning actually matter:

| class | prior | threshold bias |
|---|---|---|
| `scratch` | 40% | balanced F1 |
| `crack` | 22% | **recall** (safety-critical) |
| `dent` | 18% | mild recall |
| `discoloration` | 12% | precision |
| `missing_part` | 8% | **precision** (false line-stop is expensive) |

## Run order

```bash
# 1. install (editable, with dev extras)
pip install -e ".[dev]"

# 2. generate the synthetic dataset (+ dataset.yaml for Ultralytics)
python scripts/simulate_data.py --n-train 2000 --n-val 400

# 3. fine-tune YOLOv8 (transfer learning from COCO)
python scripts/train.py --data data/raw/dataset.yaml --epochs 50

# 4. full eval: metrics + failure taxonomy + calibration fit + threshold export
python scripts/evaluate.py --weights runs/conveyoreye/weights/best.pt \
    --data data/raw/dataset.yaml

# 5. serve (env-driven config; this launcher just sets the env + starts uvicorn)
python scripts/run_api.py --model runs/conveyoreye/weights/best.pt \
    --calibrator calibrator.pkl
```

## Layout

```
conveyoreye/
  data/          simulator, preprocessing (train aug + inference letterbox), dataset
  model/         detector (typed YOLOv8 wrapper), calibration (temp/Platt/isotonic)
  eval/          metrics (mAP + operating-point), threshold sweep, failure taxonomy
  monitoring/    async SQLite inference logger, confidence + image-stat drift
  active_learning/  uncertainty/coreset/hybrid samplers, persisted labeling queue
  serving/       FastAPI app (env config, lifespan, background tasks), Pydantic schema
configs/         augmentation.yaml (physical rationale), thresholds.yaml (business cost)
scripts/         simulate_data, train, evaluate, run_api
```

## What to learn from each layer

- **Detection pipeline** — `model/detector.py` draws a hard boundary around
  Ultralytics; `calibration.py` shows why raw YOLO confidence isn't a probability
  and three ways to fix it.
- **Real-world evaluation** — `eval/metrics.py` keeps mAP (ranking) and
  precision/recall-at-threshold (deployment) explicitly separate; `failure_taxonomy.py`
  turns errors into a worklist.
- **Monitoring** — `monitoring/logger.py` keeps disk latency off the hot path;
  `drift.py` watches confidence (PSI/KL/JS) and image stats without needing labels.
- **Active learning** — `active_learning/sampler.py` combines uncertainty and
  coreset diversity; `queue.py` persists the human-in-the-loop lifecycle.

## Key TODOs (the instructive implementations)

These are marked `# TODO:` in-code and are the most valuable exercises:

- **mAP@50:95 proper loop** (`eval/metrics.py`) — currently approximated; implement
  the real multi-IoU averaging.
- **FAISS coreset** (`active_learning/sampler.py`) — swap the O(N·k) greedy
  k-center inner loop for a FAISS index at N > 50k.
- **Live frame persistence for AL** (`serving/api.py`) — store the actual frame to
  a blob store so queued frames are labelable, not just referenced.

## Scalability notes

Every module's docstring sketches a v1 → v2 → v3 path. The recurring theme:
batch is the primary unit, config lives in YAML not code, core methods return
dataclasses not dicts, and the third-party model is wrapped behind one file so it
can be swapped for ONNX/TensorRT/Triton without touching the rest of the stack.
