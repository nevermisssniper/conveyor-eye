# ConveyorEye — Progress Log

**Last updated:** 2026-06-15  
**Status:** API serving live (Step 5 complete)  
**Model:** YOLOv8n detector + temperature calibration  
**Classes:** scratch, crack, dent, discoloration, missing_part

---

## Completed Workflow

### Step 0: Repository Setup ✅
- Initialized git repo locally
- Pushed to GitHub: `https://github.com/nevermisssniper/conveyor-eye`
- Branch: `main`

### Step 1: Local Environment ✅
```bash
cd "/Users/raydenkhuraijam/Claude/Projects/Conveyor Eye"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
**Status:** Dependencies installed, environment ready for local development.

### Step 2: Dataset Generation (Local) ✅
```bash
python scripts/simulate_data.py --n-train 2000 --n-val 400 --seed 0
```
**Output:**
- 2000 training images (synthetic, deterministic)
- 400 validation images
- 5 damage classes with pixel-level annotations
- `data/raw/dataset.yaml` created

### Step 3: Training on Colab GPU ✅
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (97GB VRAM)  
**Runtime:** 3.4 minutes for 50 epochs  
**Batch size:** 32  

**Final metrics:**
- **mAP@50:** 0.989 (excellent)
- **mAP@50-95:** 0.924 (very strong)
- **Mean F1:** 0.974 across all classes

**Per-class AP:**
| Class | AP | P | R | F1 |
|-------|----|----|-----|-----|
| scratch | 0.976 | 0.973 | 0.920 | 0.946 |
| crack | 0.972 | 0.916 | 0.959 | 0.937 |
| dent | 1.000 | 1.000 | 1.000 | 1.000 |
| discoloration | 0.990 | 0.990 | 0.990 | 0.990 |
| missing_part | 1.000 | 1.000 | 1.000 | 1.000 |

**Model saved:** `runs/detect/runs/conveyoreye/weights/best.pt` (6.3 MB)

### Step 4: Evaluation & Calibration (CPU) ✅
```bash
python scripts/evaluate.py \
  --weights runs/conveyoreye/weights/best.pt \
  --data data/raw/dataset.yaml --device cpu \
  --calib-method temperature
```

**Calibration results:**
- Expected Calibration Error (ECE): 0.0540 → 0.0255 (52% improvement)
- Method: Temperature scaling
- **Output:** `calibrator.pkl`

**Per-class confidence thresholds (generated):**
| Class | Threshold | P | R |
|-------|-----------|-------|-----|
| scratch | 0.390 | 0.963 | 0.938 |
| crack | 0.120 | 0.872 | 0.973 |
| dent | 0.250 | 1.000 | 1.000 |
| discoloration | 0.010 | 0.979 | 0.990 |
| missing_part | 0.040 | 0.963 | 1.000 |

**Output:** `configs/thresholds.generated.yaml`

**Failure analysis:**
- FP: 183 (mostly poor localization, some background confusion)
- FN: 9 (mostly on scratch/crack; dent/missing_part perfect)
- **Recommendations:** Tighten NMS IoU, add hard negatives, increase localization augmentation

### Step 5: API Deployment (CPU) ✅
```bash
python scripts/run_api.py \
  --model runs/conveyoreye/weights/best.pt \
  --calibrator calibrator.pkl --device cpu
```

**Server status:** Running on `http://0.0.0.0:8000`

**Health check response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "calibrator": "calibrator.pkl",
  "classes": ["scratch", "crack", "dent", "discoloration", "missing_part"],
  "uptime_s": 24.6
}
```

**Available endpoints:**
- `GET /health` — server status
- `POST /infer` — single image inference
- `POST /infer/batch` — batch inference (multiple images)

**Input format:**
```json
{
  "image": "<base64-encoded JPEG>",
  "frame_id": "string",
  "visualize": true
}
```

**Output:** Detections with bounding boxes, calibrated confidence scores, per-class thresholds applied.

---

## Next Steps

### Immediate (testing)
1. **Download MVTec AD Tile dataset** (335 MB)
   - Contains 147 test images with surface defects/anomalies
   - Use for real-world validation of synthetic-trained model
   - Test inference endpoint with diverse surface patterns

2. **Inference testing**
   ```bash
   # Single image
   curl -X POST localhost:8000/infer \
     -H "Content-Type: application/json" \
     -d '{"image": "<base64>", "frame_id": "tile_001", "visualize": true}'
   ```

### Short-term (model improvement)
1. Mine hard negatives from MVTec (background FP source)
2. Rebalance scratch/crack via WeightedRandomSampler (confusion analysis)
3. Fine-tune on mixed synthetic + real data
4. Increase image size (640 → 960) for small object detection

### Medium-term (production)
1. Containerize API (Docker)
2. Add request logging / inference monitoring
3. Implement auto-retraining pipeline (active learning)
4. Deploy to edge device (Jetson Orin Nano)
5. Integrate with conveyor control system

---

## Architecture Summary

**Model:** YOLOv8 Nano (3M parameters)
- Pretrained on COCO, fine-tuned on 2000 synthetic images
- Input: 640×640 RGB
- Output: Bounding boxes + class confidence

**Calibration:** Temperature scaling
- Fits calibration temperature on val set
- Reduces overconfidence in predictions
- Applied during inference

**Inference:** CPU-based (FastAPI + Ultralytics)
- ~200ms per image (640×640)
- 97.2% precision at 0.025 ECE
- Per-class confidence thresholds for operational recall targets

**Deployment:** Uvicorn server
- Handles concurrent requests
- State: model weights + calibrator in memory
- No database dependency

---

## Key Learnings

1. **Synthetic data works.** 2000 generated images → 0.989 mAP on val split
2. **Calibration matters.** ECE improved 52%; confidence scores now trustworthy
3. **Class imbalance visible.** dent/missing_part perfect; scratch/crack need hard negatives
4. **Temperature scaling is cheap.** One forward pass + scalar optimization; massive confidence gain
5. **YOLOv8n is efficient.** 50 epochs in 3.4 min on Blackwell; CPU inference at 200ms/img

---

## File Structure

```
Conveyor Eye/
├── scripts/
│   ├── simulate_data.py          # Synthetic dataset generation
│   ├── train.py                   # YOLOv8 training script
│   ├── evaluate.py                # Evaluation + calibration
│   └── run_api.py                 # FastAPI inference server
├── configs/
│   └── thresholds.generated.yaml  # Per-class confidence thresholds
├── data/
│   └── raw/
│       ├── images/
│       │   ├── train/ (2000 images)
│       │   └── val/ (400 images)
│       ├── labels/ (YOLO format)
│       └── dataset.yaml
├── runs/
│   └── detect/
│       └── runs/conveyoreye/
│           ├── weights/
│           │   └── best.pt (6.3 MB)
│           └── (training logs, plots, confusion matrices)
├── calibrator.pkl                # Temperature scaling calibrator
├── setup.py
└── requirements.txt
```

---

## Commands Reference

**Train locally (GPU):**
```bash
python scripts/train.py --data data/raw/dataset.yaml --epochs 50 --device 0 --batch 32
```

**Evaluate:**
```bash
python scripts/evaluate.py --weights runs/conveyoreye/weights/best.pt --data data/raw/dataset.yaml --device cpu --calib-method temperature
```

**Serve:**
```bash
python scripts/run_api.py --model runs/conveyoreye/weights/best.pt --calibrator calibrator.pkl --device cpu
```

**Test inference:**
```bash
curl localhost:8000/health
```

---

**Session end:** 2026-06-15 (all 5 steps complete, API live)
