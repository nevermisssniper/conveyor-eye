"""Fine-tune YOLOv8 on the synthetic ConveyorEye dataset (transfer learning).

Run order: step 3.

    python scripts/train.py --data data/raw/dataset.yaml --epochs 50

WHY transfer learning + these settings?
----------------------------------------
We start from COCO-pretrained ``yolov8n.pt`` rather than training from scratch:
the synthetic set is tiny by detection standards, and the backbone's low-level
edge/texture features transfer directly to "find a thing on a background". The
settings below are chosen for a *small, imbalanced* dataset:
  * a low initial LR (lr0) so we adapt the pretrained weights instead of
    clobbering them;
  * mosaic kept modest and closed in the final epochs (``close_mosaic``) so the
    model finishes on realistic, un-mosaicked frames;
  * augmentation that overlaps our augmentation.yaml rationale (the Ultralytics
    trainer applies its own pipeline; we mirror the *intent*, not the exact ops).

We deliberately use Ultralytics' trainer (not a custom loop) here — it is the
industry-standard path and the fastest route to a working model. The custom
``YoloDetectionDataset`` exists for when the project outgrows this (v2).
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 for ConveyorEye.")
    ap.add_argument("--data", required=True, help="Path to dataset.yaml.")
    ap.add_argument("--weights", default="yolov8n.pt", help="Pretrained weights to fine-tune.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'cpu', '0', or '0,1' etc.")
    ap.add_argument("--project", default="runs", help="Output project dir.")
    ap.add_argument("--name", default="conveyoreye", help="Run name.")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Transfer-learning-friendly optimization for a small, imbalanced set.
        lr0=0.005,            # gentle: adapt, don't overwrite the COCO backbone
        lrf=0.01,             # final LR fraction (cosine decay)
        warmup_epochs=3.0,
        patience=15,          # early-stop if val mAP plateaus
        close_mosaic=10,      # disable mosaic for the last 10 epochs -> clean finish
        # Photometric aug mirrors augmentation.yaml's lighting/compression intent.
        hsv_v=0.4, hsv_s=0.5, degrees=0.0, translate=0.1, scale=0.3, fliplr=0.5,
        verbose=True,
    )
    # best.pt lands at <project>/<name>/weights/best.pt — feed that to evaluate.py.
    print(f"Done. Best weights under {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
