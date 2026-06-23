"""Fine-tune the trained ConveyorEye model on mixed synthetic + real crack data.

Run order: after initial training (train.py) and MVTec eval showing crack transfer gap.

    # On Colab (GPU):
    python scripts/finetune.py \
        --weights runs/conveyoreye/weights/best.pt \
        --data data/finetune.yaml

WHY these settings differ from train.py
----------------------------------------
We already have a strong synthetic model (mAP@50 0.989). The goal is NOT to
retrain — it is to patch crack recognition on real textures without clobbering
the other four classes.

Three levers for that:

  freeze=10   Lock the first 10 backbone layers (shallow edge/texture detectors).
              These features transfer fine from COCO + synthetic. Only the deeper
              semantic layers and detection head update. Without this, 17 real
              images can overwrite what 2000 synthetic images taught.

  lr0=0.0005  10× lower than initial training. Small steps so real-crack signal
              nudges the weights rather than dominating them.

  epochs=20   Short by design. With freeze + low LR, the model converges fast on
              17 images. More epochs → overfit the real cracks, forget the rest.

  patience=5  Tight early-stop. If val mAP (synthetic) starts dropping, bail.

Outputs: runs/conveyoreye_ft/weights/best.pt
Feed that to evaluate.py (regression check on synthetic val) then eval_mvtec.py
(real-world crack detection check).
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune ConveyorEye on mixed real+synthetic data.")
    ap.add_argument("--weights", default="runs/conveyoreye/weights/best.pt",
                    help="Starting weights — the already-trained synthetic model.")
    ap.add_argument("--data", default="data/finetune.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="'cpu' (local/Mac), '0' (Colab GPU), etc.")
    ap.add_argument("--freeze", type=int, default=10,
                    help="Number of backbone layers to freeze (default 10).")
    ap.add_argument("--lr0", type=float, default=0.0005,
                    help="Initial LR. Only takes effect when --optimizer is not 'auto'.")
    ap.add_argument("--optimizer", default="auto",
                    help="Optimizer: 'SGD', 'Adam', 'AdamW', or 'auto'. "
                         "'auto' ignores --lr0 (Ultralytics picks its own LR).")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early-stop patience on val mAP (default 5).")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="conveyoreye_ft")
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

        # Fine-tuning levers (see module docstring for rationale)
        freeze=args.freeze,
        lr0=args.lr0,
        optimizer=args.optimizer,
        lrf=0.01,
        warmup_epochs=1.0,      # shorter warmup — we're not learning from scratch
        patience=args.patience,
        close_mosaic=5,

        # Same photometric aug as train.py — keeps crack appearance varied
        hsv_v=0.4, hsv_s=0.5, degrees=0.0, translate=0.1, scale=0.3, fliplr=0.5,
        verbose=True,
    )

    print(f"\nDone. Fine-tuned weights: {args.project}/{args.name}/weights/best.pt")
    print("Next steps:")
    print("  1. python scripts/evaluate.py --weights runs/conveyoreye_ft/weights/best.pt "
          "--data data/raw/dataset.yaml --device cpu")
    print("  2. python scripts/eval_mvtec.py --weights runs/conveyoreye_ft/weights/best.pt")


if __name__ == "__main__":
    main()
