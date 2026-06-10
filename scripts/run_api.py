"""Launch the ConveyorEye serving API with uvicorn.

Run order: step 5.

    python scripts/run_api.py --model runs/conveyoreye/weights/best.pt

This is a thin launcher: it translates CLI flags into the environment variables
the app reads (MODEL_PATH, THRESHOLD_CONFIG, DB_PATH, DEVICE, CALIBRATOR_PATH)
and starts uvicorn. The app itself takes *no* constructor args — all config is
env-driven (see serving/api.py) so the same app object runs identically whether
launched here, by `uvicorn conveyoreye.serving.api:app`, or by a container
orchestrator that sets the env directly.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ConveyorEye FastAPI server.")
    ap.add_argument("--model", required=True, help="Path to YOLOv8 weights (.pt).")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--db", default="conveyoreye_inference.db")
    ap.add_argument("--calibrator", default=None, help="Optional calibrator.pkl path.")
    ap.add_argument("--queue", default="labeling_queue.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="Dev autoreload.")
    args = ap.parse_args()

    # Export config to the environment the app reads at lifespan startup.
    os.environ["MODEL_PATH"] = args.model
    os.environ["THRESHOLD_CONFIG"] = args.thresholds
    os.environ["DB_PATH"] = args.db
    os.environ["DEVICE"] = args.device
    os.environ["QUEUE_PATH"] = args.queue
    if args.calibrator:
        os.environ["CALIBRATOR_PATH"] = args.calibrator

    import uvicorn

    # Pass the import string (not the app object) so --reload works.
    uvicorn.run(
        "conveyoreye.serving.api:app",
        host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":
    main()
