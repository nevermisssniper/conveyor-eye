# ConveyorEye — runnable workbench

Explore the cost of missed defects and false alarms on a seeded inspection stream.

## Open the interface

Python 3.10+; no npm install, API keys, model download or GPU required for the demo.

```sh
python serve_demo.py
```

Open <http://127.0.0.1:8000>. To use another port: `python serve_demo.py --port 8001`.
Serve `demo/` over HTTP rather than opening `index.html` as a file. The server binds only to localhost.

## What is implemented

The inspection stream is seeded synthetic data. Threshold counts and PSI use real calculations. This is not detector performance. The original calibration and training pipeline remains available. Core learning exercises (mAP@50:95 and FAISS coreset) remain explicit TODOs for Ray. The package now includes `scripts` so installed CLI entry points resolve.

Interface capabilities: Threshold tuning, Inspection queue, Decision export.

## Core workflow

From the repository root (install the project's existing dependencies first for model-backed commands):

```sh
python -m scripts.demo --threshold 0.6 --output demo/conveyor.json
```

## Verify

The browser's calculation tests use Node 18+ and no third-party packages:

```sh
node --test demo/core.test.cjs
python -m pytest tests -q
```

The GitHub demo workflow checks browser calculations and the dependency-free core tests where present. Full original model/service suites have separate dependencies; passing demo checks does not certify GPU training or external services.

## Before live use

Weights, real camera frames and production threshold validation are still required.

Local browser data belongs to this browser profile. Export records before clearing it or changing devices. Photo previews stay in memory; exported briefs contain filenames, not uploaded image bytes. There are no payment or outbound model calls in this demo.

## Provenance

Workbench and starter additions: AI-assisted implementation, 2026-11-13, for Ray's review. Existing repository history and upstream attribution are preserved. New starter directions are proposals where the GitHub repository had no code.
