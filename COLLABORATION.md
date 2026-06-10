# Working Agreement — ConveyorEye

How Claude and Rayden work on this project. Trimmed to what actually changes behavior.

## Memory (external)
- **PROJECT_LOG.md is the source of truth.** Every meaningful change → a Daily Log bullet (`MM.DD`, Notion-mirror) + a Decision Log entry when a *choice* was made.
- Log the **reasoning and who proposed it**, and mark the **outcome** (✅ worked / 🔲 open / ↩︎ reverted). Outcomes matter more than ideas.
- **SESSION_CONTEXT.md** is the paste-at-start snapshot (architecture+dims, hyperparams, latest metrics, open questions). Keep it current; it has no history.

## Session start / hygiene
- Begin a chat by pasting SESSION_CONTEXT.md.
- Reset the chat ~every 50 turns. Carry state forward via the docs, not chat scrollback.
- **Paste the relevant code**, don't reference "what we did last time."
- Restate the non-obvious constraints periodically (imbalanced priors, asymmetric thresholds, seed-split discipline).

## Verification habits (do these before scaling)
- **Overfit a tiny set first.** Train on ~20 frames for a few epochs; expect loss→~0 and mAP→~1. If it *can't* overfit 20 frames, the pipeline is broken — check label format, letterbox box-reversal, and class-id mapping **before** burning Colab GPU hours.
- **Measure compute/memory empirically** — Claude runs it and reports, doesn't ask Rayden to guess.
- **Validate against reference:** our `DetectionEvaluator` mAP vs Ultralytics' own `val` mAP; calibration ECE behavior vs textbook temperature scaling; PSI severity vs the 0.1/0.2 convention.
- Guard the splits: train seed ≠ val seed (we use `0` / `0+10000`). Watch for leakage if data gen changes.

## Lean on Claude for
- Concept drilling & math: calibration (temp/Platt/isotonic) derivations, k-center geometry, PSI/KL/JS, AP integral.
- Conceptual bugs: data leakage, wrong splits, box-format / letterbox-reversal errors, class-id mismatches.
- Code review, scoping, docs, writing, log upkeep.

## Don't lean on Claude for
- Final production weights/configs shipped without Rayden's review.
- Authoritative claims on recent research — **search first**, cite, flag uncertainty.
- Replacing hands-on implementation of the core TODOs (mAP@50:95, FAISS coreset) — those are Rayden's to write; Claude scaffolds and reviews.
