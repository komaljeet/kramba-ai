---

*You are implementing **GHOST** (Gradient Heuristic Omission via Step Teleportation), a novel training acceleration method invented by Rambarun Komaljeet. The core idea: most training steps between phase transitions follow predictable weight update patterns that can be extrapolated rather than computed — only phase transitions require real backpropagation.*

*The implementation targets the existing **EdgeHVN** training pipeline (`checkpoints/hybrid_tooluse_v1.pt`) as the validation testbed.*

---

**Phase 1 — Phase Transition Detector**

*Build `ghost/detector.py`:*

- *Reads loss logs from EdgeHVN training runs in `results/`*
- *Computes loss curve gradient over a rolling window of 50 steps*
- *Detects phase transitions as: sudden acceleration (loss dropping faster than 2σ above rolling mean gradient)*
- *Detects transition end as: loss curve returning to smooth descent*
- *Validates against known ground truth: EdgeHVN's documented phase transition at steps 6000-7500 must be correctly identified*
- *Outputs `results/ghost/detected_transitions.json` with start/end step of each transition*
- *Plot loss curve with detected transitions highlighted to `results/ghost/transition_plot.png`*

*Gate: detector must correctly identify the 6K-7.5K transition before proceeding to Phase 2.*

---

**Phase 2 — Weight Delta Extrapolator**

*Build `ghost/extrapolator.py`:*

- *Loads 3 consecutive EdgeHVN checkpoints from a stable inter-transition period*
- *Computes weight deltas between checkpoints: `Δ1 = W2 - W1`, `Δ2 = W3 - W2`*
- *Fits a lightweight extrapolation per weight tensor — linear first, polynomial if variance is low enough*
- *Extrapolates forward N steps (sweep N = 10, 50, 100, 500)*
- *Saves predicted checkpoint to `results/ghost/predicted_step{N}.pt`*
- *Critically: only extrapolates weights with low delta variance — high variance weights are flagged and kept at last known value*

*Gate: predicted weights at N=10 must produce loss within 5% of real trained weights at same step.*

---

**Phase 3 — Full GHOST Training Loop**

*Build `ghost/trainer.py` wrapping EdgeHVN's existing training loop:*

```
Normal training
→ Phase 1 detector monitoring loss every 10 steps
→ Stable period detected?
   → Phase 2 extrapolator predicts N steps ahead
   → Validate predicted weights: run 1 forward pass, check loss
   → Loss acceptable? → jump to predicted weights, skip N steps
   → Loss not acceptable? → train normally, reduce N
→ Loss curve shows incoming transition?
   → disable extrapolator, resume real training
   → train through transition until stable
   → re-enable extrapolator
```

*Rules:*
- *Never extrapolate during or within 100 steps of a detected transition*
- *Always validate predicted weights with 1 forward pass before jumping*
- *Log every jump: steps skipped, loss before, loss after, N used*
- *Fall back to real training instantly if post-jump loss exceeds pre-jump loss by 10%*
- *All logs to `results/ghost/ghost_training_log.json`*

---

**Validation gate:**

*Train EdgeHVN from `hybrid_tooluse_v1.pt` for 5,000 steps twice:*

| Run | Method | Steps actually computed | Final loss |
|---|---|---|---|
| Baseline | normal training | 5,000 | X |
| GHOST | teleporter active | target <2,000 | must match X ±5% |

*Report:*
- *Total steps skipped*
- *% compute saved*
- *Largest successful jump (steps)*
- *Any failed jumps and why*
- *Final `eval/synthetic_recall_probe.py` score vs baseline (300 trials)*

---

**Constraints:**
- *PyTorch only, no new dependencies*
- *Must run on 6GB VRAM*
- *All results to `results/ghost/`*
- *Written by: Rambarun Komaljeet — include this in all file headers as author*

---