# Quick Start & Setup

> **Moved from README:** §5 Quick Start & Usage — the install / demo / train / generate
> commands and library snippet. Runtime feature walkthroughs are in [usage.md](usage.md);
> training flag reference and VRAM envelopes in [training.md](training.md).

## 5. Quick Start & Usage

```bash
# Install dependencies (CUDA GPU version)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy

# Run the architecture demo (verifies shapes & forward pass)
python demos/demo.py

# Train the model on the bundled sample text
python train.py --epochs 50 --generate --prompt "The "

# Generate text from a trained checkpoint
python generate.py --prompt "The " --tokens 200 --temp 0.7
```

```python
import torch
from hvn.core import HamiltonianVortexNetwork

model = HamiltonianVortexNetwork(
    vocab_size=1000, d_model=128, num_layers=2,
    bond_dimension=32,   # SVD truncation rank (quantum renormalization)
)
input_ids = torch.randint(0, 1000, (4, 64))   # [batch, seq_len]
logits = model(input_ids)                       # [batch, seq_len, vocab_size]
```

### Long-running benchmarks & monitors — always run in the background

Speed/throughput benchmarks and log monitors (e.g. the `scripts/measure_*.py` probes for
issue #95) take minutes — mostly `torch.compile` warmup. **Always launch them as background
tasks and tail their logs with a monitor** rather than blocking the session:

- run the script with a background shell (`run_in_background: true`) writing to a log file
  (`... 2>&1 | tee scripts/<name>.log`),
- tail that log with a monitor filtering for the result lines + error signatures
  (`Traceback|Error|RuntimeError|...`),
- keep working; report results when the run completes, and surface errors immediately.

This keeps the session responsive while long GPU jobs run, and the filtered monitor
notifies the moment a result or failure lands — no polling.

### Real training runs: always pass `--fast`

`train_edge_conversational.py --fast` turns on the recommended stack in one switch (bf16 +
TF32 + `torch.compile` + Triton scan + 4 DataLoader workers). Every real run in this project
was launched with it — the pretrain and hybrid-FT logs all open with the `[--fast]` banner —
and without it training is roughly 5x slower (0.2 vs ~0.8 steps/s measured at d=512/12L).

The bare defaults (fp32, no compile, `num_workers=0`) are deliberate: they exist so a
controlled fp32 A/B has bit-comparable arms. Use them for parity experiments, `--fast` for
everything else. Details and caveats: [`training.md`](training.md) section "`--fast`".

### GPU / CUDA requirement

All training, benchmarking, and evaluation **must run on GPU, never CPU**. Environment:
NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), driver 596.36, PyTorch with CUDA
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`). Verify
`torch.cuda.is_available()` before running; if False, fix the install rather than falling
back to CPU. (Windows distributed/GPU quirks in [ai_rules.md](ai_rules.md).)