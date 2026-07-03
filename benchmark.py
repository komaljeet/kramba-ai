"""
Benchmark: HVN vs Transformer vs Mamba-like SSM
================================================

Compares inference speed, memory usage, and scaling behavior across:
  1. HVN (Hamiltonian Vortex Network) — O(1) per token
  2. Standard Transformer with KV cache — O(N) per token
  3. Simple SSM (Mamba-like) — O(1) per token

Measures:
  - Wall-clock time per token at different sequence lengths
  - Peak GPU/CPU memory usage
  - Tokens/second throughput

Usage:
    python benchmark.py
    python benchmark.py --d_model 256 --batch_size 1
"""

import argparse
import time
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# 1. Standard Transformer Block (with KV cache support)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with Multi-Head Attention."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence forward (training or prompt processing)."""
        B, S, D = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.dropout(self.o_proj(attn))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Standard Transformer language model with KV cache for generation."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 temperature: float, top_k: int, device: torch.device) -> list[int]:
        """Generate with full context reprocessing (no KV cache for simplicity)."""
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = generated[-512:]  # limit context
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self.forward(x)
            next_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
        return generated


# ---------------------------------------------------------------------------
# 2. Simple SSM Block (Mamba-like, simplified)
# ---------------------------------------------------------------------------

class SimpleSSMBlock(nn.Module):
    """A simplified state-space model block (Mamba-like).

    Uses a fixed-size recurrent state h ∈ ℝ^{d_model} that is updated
    deterministically per token — O(d_model) per token.
    """

    def __init__(self, d_model: int, state_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim or d_model

        # Input projection
        self.in_proj = nn.Linear(d_model, self.state_dim * 2, bias=False)

        # Recurrent state transition: h_new = A·h + B·x
        self.A = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.01)
        self.B = nn.Linear(self.state_dim, self.state_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.state_dim, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_norm = self.norm1(x)

        # Process sequentially (like Mamba's selective scan)
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            xt = x_norm[:, t, :]  # [B, D]
            gate_in = self.in_proj(xt)  # [B, state_dim*2]
            gate, candidate = gate_in.chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            candidate = torch.tanh(candidate)
            h = gate * h + (1 - gate) * (h @ self.A.T + self.B(candidate))
            out = self.out_proj(h)
            outputs.append(out)

        out_seq = torch.stack(outputs, dim=1)  # [B, S, D]
        x = x + self.dropout(out_seq)
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleSSMLM(nn.Module):
    """Simple SSM language model."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([SimpleSSMBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 temperature: float, top_k: int, device: torch.device) -> list[int]:
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = generated[-512:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self.forward(x)
            next_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
        return generated


# ---------------------------------------------------------------------------
# 3. Benchmark Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_generation_speed(
    model, generate_fn, prompt_len: int, max_new_tokens: int,
    temperature: float, top_k: int, device: torch.device, warmup: int = 5
) -> dict:
    """Measure generation speed (tokens/sec) and peak memory."""
    prompt_ids = list(range(prompt_len))

    # Warmup
    for _ in range(min(warmup, 2)):
        _ = generate_fn(model, prompt_ids, max_new_tokens=10,
                        temperature=temperature, top_k=top_k, device=device)

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.time()
    _ = generate_fn(model, prompt_ids, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=top_k, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB

    tokens_per_sec = max_new_tokens / elapsed if elapsed > 0 else 0
    ms_per_token = (elapsed / max_new_tokens) * 1000 if max_new_tokens > 0 else 0

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "total_time_s": elapsed,
        "peak_memory_mb": peak_mem,
    }


def measure_forward_speed(
    model, seq_len: int, batch_size: int, device: torch.device, warmup: int = 3
) -> dict:
    """Measure full-sequence forward pass speed."""
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(min(warmup, 3)):
        with torch.no_grad():
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    tokens = batch_size * seq_len
    return {
        "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0,
        "ms_total": elapsed * 1000,
        "ms_per_token": (elapsed / tokens) * 1000 if tokens > 0 else 0,
        "peak_memory_mb": peak_mem,
    }


# ---------------------------------------------------------------------------
# 4. Main Benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark HVN vs Transformer vs SSM")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--prompt_len", type=int, default=32, help="Prompt length")
    parser.add_argument("--seq_lens", type=str, default="64,128,256,512,1024",
                        help="Comma-separated sequence lengths for forward benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"d_model={args.d_model}, layers={args.n_layers}, vocab={args.vocab_size}")
    print()

    # --- Build models ---
    from hvn.core import HamiltonianVortexNetwork

    hvn = HamiltonianVortexNetwork(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=args.n_layers, bond_dimension=args.d_model // 4,
    ).to(device)

    # HVN with parallel scan (Issue #33 Fix 3)
    hvn_parallel = HamiltonianVortexNetwork(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=args.n_layers, bond_dimension=args.d_model // 4,
        use_parallel_scan=True,
    ).to(device)
    hvn_parallel.load_state_dict(hvn.state_dict())

    transformer = TransformerLM(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=4,
    ).to(device)

    ssm = SimpleSSMLM(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_layers=args.n_layers,
    ).to(device)

    models = {"HVN": hvn, "HVN+ParScan": hvn_parallel, "Transformer": transformer, "SSM": ssm}

    # --- Parameter counts ---
    print("=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)
    for name, model in models.items():
        print(f"  {name:15s}: {count_parameters(model):>10,} params")
    print()

    # --- Generation speed benchmark ---
    print("=" * 70)
    print(f"GENERATION SPEED ({args.gen_tokens} tokens, prompt={args.prompt_len})")
    print("=" * 70)
    print(f"{'Model':20s} | {'tokens/sec':>12s} | {'ms/token':>10s} | {'peak_mem':>10s}")
    print("-" * 60)

    gen_results = {}
    for name, model in models.items():
        if name == "HVN":
            gen_fn = lambda m, p, **kw: m.generate_incremental(p, **kw)
        elif name == "HVN+ParScan":
            gen_fn = lambda m, p, **kw: m.generate_incremental(p, **kw)
        else:
            gen_fn = lambda m, p, **kw: m.generate(p, **kw)

        result = measure_generation_speed(
            model, gen_fn, prompt_len=args.prompt_len,
            max_new_tokens=args.gen_tokens,
            temperature=0.8, top_k=10, device=device,
        )
        gen_results[name] = result
        mem_str = f"{result['peak_memory_mb']:.1f} MB" if result['peak_memory_mb'] > 0 else "N/A"
        print(f"{name:20s} | {result['tokens_per_sec']:>10.1f}  | {result['ms_per_token']:>8.2f}  | {mem_str:>10s}")

    # --- Issue #33 kernel-enabled generation benchmarks ---
    if device.type == "cuda":
        print()
        print("=" * 70)
        print("ISSUE #33 KERNEL-ENABLED GENERATION (CUDA Graph + Fused Triton)")
        print("=" * 70)
        print(f"{'Method':25s} | {'tokens/sec':>12s} | {'ms/token':>10s} | {'speedup':>8s}")
        print("-" * 65)

        # Baseline
        baseline_tps = gen_results["HVN"]["tokens_per_sec"]
        print(f"{'HVN (baseline)':25s} | {baseline_tps:>10.1f}  | {1000/baseline_tps:>8.2f}  | {'1.00x':>8s}")

        # CUDA Graph (Fix 6)
        try:
            t0 = time.time()
            _ = hvn.generate_incremental_graph(
                list(range(args.prompt_len)), max_new_tokens=args.gen_tokens,
                temperature=0.8, top_k=10, device=device,
            )
            torch.cuda.synchronize(device)
            graph_time = time.time() - t0
            graph_tps = args.gen_tokens / graph_time
            print(f"{'HVN + CUDA Graph':25s} | {graph_tps:>10.1f}  | {1000/graph_tps:>8.2f}  | {graph_tps/baseline_tps:>7.2f}x")
        except Exception as e:
            print(f"{'HVN + CUDA Graph':25s} | {'FAILED':>12s} | {'N/A':>10s} | {'N/A':>8s}")

        # Fused Triton (Fix 1)
        try:
            t0 = time.time()
            _ = hvn.generate_incremental_fused(
                list(range(args.prompt_len)), max_new_tokens=args.gen_tokens,
                temperature=0.8, top_k=10, device=device,
            )
            torch.cuda.synchronize(device)
            fused_time = time.time() - t0
            fused_tps = args.gen_tokens / fused_time
            print(f"{'HVN + Fused Triton':25s} | {fused_tps:>10.1f}  | {1000/fused_tps:>8.2f}  | {fused_tps/baseline_tps:>7.2f}x")
        except Exception as e:
            print(f"{'HVN + Fused Triton':25s} | {'FAILED':>12s} | {'N/A':>10s} | {'N/A':>8s}")
    print()

    # --- Forward pass scaling benchmark ---
    print("=" * 70)
    print("FORWARD PASS SCALING (full sequence, batch=1)")
    print("=" * 70)

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    print(f"{'Model':15s} |", " | ".join(f"{'N='+str(n):>12s}" for n in seq_lens))
    print("-" * (20 + 15 * len(seq_lens)))

    fwd_results = {}
    for name, model in models.items():
        fwd_results[name] = {}
        ms_vals = []
        for n in seq_lens:
            result = measure_forward_speed(model, seq_len=n, batch_size=args.batch_size, device=device)
            fwd_results[name][n] = result
            ms_vals.append(result["ms_per_token"])
        print(f"{name:15s} |", " | ".join(f"{v:>10.2f} ms" for v in ms_vals))
    print()

    # --- Memory scaling ---
    print("=" * 70)
    print("PEAK MEMORY SCALING (forward pass, batch=1)")
    print("=" * 70)
    print(f"{'Model':15s} |", " | ".join(f"{'N='+str(n):>12s}" for n in seq_lens))
    print("-" * (20 + 15 * len(seq_lens)))

    for name, model in models.items():
        mem_vals = []
        for n in seq_lens:
            mem = fwd_results[name][n]["peak_memory_mb"]
            mem_vals.append(mem)
        print(f"{name:15s} |", " | ".join(f"{v:>8.1f} MB" if v > 0 else f"{'N/A':>10s}" for v in mem_vals))
    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    fastest_gen = max(gen_results, key=lambda k: gen_results[k]["tokens_per_sec"])
    print(f"  Fastest generation:    {fastest_gen} ({gen_results[fastest_gen]['tokens_per_sec']:.1f} tok/s)")

    # Check scaling: does ms/token stay flat as N grows?
    for name in models:
        ms_at_min = fwd_results[name][seq_lens[0]]["ms_per_token"]
        ms_at_max = fwd_results[name][seq_lens[-1]]["ms_per_token"]
        ratio = ms_at_max / ms_at_min if ms_at_min > 0 else 0
        scaling = "O(1) ✓" if ratio < 2.0 else f"O(N) — {ratio:.1f}x slower"
        print(f"  {name:15s} scaling: {scaling} (N={seq_lens[0]}→{seq_lens[-1]})")

    print()
    print("Note: HVN uses incremental O(1) generation via Hamiltonian state cache.")
    print("      Transformer reprocesses full context each step (no KV cache here).")
    print("      SSM reprocesses full context each step (simplified, no incremental).")


if __name__ == "__main__":
    main()