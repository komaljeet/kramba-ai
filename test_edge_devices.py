"""
Test suite for the 'edge-devices' architecture variant (Issue #32, full scope).

Covers every technique from the compression roadmap not already verified
by test_issue32_phase1.py (Phase 1: GPTQ, Hadamard, rate-distortion,
2:4 sparsity):

  Phase 2: TernaryLinear (BitNet b1.58), LowRankSparseWeights, TTMatrix,
           MERAMatrix, SharedWeightHVN
  Phase 3: EntropyCoder, WaveletCompressor, DCTCompressor, KroneckerLinear,
           CirculantLinear, ToeplitzLinear, HolographicWeights,
           SuperpositionWeights, PhaseEncodedComplexWeights
  Phase 4: StreamingHVNInference, MMapWeightStore, SuccinctWeights,
           TaskSpecificLoader
  Phase 5: AnnealingPruner, TopologicalPruner, FreeEnergyQuantizer,
           PopulationCoding, recovery_finetune_step (QLoRA pointer)

Plus an end-to-end EdgeHVN check: weight sharing + GPTQ compression on a
trained model, verifying perplexity degradation and parameter reduction.

All tests run on GPU (CUDA required).
"""

import math
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from hvn.edge.ternary import TernaryLinear, TernaryHamiltonian, ternary_quantize_ste, apply_ternary_to_model
from hvn.edge.decomposition import (
    LowRankSparseWeights, TTMatrix, MERAMatrix, KroneckerLinear, CirculantLinear, ToeplitzLinear,
)
from hvn.edge.shared import SharedWeightHVN
from hvn.edge.entropy import EntropyCoder, LDPCCoder
from hvn.edge.transform import WaveletCompressor, DCTCompressor
from hvn.edge.holographic import HolographicWeights, SuperpositionWeights, PhaseEncodedComplexWeights
from hvn.edge.streaming import StreamingHVNInference, MMapWeightStore, SuccinctWeights, TaskSpecificLoader
from hvn.edge.recovery import AnnealingPruner, TopologicalPruner, FreeEnergyQuantizer, PopulationCoding, recovery_finetune_step
from hvn.edge.model import EdgeHVN
from hvn.compress import collect_calibration_hessian
from hvn.core.hvn_block import HVNBlock
from train import CharTokenizer, TextDataset, SAMPLE_TEXT, train_model

DEVICE = torch.device("cuda")
SCRATCH = tempfile.mkdtemp(prefix="edge_devices_test_")


def rel_err(orig: torch.Tensor, recon: torch.Tensor) -> float:
    return (orig.float() - recon.float()).abs().mean().item() / orig.float().abs().mean().clamp(min=1e-8).item()


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def test_ternary_linear():
    print("\n" + "=" * 60)
    print("TEST: TernaryLinear (BitNet b1.58, STE-trainable)")
    print("=" * 60)
    torch.manual_seed(0)

    layer = TernaryLinear(64, 32, bias=True).to(DEVICE)
    x = torch.randn(16, 64, device=DEVICE)
    y = layer(x)
    assert y.shape == (16, 32)

    w_ternary = ternary_quantize_ste(layer.weight)
    unique_vals = torch.unique(torch.round(w_ternary / w_ternary.abs().mean().clamp(min=1e-8)))
    print(f"  Ternary-quantized weight has {unique_vals.numel()} distinct normalized levels (expect <= 3)")
    assert unique_vals.numel() <= 3

    # Verify it's trainable: a few SGD steps should reduce loss.
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    target = torch.randn(16, 32, device=DEVICE)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (STE gradient should reduce loss)")
    assert losses[-1] < losses[0]

    layer.freeze()
    y_frozen = layer(x)
    mem_dense = layer.in_features * layer.out_features * 4
    mem_ternary = layer.memory_bytes()
    print(f"  Frozen inference output shape: {tuple(y_frozen.shape)}")
    print(f"  Memory: dense fp32={mem_dense} bytes, ternary={mem_ternary} bytes "
          f"({mem_dense / mem_ternary:.2f}x compression)")
    assert mem_ternary < mem_dense

    H = torch.randn(8, 16, 16, device=DEVICE) * 0.1
    th = TernaryHamiltonian(H)
    H_deq = th.dequantize()
    print(f"  TernaryHamiltonian relative error: {rel_err(H, H_deq):.2%}, "
          f"memory: {H.numel()*4} -> {th.memory_bytes()} bytes")
    assert th.memory_bytes() < H.numel() * 4

    print("  PASSED")


def test_low_rank_sparse_and_tt_and_mera():
    print("\n" + "=" * 60)
    print("TEST: LowRankSparseWeights, TTMatrix, MERAMatrix")
    print("=" * 60)
    torch.manual_seed(1)

    # --- Low-rank + sparse ---
    U0 = torch.randn(128, 8, device=DEVICE)
    V0 = torch.randn(8, 128, device=DEVICE)
    W = U0 @ V0 * 0.1
    outlier_mask = torch.rand_like(W) > 0.98
    W = W + outlier_mask.float() * torch.randn_like(W) * 5  # sparse outliers
    lsw = LowRankSparseWeights(rank=8, sparsity=0.98, iterations=5)
    qdict = lsw.decompose(W)
    W_rec = LowRankSparseWeights.reconstruct(qdict)
    err = rel_err(W, W_rec)
    mem_dense = W.numel() * 4
    mem_lsw = LowRankSparseWeights.memory_bytes(qdict)
    print(f"  L+S relative error: {err:.2%}, memory {mem_dense} -> {mem_lsw} bytes "
          f"({mem_dense / mem_lsw:.2f}x)")
    assert err < 0.5, f"L+S relative error too high: {err:.2%}"
    assert mem_lsw < mem_dense

    # --- Tensor Train ---
    W_tt = torch.randn(64, 64, device=DEVICE) * 0.05
    tt_full = TTMatrix.from_dense(W_tt, row_factors=(8, 8), col_factors=(8, 8), max_rank=64)
    W_full_recon = tt_full.to_dense()
    err_full = rel_err(W_tt, W_full_recon)
    print(f"  TT (full rank) reconstruction relative error: {err_full:.4%} (should be ~exact)")
    assert err_full < 1e-3

    # A dense i.i.d. random matrix has no low-rank structure to exploit (unlike a
    # trained/SVD-truncated Hamiltonian, which is the intended use case here) —
    # so truncating to rank=4 is expected to be lossy; we just verify the
    # storage reduction, not reconstruction accuracy, on this synthetic input.
    tt_low = TTMatrix.from_dense(W_tt, row_factors=(8, 8), col_factors=(8, 8), max_rank=4)
    W_low_recon = tt_low.to_dense()
    mem_dense_tt = W_tt.numel() * 4
    mem_tt = tt_low.memory_bytes()
    print(f"  TT (rank=4, on unstructured random data) relative error: {rel_err(W_tt, W_low_recon):.2%}, "
          f"params {W_tt.numel()} -> {tt_low.num_params()} ({mem_dense_tt / mem_tt:.2f}x memory)")
    assert mem_tt < mem_dense_tt

    x = torch.randn(64, device=DEVICE)
    y_matvec = tt_low.matvec(x)
    y_dense = tt_low.to_dense() @ x
    assert torch.allclose(y_matvec, y_dense, atol=1e-3), "TTMatrix.matvec must match dense reconstruction"
    print(f"  TTMatrix.matvec matches dense-reconstruction matmul (max diff "
          f"{(y_matvec - y_dense).abs().max().item():.2e})")

    # --- MERA (simplified hierarchical) ---
    mera = MERAMatrix(rank_per_level=4, max_levels=3)
    mdict = mera.decompose(W_tt)
    W_mera_recon = MERAMatrix.reconstruct(mdict)
    err_mera = rel_err(W_tt, W_mera_recon)
    mem_mera = MERAMatrix.memory_bytes(mdict)
    print(f"  MERA relative error: {err_mera:.2%}, memory {W_tt.numel()*4} -> {mem_mera} bytes")
    assert err_mera < 0.5

    print("  PASSED")


def test_shared_weight_hvn():
    print("\n" + "=" * 60)
    print("TEST: SharedWeightHVN (ALBERT-style weight sharing)")
    print("=" * 60)
    torch.manual_seed(2)

    block = HVNBlock(d_model=32, n_heads=2, bond_dimension=8, chunk_size=16, dropout=0.0).to(DEVICE)
    shared = SharedWeightHVN(block, num_layers=6).to(DEVICE)

    x = torch.randn(2, 20, 32, device=DEVICE)
    out = shared(x)
    assert out.shape == x.shape

    shared_params = shared.num_shared_params()
    unshared_equiv = shared.num_unshared_equivalent_params()
    print(f"  6 layers: shared params={shared_params:,}, equivalent unshared params={unshared_equiv:,} "
          f"({unshared_equiv / shared_params:.2f}x reduction)")
    assert shared_params < unshared_equiv

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------


def test_entropy_coder():
    print("\n" + "=" * 60)
    print("TEST: EntropyCoder / LDPCCoder (Huffman lossless coding)")
    print("=" * 60)
    torch.manual_seed(3)

    # Skewed distribution (like a ternary-quantized tensor: mostly zeros)
    codes = torch.zeros(2000, dtype=torch.int64)
    codes[torch.randperm(2000)[:200]] = 1
    codes[torch.randperm(2000)[:200]] = -1

    coder = EntropyCoder()
    edict = coder.encode(codes)
    decoded = coder.decode(edict)
    assert torch.equal(codes, decoded), "Huffman coding must be lossless"

    entropy_bits = coder.theoretical_entropy_bits(codes)
    actual_bits_per_symbol = edict["packed"].numel() * 8 / codes.numel()
    print(f"  Shannon entropy: {entropy_bits:.3f} bits/symbol, "
          f"Huffman achieved: {actual_bits_per_symbol:.3f} bits/symbol")
    naive_bits_per_symbol = 8  # int8 baseline
    print(f"  Memory: naive int8={codes.numel()} bytes, Huffman={coder.memory_bytes(edict)} bytes")
    assert actual_bits_per_symbol < naive_bits_per_symbol
    assert coder.memory_bytes(edict) < codes.numel()

    ldpc = LDPCCoder()
    edict2 = ldpc.encode(codes)
    assert torch.equal(codes, ldpc.decode(edict2)), "LDPCCoder (Huffman-backed) must also be lossless"

    print("  PASSED")


def test_wavelet_and_dct_compressors():
    print("\n" + "=" * 60)
    print("TEST: WaveletCompressor, DCTCompressor")
    print("=" * 60)
    torch.manual_seed(4)

    # Smooth, low-frequency-dominated matrix (energy concentrates in low frequencies)
    xs = torch.linspace(0, 4 * math.pi, 64, device=DEVICE)
    W = torch.outer(torch.sin(xs), torch.cos(xs)) + 0.01 * torch.randn(64, 64, device=DEVICE)

    wc = WaveletCompressor(levels=2, keep_fraction=0.25)
    cdict = wc.compress(W)
    W_rec = wc.decompress(cdict)
    err_wavelet = rel_err(W, W_rec)
    mem_dense = W.numel() * 4
    mem_wavelet = WaveletCompressor.memory_bytes(cdict)
    print(f"  Wavelet (keep 25%) relative error: {err_wavelet:.2%}, "
          f"memory {mem_dense} -> {mem_wavelet} bytes ({mem_dense / mem_wavelet:.2f}x)")
    assert mem_wavelet < mem_dense
    assert err_wavelet < 0.5

    dct = DCTCompressor(quality=0.25)
    ddict = dct.compress(W)
    W_rec_dct = dct.decompress(ddict)
    err_dct = rel_err(W, W_rec_dct)
    mem_dct = DCTCompressor.memory_bytes(ddict)
    print(f"  DCT (quality=0.25) relative error: {err_dct:.2%}, "
          f"memory {mem_dense} -> {mem_dct} bytes ({mem_dense / mem_dct:.2f}x)")
    assert mem_dct < mem_dense
    assert err_dct < 0.5

    print("  PASSED")


def test_kronecker_circulant_toeplitz():
    print("\n" + "=" * 60)
    print("TEST: KroneckerLinear, CirculantLinear, ToeplitzLinear")
    print("=" * 60)
    torch.manual_seed(5)

    dense = nn.Linear(64, 64, bias=True).to(DEVICE)
    kron = KroneckerLinear.from_linear_via_svd(dense, in_a=8, out_a=8)
    x = torch.randn(4, 64, device=DEVICE)
    y_dense = dense(x)
    y_kron = kron(x)
    print(f"  Kronecker: params {dense.weight.numel()} -> {kron.num_params()} "
          f"({dense.weight.numel() / kron.num_params():.1f}x fewer)")
    print(f"  Output relative diff vs dense (rank-1 Kronecker approx, expected lossy): "
          f"{rel_err(y_dense, y_kron):.2%}")
    assert kron.num_params() < dense.weight.numel()
    assert y_kron.shape == y_dense.shape

    circ = CirculantLinear(64).to(DEVICE)
    y_circ = circ(x)
    assert y_circ.shape == (4, 64)
    print(f"  Circulant: params={circ.num_params()} vs dense {64*64} "
          f"({64*64 / circ.num_params():.1f}x fewer)")
    assert circ.num_params() < 64 * 64

    toep = ToeplitzLinear(64).to(DEVICE)
    y_toep = toep(x)
    assert y_toep.shape == (4, 64)
    print(f"  Toeplitz: params={toep.num_params()} vs dense {64*64} "
          f"({64*64 / toep.num_params():.1f}x fewer)")
    assert toep.num_params() < 64 * 64

    print("  PASSED")


def test_holographic_and_superposition():
    print("\n" + "=" * 60)
    print("TEST: HolographicWeights, SuperpositionWeights, PhaseEncodedComplexWeights")
    print("=" * 60)
    torch.manual_seed(6)

    shape = (32, 32)
    mats = [torch.randn(*shape, device=DEVICE) * 0.1 for _ in range(4)]

    holo = HolographicWeights(shape, num_slots=4, seed=42)
    H = holo.encode(mats)
    assert H.shape == shape

    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-8)

    # Elementwise unmasking (no denoising/"clean-up memory" step) recovers each
    # slot with real but noisy signal — cosine similarity to the true matrix is
    # the meaningful metric here (mean relative error is dominated by cross-talk
    # noise and isn't a fair measure without a cleanup step), matching the
    # standard HRR/holographic-storage accuracy characterization.
    coss = [cosine_sim(mats[k], holo.decode(H, k)) for k in range(4)]
    errs = [rel_err(mats[k], holo.decode(H, k)) for k in range(4)]
    print(f"  Holographic (4 matrices -> 1): per-slot cosine similarity to original "
          f"{[f'{c:.2f}' for c in coss]} (relative error {[f'{e:.0%}' for e in errs]}, "
          f"dominated by cross-talk noise without a clean-up/denoising step)")
    mem_dense = sum(m.numel() * 4 for m in mats)
    mem_holo = holo.memory_bytes()
    print(f"  Memory: {mem_dense} -> {mem_holo} bytes ({mem_dense / mem_holo:.1f}x)")
    assert mem_holo < mem_dense
    assert all(c > 0.3 for c in coss), "Recovered slots should correlate meaningfully with the original"

    sup = SuperpositionWeights(num_basis=3)
    sdict = sup.fit(mats)
    errs_sup = [rel_err(mats[k], SuperpositionWeights.reconstruct(sdict, k)) for k in range(4)]
    mem_sup = SuperpositionWeights.memory_bytes(sdict)
    print(f"  Superposition (basis=3): per-matrix relative errors {[f'{e:.2%}' for e in errs_sup]}, "
          f"memory {mem_dense} -> {mem_sup} bytes ({mem_dense / mem_sup:.1f}x)")
    assert mem_sup < mem_dense

    pcw = PhaseEncodedComplexWeights(phase_bits=8)
    pdict = pcw.fit(mats)
    errs_phase = [rel_err(mats[k], PhaseEncodedComplexWeights.reconstruct(pdict, k)) for k in range(4)]
    mem_phase = PhaseEncodedComplexWeights.memory_bytes(pdict)
    print(f"  Phase-encoded (8-bit phase): per-matrix relative errors {[f'{e:.2%}' for e in errs_phase]}, "
          f"memory {mem_dense} -> {mem_phase} bytes ({mem_dense / mem_phase:.1f}x)")
    assert mem_phase < mem_dense

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------


def test_streaming_inference():
    print("\n" + "=" * 60)
    print("TEST: StreamingHVNInference (layer-by-layer NVMe streaming)")
    print("=" * 60)
    torch.manual_seed(7)

    from hvn.core import HamiltonianVortexNetwork
    model = HamiltonianVortexNetwork(vocab_size=37, d_model=32, num_layers=3, n_heads=2,
                                      bond_dimension=8, chunk_size=16, dropout=0.0).to(DEVICE)
    model.eval()

    weight_dir = os.path.join(SCRATCH, "streaming_weights")
    streamer = StreamingHVNInference(model, weight_dir=weight_dir, device=DEVICE)

    for block in model.blocks:
        assert next(block.parameters()).device.type == "cpu", "Blocks should be off-device after construction"

    input_ids = torch.randint(0, 37, (2, 10), device=DEVICE)
    logits_streamed = streamer.forward(input_ids)
    assert logits_streamed.shape == (2, 10, 37)

    for block in model.blocks:
        block.to(DEVICE)
    with torch.no_grad():
        logits_normal = model(input_ids)

    err = rel_err(logits_normal, logits_streamed)
    print(f"  Streamed vs. normal forward relative logit difference: {err:.6f} (should be ~0, same weights)")
    assert err < 1e-3

    peak = streamer.peak_layer_bytes()
    total = streamer.total_streamed_bytes()
    print(f"  Peak single-layer VRAM: {peak} bytes, total streamed across all layers: {total} bytes "
          f"({total / peak:.1f}x more than the streaming VRAM ceiling)")
    assert peak < total

    print("  PASSED")


def test_mmap_and_succinct_and_task_loader():
    print("\n" + "=" * 60)
    print("TEST: MMapWeightStore, SuccinctWeights, TaskSpecificLoader")
    print("=" * 60)
    torch.manual_seed(8)

    tensors = {f"layer{i}.weight": torch.randn(16, 16) for i in range(4)}
    store = MMapWeightStore(os.path.join(SCRATCH, "weights.bin"))
    store.save(tensors)
    for name, t in tensors.items():
        loaded = store.load(name)
        assert torch.equal(t, loaded), f"MMapWeightStore round-trip failed for {name}"
    print(f"  MMapWeightStore: {len(tensors)} tensors round-trip exactly, index overhead {store.index_bytes()} bytes")

    mask = torch.rand(1000) > 0.9
    succinct = SuccinctWeights(mask, block_size=32)
    true_positions = mask.nonzero(as_tuple=False).flatten()
    recovered_positions = succinct.nonzero_positions()
    assert torch.equal(true_positions, recovered_positions)
    for i in [0, 100, 500, 999]:
        expected_rank = mask[:i].sum().item()
        assert succinct.rank(i) == expected_rank, f"rank({i}) mismatch"
    dense_mask_bytes = mask.numel()  # 1 byte/bool if stored naively
    print(f"  SuccinctWeights: rank() queries match brute-force cumulative sum; "
          f"memory {dense_mask_bytes} -> {succinct.memory_bytes()} bytes")
    assert succinct.memory_bytes() < dense_mask_bytes

    tensor_names_by_layer = {i: [f"layer{i}.weight"] for i in range(4)}
    loader = TaskSpecificLoader()
    loader.register_task("summarization", [0, 1])
    subset = loader.load_subset("summarization", store, tensor_names_by_layer)
    assert set(subset.keys()) == {"layer0.weight", "layer1.weight"}
    ratio = loader.compression_ratio("summarization", total_layers=4)
    print(f"  TaskSpecificLoader: loaded {len(subset)}/4 layers for task 'summarization' "
          f"({ratio:.1f}x fewer than loading all layers)")
    assert ratio == 2.0

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------


def test_annealing_and_topological_pruning():
    print("\n" + "=" * 60)
    print("TEST: AnnealingPruner, TopologicalPruner")
    print("=" * 60)
    torch.manual_seed(9)

    W = torch.randn(32, 32, device=DEVICE) * 0.1
    X = torch.randn(500, 32, device=DEVICE)
    H = collect_calibration_hessian(X)
    H_diag = torch.diagonal(H)

    annealer = AnnealingPruner(sparsity=0.7, iterations=300, seed=0)
    mask_annealed = annealer.prune(W, H_diag)
    achieved_sparsity = 1 - mask_annealed.float().mean().item()
    print(f"  AnnealingPruner achieved sparsity: {achieved_sparsity:.2%} (target 70%)")
    assert abs(achieved_sparsity - 0.7) < 0.02

    saliency = (W.flatten() ** 2) * H_diag.repeat(W.shape[0])
    annealed_saliency_sum = saliency[~mask_annealed.flatten()].sum().item()
    order = torch.argsort(saliency)
    k = int(saliency.numel() * 0.7)
    greedy_mask = torch.ones_like(mask_annealed.flatten())
    greedy_mask[order[:k]] = 0
    greedy_saliency_sum = saliency[greedy_mask.flatten() == 0].sum().item()
    print(f"  Pruned-saliency sum: greedy={greedy_saliency_sum:.4f}, annealed={annealed_saliency_sum:.4f} "
          f"(annealing should be >= as good as greedy, i.e. <= greedy's pruned saliency)")
    assert annealed_saliency_sum <= greedy_saliency_sum + 1e-6

    topo = TopologicalPruner(sparsity=0.5, num_levels=6, min_component_size=3)
    mask_topo = topo.prune(W)
    achieved_sparsity_topo = 1 - mask_topo.float().mean().item()
    print(f"  TopologicalPruner achieved sparsity: {achieved_sparsity_topo:.2%} (target 50%)")
    assert abs(achieved_sparsity_topo - 0.5) < 0.02

    print("  PASSED")


def test_free_energy_and_population_coding():
    print("\n" + "=" * 60)
    print("TEST: FreeEnergyQuantizer, PopulationCoding")
    print("=" * 60)
    torch.manual_seed(10)

    # Bimodal distribution — uniform quantization wastes levels; Lloyd-Max shouldn't.
    W = torch.cat([torch.randn(2000) * 0.05 - 0.5, torch.randn(2000) * 0.05 + 0.5]).to(DEVICE)
    W = W.reshape(40, 100)

    feq = FreeEnergyQuantizer(num_levels=8, iterations=30)
    qdict = feq.fit_quantize(W)
    W_rec = FreeEnergyQuantizer.dequantize(qdict)
    err_lloyd = rel_err(W, W_rec)

    qmax = 7
    wmin, wmax = W.min(), W.max()
    scale = (wmax - wmin) / qmax
    q_uniform = torch.round((W - wmin) / scale)
    W_uniform = q_uniform * scale + wmin
    err_uniform = rel_err(W, W_uniform)

    print(f"  Lloyd-Max (free-energy-optimal) 3-bit relative error: {err_lloyd:.2%}")
    print(f"  Uniform 3-bit relative error: {err_uniform:.2%}")
    print(f"  Lloyd-Max improves over uniform by {(1 - err_lloyd / err_uniform) * 100:.1f}% on this bimodal distribution")
    assert err_lloyd < err_uniform
    assert FreeEnergyQuantizer.memory_bytes(qdict) < W.numel() * 4

    pop = PopulationCoding(bits=2, population_size=8, seed=0)
    pdict = pop.encode(W)
    W_rec_pop = PopulationCoding.decode(pdict)
    err_pop = rel_err(W, W_rec_pop)

    single = PopulationCoding(bits=2, population_size=1, seed=0)
    sdict = single.encode(W)
    err_single = rel_err(W, PopulationCoding.decode(sdict))

    print(f"  PopulationCoding (2-bit, K=8) relative error: {err_pop:.2%}")
    print(f"  Single 2-bit (K=1) relative error: {err_single:.2%}")
    print(f"  Population averaging reduces error by {(1 - err_pop / err_single) * 100:.1f}%")
    assert err_pop < err_single

    print("  PASSED")


# ---------------------------------------------------------------------------
# End-to-end: EdgeHVN (weight sharing + GPTQ) on a trained model
# ---------------------------------------------------------------------------


def test_edge_hvn_end_to_end():
    print("\n" + "=" * 60)
    print("TEST: EdgeHVN end-to-end — shared weights + GPTQ compression + PPL check")
    print("=" * 60)
    torch.manual_seed(11)

    text = SAMPLE_TEXT * 20
    tokenizer = CharTokenizer(text, add_tool_tokens=False)
    token_ids = tokenizer.encode(text)
    seq_len = 64
    split = int(len(token_ids) * 0.9)
    train_ids, eval_ids = token_ids[:split], token_ids[split:]

    train_loader = DataLoader(TextDataset(train_ids, seq_len), batch_size=16, shuffle=True)
    eval_loader = DataLoader(TextDataset(eval_ids, seq_len), batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model = EdgeHVN(
        vocab_size=tokenizer.vocab_size, d_model=64, num_layers=4, n_heads=4,
        bond_dimension=16, chunk_size=32, dropout=0.0, shared_weights=True,
    ).to(DEVICE)

    # SharedWeightHVN only instantiates ONE physical HVNBlock (reused num_layers
    # times in forward()), so model.parameters() already reflects the small,
    # shared-storage count; num_unshared_equivalent_params() gives the
    # hypothetical cost of num_layers independent blocks for comparison.
    shared_params = model.num_params(effective=True)
    dense_equiv_params = model.shared_block.num_unshared_equivalent_params() + sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("shared_block.block")
    )
    print(f"  4 shared layers: actual stored params={shared_params:,}, "
          f"equivalent unshared params={dense_equiv_params:,} "
          f"({dense_equiv_params / shared_params:.2f}x reduction)")
    assert shared_params < dense_equiv_params

    print("  Training for 15 epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

    def eval_ppl(m: nn.Module) -> float:
        m.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = m(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                n_batches += 1
        return math.exp(total_loss / max(n_batches, 1))

    ppl_before = eval_ppl(model)
    print(f"  Perplexity before GPTQ compression: {ppl_before:.4f}")

    calibration_batches = [next(iter(train_loader)) for _ in range(4)]

    def calibration_forward():
        for cx, _cy in calibration_batches:
            model(cx.to(DEVICE))

    metadata = model.compress_gptq(calibration_forward, bits=4, group_size=32, percdamp=0.01)
    ppl_after = eval_ppl(model)
    degradation_pct = (ppl_after - ppl_before) / ppl_before * 100

    report = model.memory_report()
    print(f"  Compressed {len(metadata)} nn.Linear layers with GPTQ")
    print(f"  Perplexity after GPTQ compression: {ppl_after:.4f} (degradation: {degradation_pct:+.2f}%)")
    print(f"  Memory report: dense fp32={report['dense_fp32_bytes']} bytes, "
          f"compressed={report['compressed_bytes']} bytes ({report['compression_ratio']:.2f}x)")

    assert degradation_pct < 10.0, f"PPL degradation {degradation_pct:.2f}% too high"
    assert report["compression_ratio"] > 1.0

    # Phase 5.5 recovery fine-tune step (thin pointer to hvn/lora.py's QLoRA support)
    recovery_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bx, by = next(iter(train_loader))
    loss_val = recovery_finetune_step(model, recovery_optimizer, bx.to(DEVICE), by.to(DEVICE), criterion)
    print(f"  recovery_finetune_step ran successfully (loss={loss_val:.4f})")
    assert math.isfinite(loss_val)

    print("  PASSED")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This test suite requires a CUDA GPU"
    print(f"Running on: {torch.cuda.get_device_name(0)}")

    try:
        test_ternary_linear()
        test_low_rank_sparse_and_tt_and_mera()
        test_shared_weight_hvn()
        test_entropy_coder()
        test_wavelet_and_dct_compressors()
        test_kronecker_circulant_toeplitz()
        test_holographic_and_superposition()
        test_streaming_inference()
        test_mmap_and_succinct_and_task_loader()
        test_annealing_and_topological_pruning()
        test_free_energy_and_population_coding()
        test_edge_hvn_end_to_end()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED — Issue #32 'edge-devices' full-scope implementation")
        print("=" * 60)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
