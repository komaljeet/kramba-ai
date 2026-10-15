"""
HVNBlock
========

The wrapper that links the fluid vortex step (VortexInjectionLayer) to the
quantum state evolution (HamiltonianMemoryBlock).

For each token in the sequence:
  1. VortexInjectionLayer computes the velocity field v_t = Γ · (H · ω_t) and
     produces the deflected representation r̃_t.
  2. HamiltonianMemoryBlock absorbs v_t into H via entanglement renormalization.

Key insight: the Hamiltonian H is updated *sequentially* token-by-token,
but each update is O(d_model²) — constant in N.

**Multi-head Hamiltonian**: Instead of a single H matrix, we maintain `n_heads`
parallel Hamiltonian matrices, each operating on a `d_head = d_model // n_heads`
slice of the hidden dimension.  This is analogous to multi-head attention and
allows the model to capture different types of associations simultaneously.

**Chunked parallel processing**: For training speed, tokens are processed in
chunks of `chunk_size`.  Within a chunk, all tokens share the same Hamiltonian
state (computed from the previous chunk), enabling parallel GPU computation.
The Hamiltonian is then updated sequentially with the chunk's velocities.  This
trades a small amount of causal accuracy (tokens within a chunk see the H from
*before* the chunk, not the intra-chunk updates) for a large speedup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vortex_injection import VortexInjectionLayer
from .hamiltonian_memory import HamiltonianMemoryBlock
from .batched_vortex import MultiHeadVortexInjection, MultiThreadVortexInjection
from .batched_memory import MultiHeadHamiltonianMemory, MultiThreadHamiltonianMemory


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (faster than LayerNorm).

    RMSNorm normalizes by the RMS of the activations without subtracting the mean:
        x_norm = x / sqrt(mean(x²) + eps) * weight

    This is faster than LayerNorm (no mean computation) and works better
    with no-bias architectures. Used by GPT-J, PaLM, LLaMA, etc.
    """

    #: Fused path (2026-10-15): ``F.rms_norm``'s single CUDA kernel instead of
    #: the six-op eager chain below (pow, mean, add, rsqrt, mul, mul) and its
    #: ~10-op autograd backward. Measured at the 1B shape [1, 4096, 2048] fp32:
    #: 1.33 -> 0.39 ms forward, 6.35 -> 1.89 ms forward+backward. Same formula,
    #: different reduction order, so ulp-level rather than bit-identical (like
    #: the fused l2norm). A class attribute, set per module by
    #: ``apply_fast_stack`` under ``--fast``, so it never enters a state_dict.
    #: fp32 input + fp32 weight on CUDA only: for bf16 inputs the eager chain
    #: rounds every intermediate to bf16 and the fused kernel does not, which
    #: would be a real numerics change rather than an ulp one.
    use_fused = False

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.use_fused and x.is_cuda and x.dtype == torch.float32
                and self.weight.dtype == torch.float32):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU (Gated Linear Unit) feed-forward network (Strategy 6).

    Used by LLaMA, PaLM, and other modern Transformer architectures.
    SwiGLU replaces the standard GELU FFN with a gated mechanism:

        FFN(x) = W2(silu(W1(x)) * W3(x))

    where silu is the SiLU/Swish activation.  This provides:
    - Better quality than standard GELU FFN (proven in LLaMA)
    - Gated information flow (the W3 gate selectively passes information)
    - 3× FFN FLOPs (but FFN is a small fraction of total cost)

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    expansion : int, optional
        Expansion factor for the hidden dimension.  Default: 3 (as in LLaMA).
    dropout : float, optional
        Dropout rate.
    """

    def __init__(self, d_model: int, expansion: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CrossHeadMixer(nn.Module):
    """Cross-head Hamiltonian mixing layer (Strategy 3).

    After per-head Hamiltonian updates, this module mixes Hamiltonian states
    across heads using a learned mixing weight — analogous to the output
    projection in multi-head attention.

        H_mixed[h] = Σ_{h'} α_{h,h'} · H[h']

    The mixing weight is initialized as the identity (no mixing), so the
    model starts with independent heads and can learn to share associations.

    Parameters
    ----------
    n_heads : int
        Number of parallel heads.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.eye(n_heads))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Mix Hamiltonian states across heads.

        Parameters
        ----------
        H : torch.Tensor
            Hamiltonian states, shape [B, n_heads, d_head, d_head].

        Returns
        -------
        H_mixed : torch.Tensor
            Mixed Hamiltonian states, same shape.
        """
        # H_mixed[b, h] = Σ_{h'} mix_weight[h, h'] · H[b, h']
        # mix_weight: [n_heads_out, n_heads_in], H: [B, n_heads_in, d, k]
        return torch.einsum("hj,bjdk->bhdk", self.mix_weight, H)


class HVNBlock(nn.Module):
    """A single Hamiltonian Vortex Network block with multi-head memory.

    Combines fluid-dynamical vortex injection with quantum entanglement memory
    evolution into a single residual block.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of parallel Hamiltonian heads (like multi-head attention).
        Each head operates on d_model // n_heads dimensions.
    bond_dimension : int, optional
        SVD truncation rank for quantum renormalization.  None = full rank.
    chunk_size : int | None
        Number of tokens to process in parallel within a chunk.  Larger = faster
        but less causally accurate within the chunk.  Defaults to 64, and can be
        set to None to use an adaptive chunk size based on the sequence length.
    dropout : float
        Dropout rate applied to the block output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        bond_dimension: int | None = None,
        chunk_size: int | None = 64,
        dropout: float = 0.1,
        rank_k: int = 1,
        use_nonlinear_update: bool = False,
        use_causal_reinjection: bool = False,
        use_cross_head_mixing: bool = False,
        use_swiglu: bool = False,
        use_per_token_eta: bool = False,
        n_threads: int = 1,
        use_parallel_scan: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.rank_k = rank_k
        self.use_nonlinear_update = use_nonlinear_update
        self.use_causal_reinjection = use_causal_reinjection
        self.use_cross_head_mixing = use_cross_head_mixing
        self.use_per_token_eta = use_per_token_eta
        self.n_threads = n_threads
        self.n_units = n_heads * n_threads  # total (head, thread) pairs
        self.use_parallel_scan = use_parallel_scan  # Issue #33 Fix 3

        # Batched multi-head vortex injection
        # Use MultiThreadVortexInjection when n_threads > 1, else MultiHeadVortexInjection
        if n_threads > 1:
            self.vortex = MultiThreadVortexInjection(self.d_head, n_heads, n_threads)
        else:
            self.vortex = MultiHeadVortexInjection(self.d_head, n_heads)

        # Batched multi-head quantum memory
        # Use MultiThreadHamiltonianMemory when n_threads > 1, else MultiHeadHamiltonianMemory
        if n_threads > 1:
            self.memory = MultiThreadHamiltonianMemory(
                self.d_head,
                n_heads,
                n_threads=n_threads,
                bond_dimension=bond_dimension,
                rank_k=rank_k,
                use_nonlinear_update=use_nonlinear_update,
            )
        else:
            self.memory = MultiHeadHamiltonianMemory(
                self.d_head,
                n_heads,
                bond_dimension=bond_dimension,
                rank_k=rank_k,
                use_nonlinear_update=use_nonlinear_update,
            )

        # Backward-compatible references (for checkpoint loading from old format)
        self.vortex_heads = nn.ModuleList([
            VortexInjectionLayer(self.d_head)
            for _ in range(n_heads)
        ])
        self.memory_heads = nn.ModuleList([
            HamiltonianMemoryBlock(self.d_head, bond_dimension=bond_dimension)
            for _ in range(n_heads)
        ])

        # Input projection: d_model → n_heads * d_head (identity if d_model == n_heads * d_head)
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Output projection: n_heads * d_head → d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Strategy 6: SwiGLU FFN (replaces standard GELU FFN)
        if use_swiglu:
            self.ffn = SwiGLU(d_model, expansion=3, dropout=dropout)
        else:
            # Standard 2x expansion GELU FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

        # Strategy 3: Cross-head Hamiltonian mixing
        if use_cross_head_mixing:
            self.cross_head_mixer = CrossHeadMixer(n_heads)
        else:
            self.cross_head_mixer = None

        # RMSNorm (faster than LayerNorm — no mean subtraction)
        self.norm = RMSNorm(d_model)

        # Dropout for the vortex sub-layer output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequence through the multi-head HVN block.

        Uses parallel residual architecture (like GPT-J / PaLM):
            x = x + vortex(norm(x)) + ffn(norm(x))
        Both branches are computed from the same normed input in parallel.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape [batch, seq_len, d_model].

        Returns
        -------
        output : torch.Tensor
            Output representations, shape [batch, seq_len, d_model].
        """
        B, S, D = x.shape

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        # Single shared norm for parallel residual (both branches use same norm)
        x_norm = self.norm(x)  # [B, S, D]

        # --- Vortex branch ---
        x_proj = self.input_proj(x_norm)  # [B, S, D]
        x_heads = x_proj.view(B, S, self.n_heads, self.d_head)
        x_batched = x_heads.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, S, self.d_head)

        if position_ids is not None:
            pos_batched = position_ids.unsqueeze(1).expand(B, self.n_heads, S).contiguous()
            pos_batched = pos_batched.view(B * self.n_heads, S)
        else:
            pos_batched = None

        out_batched = self._process_batched_chunked(x_batched, B, S, pos_batched)

        deflected = out_batched.view(B, self.n_heads, S, self.d_head).permute(0, 2, 1, 3).contiguous()
        deflected = deflected.view(B, S, self.n_heads * self.d_head)
        deflected = self.output_proj(deflected)

        # --- FFN branch (computed in parallel with vortex) ---
        ffn_out = self.ffn(x_norm)

        # Parallel residual: x = x + vortex(norm(x)) + ffn(norm(x))
        x = x + self.dropout(deflected) + ffn_out

        return x

    def _process_batched_chunked(
        self,
        x_batched: torch.Tensor,
        B: int,
        S: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process all heads simultaneously with chunked parallel vortex injection.

        Supports multi-thread memory (Issue #22), Strategy 2 (causal re-injection),
        Strategy 3 (cross-head mixing), and Strategy 5 (per-token η).

        When ``n_threads > 1``, the Hamiltonian has shape ``[B*n_units, D, D]``
        where ``n_units = n_heads * n_threads``.  The vortex injection handles
        thread expansion internally and returns mixed deflected outputs
        (shape ``[B*n_heads, S, D]``) and per-thread velocities
        (shape ``[B*n_units, S, D]``).
        """
        BH = B * self.n_heads  # effective batch size for vortex I/O
        BU = B * self.n_units  # effective batch size for Hamiltonian (heads × threads)

        # Initialize Hamiltonian for all (head, thread) pairs at once
        H = self.memory.init_hamiltonian(B, device=x_batched.device, dtype=x_batched.dtype)
        # H shape: [B*n_units, d_head, d_head]

        # --- Expand inputs for threads (for memory updates) ---
        if self.n_threads > 1:
            # x_batched: [BH, S, D] → expand to [BU, S, D] for per-thread memory
            x_expanded = x_batched.view(B, self.n_heads, S, self.d_head)
            x_expanded = x_expanded.unsqueeze(2).expand(
                B, self.n_heads, self.n_threads, S, self.d_head
            )
            x_threaded = x_expanded.reshape(BU, S, self.d_head)
        else:
            x_threaded = x_batched

        if self.use_causal_reinjection:
            # --- Strategy 2: Two-pass causal re-injection ---
            # PASS 1: Compute velocities with initial H, update H chunk-by-chunk
            deflected_pass1, velocity_all = self.vortex(
                x_batched, H, position_ids=position_ids
            )  # deflected: [BH, S, d_head], velocity: [BU, S, d_head]

            # Update H chunk-by-chunk to get the correct H at each chunk boundary
            chunk_size = self._resolve_chunk_size(S)
            H_per_chunk = [H]  # H before any chunk
            H_current = H
            for chunk_start in range(0, S, chunk_size):
                chunk_end = min(chunk_start + chunk_size, S)
                velocity_chunk = velocity_all[:, chunk_start:chunk_end, :]
                input_chunk = x_threaded[:, chunk_start:chunk_end, :]
                if self.use_per_token_eta:
                    H_current = self.memory.update_batch_per_token(
                        H_current, velocity_chunk, inputs=input_chunk
                    )
                else:
                    H_current = self.memory.update_batch(
                        H_current, velocity_chunk, inputs=input_chunk
                    )
                H_per_chunk.append(H_current)

            # Strategy 3: Cross-head mixing after H updates
            if self.cross_head_mixer is not None:
                H_mixed = H_current.view(B, self.n_units, self.d_head, self.d_head)
                # Cross-head mixing operates on heads, averaging over threads
                H_heads = H_mixed.view(B, self.n_heads, self.n_threads, self.d_head, self.d_head)
                H_heads = H_heads.mean(dim=2)  # [B, n_heads, d_head, d_head]
                H_heads = self.cross_head_mixer(H_heads)
                # Broadcast back to threads
                H_heads = H_heads.unsqueeze(2).expand(B, self.n_heads, self.n_threads, self.d_head, self.d_head)
                H_current = H_heads.reshape(BU, self.d_head, self.d_head)

            # PASS 2: Re-compute deflected representations with updated H per chunk
            deflected_all = torch.empty_like(deflected_pass1)
            for chunk_idx, chunk_start in enumerate(range(0, S, chunk_size)):
                chunk_end = min(chunk_start + chunk_size, S)
                x_chunk = x_batched[:, chunk_start:chunk_end, :]
                # Use the H that was updated by PREVIOUS chunks (not the initial H)
                H_for_chunk = H_per_chunk[chunk_idx]
                chunk_pos = None
                if position_ids is not None:
                    chunk_pos = position_ids[:, chunk_start:chunk_end]
                deflected_chunk, _ = self.vortex(
                    x_chunk, H_for_chunk, position_ids=chunk_pos
                )
                deflected_all[:, chunk_start:chunk_end, :] = deflected_chunk

            return deflected_all  # [BH, S, d_head]

        # --- Default path (no causal re-injection) ---
        # Phase 1: Vortex injection for the ENTIRE sequence in one call
        deflected_all, velocity_all = self.vortex(
            x_batched, H, position_ids=position_ids
        )  # deflected: [BH, S, d_head], velocity: [BU, S, d_head]

        # Phase 2: Hamiltonian update
        if self.use_parallel_scan and hasattr(self.memory, 'parallel_update'):
            # Issue #33 Fix 3: Parallel scan (associative Hamiltonian update)
            # Replaces the sequential chunk loop with a single parallel scan.
            # O(log S) parallel depth instead of O(S/chunk_size) sequential steps.
            H = self.memory.parallel_update(
                H, velocity_all, inputs=x_threaded,
                correction_pass=True,
            )
        else:
            # Original chunked sequential update
            chunk_size = self._resolve_chunk_size(S)
            for chunk_start in range(0, S, chunk_size):
                chunk_end = min(chunk_start + chunk_size, S)
                velocity_chunk = velocity_all[:, chunk_start:chunk_end, :]
                input_chunk = x_threaded[:, chunk_start:chunk_end, :]
                if self.use_per_token_eta:
                    H = self.memory.update_batch_per_token(
                        H, velocity_chunk, inputs=input_chunk
                    )
                else:
                    H = self.memory.update_batch(
                        H, velocity_chunk, inputs=input_chunk
                    )

        # Strategy 3: Cross-head mixing after H updates
        if self.cross_head_mixer is not None:
            H_reshaped = H.view(B, self.n_units, self.d_head, self.d_head)
            # Cross-head mixing operates on heads, averaging over threads
            H_heads = H_reshaped.view(B, self.n_heads, self.n_threads, self.d_head, self.d_head)
            H_heads = H_heads.mean(dim=2)  # [B, n_heads, d_head, d_head]
            H_heads = self.cross_head_mixer(H_heads)
            # Broadcast back to threads
            H_heads = H_heads.unsqueeze(2).expand(B, self.n_heads, self.n_threads, self.d_head, self.d_head)
            H = H_heads.reshape(BU, self.d_head, self.d_head)

        return deflected_all  # [BH, S, d_head]

    def parallel_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        num_chunks: int = 4,
        correction_pass: bool = True,
    ) -> torch.Tensor:
        """Multi-chunk parallel forward with barrier sync (Issue #33 Fix 4).

        Splits the sequence into K independent chunks, processes all K in
        parallel, then merges their Hamiltonian updates at a barrier.

        Phase 1 (parallel): Each chunk computes velocities using H_0 (initial state)
        Phase 2 (parallel): Each chunk computes its ΔH_k (Hamiltonian delta)
        Phase 3 (barrier):  Merge: H_final = H_0 + Σ_k ΔH_k
        Phase 4 (optional): Re-run chunks with updated H_final for better accuracy

        This provides K× speedup where K = num_chunks, at the cost of chunks
        not seeing each other's updates during Phase 1.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape [batch, seq_len, d_model].
        position_ids : torch.Tensor, optional
            Position IDs, shape [batch, seq_len].
        num_chunks : int
            Number of parallel chunks (K).
        correction_pass : bool
            If True, re-run vortex injection with the merged H for accuracy.

        Returns
        -------
        output : torch.Tensor
            Output representations, shape [batch, seq_len, d_model].
        """
        B, S, D = x.shape

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        x_norm = self.norm(x)
        x_proj = self.input_proj(x_norm)  # [B, S, D]
        x_heads = x_proj.view(B, S, self.n_heads, self.d_head)
        x_batched = x_heads.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, S, self.d_head)

        pos_batched = position_ids.unsqueeze(1).expand(B, self.n_heads, S).contiguous()
        pos_batched = pos_batched.view(B * self.n_heads, S)

        BH = B * self.n_heads
        BU = B * self.n_units
        H_0 = self.memory.init_hamiltonian(B, device=x_batched.device, dtype=x_batched.dtype)

        # Expand inputs for threads
        if self.n_threads > 1:
            x_expanded = x_batched.view(B, self.n_heads, S, self.d_head)
            x_expanded = x_expanded.unsqueeze(2).expand(B, self.n_heads, self.n_threads, S, self.d_head)
            x_threaded = x_expanded.reshape(BU, S, self.d_head)
        else:
            x_threaded = x_batched

        # --- Phase 1: Compute velocities for ALL chunks using H_0 ---
        # All tokens see the same initial H — fully parallel
        deflected_all, velocity_all = self.vortex(
            x_batched, H_0, position_ids=pos_batched
        )

        # --- Phase 2: Compute ΔH_k for each chunk in parallel ---
        chunk_size = max(1, S // num_chunks)
        chunks = []
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            velocity_chunk = velocity_all[:, chunk_start:chunk_end, :]
            input_chunk = x_threaded[:, chunk_start:chunk_end, :]
            chunks.append((velocity_chunk, input_chunk))

        # Process all chunks simultaneously by stacking them into a batch
        # Each chunk uses H_0 as base → ΔH_k = update(H_0, chunk_k) - H_0
        # We batch all chunks together for a single update_batch call
        if len(chunks) > 1:
            # Stack all velocity chunks: [BU * K, chunk_len, D]
            # But chunks may have different lengths — pad to max
            max_chunk_len = max(vc.shape[1] for vc, _ in chunks)
            K = len(chunks)
            velocity_stacked = torch.zeros(BU, K, max_chunk_len, self.d_head,
                                          device=x_batched.device, dtype=x_batched.dtype)
            input_stacked = torch.zeros(BU, K, max_chunk_len, self.d_head,
                                        device=x_batched.device, dtype=x_batched.dtype)
            for k, (vc, ic) in enumerate(chunks):
                cl = vc.shape[1]
                velocity_stacked[:, k, :cl, :] = vc
                input_stacked[:, k, :cl, :] = ic

            # Reshape to [BU * K, max_chunk_len, D] for batched update
            velocity_flat = velocity_stacked.view(BU * K, max_chunk_len, self.d_head)
            input_flat = input_stacked.view(BU * K, max_chunk_len, self.d_head)

            # Expand H_0 for K chunks: [BU * K, D, D]
            H_0_expanded = H_0.unsqueeze(1).expand(BU, K, self.d_head, self.d_head)
            H_0_expanded = H_0_expanded.reshape(BU * K, self.d_head, self.d_head)

            # Single batched update for all K chunks
            if self.use_per_token_eta:
                H_chunks = self.memory.update_batch_per_token(
                    H_0_expanded, velocity_flat, inputs=input_flat
                )
            else:
                H_chunks = self.memory.update_batch(
                    H_0_expanded, velocity_flat, inputs=input_flat
                )

            # Compute ΔH_k = H_chunk_k - H_0 and sum
            H_chunks = H_chunks.view(BU, K, self.d_head, self.d_head)
            delta_H_sum = (H_chunks - H_0.unsqueeze(1)).sum(dim=1)  # [BU, D, D]
        else:
            # Single chunk — just do a regular update
            vc, ic = chunks[0]
            if self.use_per_token_eta:
                H_final = self.memory.update_batch_per_token(H_0, vc, inputs=ic)
            else:
                H_final = self.memory.update_batch(H_0, vc, inputs=ic)
            delta_H_sum = H_final - H_0

        # --- Phase 3: Barrier merge ---
        H_final = H_0 + delta_H_sum

        # Spectral norm clamping on the merged H
        fro_norm = H_final.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (self.d_head ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        H_final = H_final * scale

        # --- Phase 4: Optional correction pass ---
        if correction_pass:
            # Re-run vortex injection with the merged H_final
            deflected_all, _ = self.vortex(
                x_batched, H_final, position_ids=pos_batched
            )

        # Reshape back to [B, S, D]
        deflected = deflected_all.view(B, self.n_heads, S, self.d_head).permute(0, 2, 1, 3).contiguous()
        deflected = deflected.view(B, S, self.n_heads * self.d_head)
        deflected = self.output_proj(deflected)

        # FFN branch
        ffn_out = self.ffn(x_norm)

        # Parallel residual
        x = x + self.dropout(deflected) + ffn_out

        return x

    def _resolve_chunk_size(self, seq_len: int) -> int:
        """Choose a chunk size that balances parallelism and accuracy."""
        if self.chunk_size is None:
            # For short prompts, use the full sequence to avoid extra bookkeeping.
            # For longer sequences, use a larger chunk to keep the GPU busy.
            return max(1, min(128, seq_len))
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {self.chunk_size}")
        return int(self.chunk_size)

    def _process_head_chunked(
        self,
        head_idx: int,
        x_h: torch.Tensor,
        B: int,
        S: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process one head with chunked parallel vortex injection.

        Tokens within a chunk share the same Hamiltonian state (from before
        the chunk), enabling parallel computation.  After each chunk, the
        Hamiltonian is updated with all velocities from that chunk.
        """
        vortex = self.vortex_heads[head_idx]
        memory = self.memory_heads[head_idx]

        # Initialize Hamiltonian for this head
        H = memory.init_hamiltonian(B, device=x_h.device, dtype=x_h.dtype)  # [B, d_head, d_head]

        chunk_size = self._resolve_chunk_size(S)
        all_outputs = []

        # Process in chunks
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            x_chunk = x_h[:, chunk_start:chunk_end, :]  # [B, chunk_len, d_head]

            # All tokens in this chunk use the SAME Hamiltonian H
            # This enables parallel computation across the chunk
            chunk_position_ids = None
            if position_ids is not None:
                chunk_position_ids = position_ids[:, chunk_start:chunk_end]
            deflected_chunk, velocity_chunk = vortex(x_chunk, H, position_ids=chunk_position_ids)  # [B, chunk_len, d_head] x2
            all_outputs.append(deflected_chunk)

            # Update Hamiltonian with all velocities in the chunk at once (batched)
            # This replaces the sequential for-loop with a single batched operation,
            # eliminating the Python-level bottleneck that caused 63x slowdown.
            H = memory.update_batch(H, velocity_chunk)

        return torch.cat(all_outputs, dim=1)  # [B, S, d_head]

    def init_state(self, batch_size: int, device=None, dtype=None) -> list[list[torch.Tensor]]:
        """Initialize the causal Hamiltonian state for all heads (incremental generation).

        Returns a list of per-head states for backward compatibility with
        the incremental generation path (which still uses the per-head loop
        for simplicity — it only processes 1 token at a time so speed is fine).
        """
        return [
            [head.init_hamiltonian(batch_size, device=device, dtype=dtype)]
            for head in self.memory_heads
        ]

    def init_state_batched(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Initialize batched Hamiltonian state for all heads at once.

        Uses the batched MultiHeadHamiltonianMemory module, producing a single
        tensor of shape [B*n_heads, d_head, d_head] instead of a list of per-head
        states.  This is the efficient path for batched multi-sequence generation.

        Returns
        -------
        H : torch.Tensor, shape [B*n_heads, d_head, d_head]
        """
        return self.memory.init_hamiltonian(batch_size, device=device, dtype=dtype)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: list[list[torch.Tensor]],
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """Process a SINGLE token incrementally — O(1) per token generation.

        Parameters
        ----------
        x_t : torch.Tensor
            Single token embedding, shape [batch, 1, d_model].
        state : list[list[torch.Tensor]]
            Per-head Hamiltonian states: [[H_0], [H_1], ...].

        Returns
        -------
        output : torch.Tensor
            Output for this token, shape [batch, 1, d_model].
        new_state : list[list[torch.Tensor]]
            Updated per-head Hamiltonian states.
        """
        B, _, D = x_t.shape

        if position_idx is None:
            position_ids = torch.arange(1, device=x_t.device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.full((B, 1), int(position_idx), device=x_t.device, dtype=torch.long)

        # Single shared norm for parallel residual (matches forward())
        x_norm = self.norm(x_t)  # [B, 1, D]
        x_proj = self.input_proj(x_norm)  # [B, 1, D]
        x_heads = x_proj.view(B, 1, self.n_heads, self.d_head)  # [B, 1, n_heads, d_head]

        head_outputs = []
        new_state = []
        for h in range(self.n_heads):
            x_h = x_heads[:, :, h, :]  # [B, 1, d_head]
            H = state[h][0]  # [B, d_head, d_head]

            # Vortex injection
            deflected_t, velocity_t = self.vortex_heads[h](x_h, H, position_ids=position_ids)

            # Update Hamiltonian
            H_new = self.memory_heads[h].update(H, velocity_t.squeeze(1))

            head_outputs.append(deflected_t)
            new_state.append([H_new])

        # Concatenate and project
        deflected = torch.cat(head_outputs, dim=-1)  # [B, 1, D]
        deflected = self.output_proj(deflected)

        # Parallel residual: x = x + vortex(norm(x)) + ffn(norm(x))
        x_t = x_t + self.dropout(deflected) + self.ffn(x_norm)

        return x_t, new_state

    def forward_step_fused(
        self,
        x_t: torch.Tensor,
        H: torch.Tensor,
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused single-token forward step using Triton kernels (Issue #33 Fix 1).

        Uses the FusedHVNStep Triton kernel to combine vortex injection and
        Hamiltonian update into 2 kernel launches instead of ~25 PyTorch ops.

        This is the fast path for incremental generation — same math as
        ``forward_step_batched`` but with fused Triton kernels.

        Parameters
        ----------
        x_t : torch.Tensor
            Single token embedding, shape [B, 1, d_model].
        H : torch.Tensor
            Batched Hamiltonian state, shape [B*n_heads, d_head, d_head].
        position_idx : int, optional
            Position index for RoPE.

        Returns
        -------
        output : torch.Tensor
            Output for this token, shape [B, 1, d_model].
        H_new : torch.Tensor
            Updated Hamiltonian state, same shape as input H.
        """
        from ..kernels import FusedHVNStep

        B, _, D = x_t.shape

        if position_idx is None:
            position_ids = torch.arange(1, device=x_t.device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.full((B, 1), int(position_idx), device=x_t.device, dtype=torch.long)

        # Single shared norm for parallel residual (matches forward())
        x_norm = self.norm(x_t)  # [B, 1, D]
        x_proj = self.input_proj(x_norm)  # [B, 1, D]
        x_heads = x_proj.view(B, 1, self.n_heads, self.d_head)  # [B, 1, n_heads, d_head]

        # Reshape to [B*n_heads, 1, d_head] for batched processing
        x_batched = x_heads.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, 1, self.d_head)

        # Use fused Triton kernel
        if not hasattr(self, '_fused_step'):
            self._fused_step = FusedHVNStep(self)

        deflected_batched, velocity_batched, H_new = self._fused_step.step(
            x_batched, H, position_idx=position_idx or 0
        )

        # --- Reshape back to [B, 1, D] ---
        deflected = deflected_batched.view(B, self.n_heads, 1, self.d_head).permute(0, 2, 1, 3).contiguous()
        deflected = deflected.view(B, 1, self.n_heads * self.d_head)  # [B, 1, D]
        deflected = self.output_proj(deflected)

        # Parallel residual: x = x + vortex(norm(x)) + ffn(norm(x))
        x_t = x_t + self.dropout(deflected) + self.ffn(x_norm)

        return x_t, H_new

    def supports_served(self) -> bool:
        """Whether this block has the concurrent-serving decode path (Issue #43).

        The grouped-GEMM serving step is implemented for the default single-thread
        multi-head HVN block.  Multi-thread memory (Issue #22) is not covered.
        """
        return (
            self.n_threads == 1
            and isinstance(self.vortex, MultiHeadVortexInjection)
            and hasattr(self.memory, "update_single_served")
        )

    def forward_step_served(
        self,
        x_t: torch.Tensor,
        H: torch.Tensor,
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concurrent-serving decode step: grouped GEMM over streams (Issue #43).

        Same math as :meth:`forward_step_batched` but routes the shared-weight
        vortex projections through :meth:`MultiHeadVortexInjection.forward_served`
        (grouped GEMM across concurrent streams instead of per-stream GEMVs with a
        replicated weight) and the Hamiltonian update through
        :meth:`update_single_served` (batch-outer ordering).  For a single stream
        (``B == 1``) it is bit-for-bit the same routing as ``forward_step_batched``;
        for ``B > 1`` it keeps every stream's output identical to running it alone.
        """
        B, _, D = x_t.shape

        if position_idx is None:
            position_ids = torch.arange(1, device=x_t.device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.full((B, 1), int(position_idx), device=x_t.device, dtype=torch.long)

        x_norm = self.norm(x_t)  # [B, 1, D]
        x_proj = self.input_proj(x_norm)  # [B, 1, D]
        x_heads = x_proj.view(B, 1, self.n_heads, self.d_head)
        # Batch-outer layout: bh = b*n_heads + h
        x_batched = x_heads.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, 1, self.d_head)

        pos_batched = position_ids.unsqueeze(1).expand(B, self.n_heads, 1).contiguous()
        pos_batched = pos_batched.view(B * self.n_heads, 1)

        deflected_batched, velocity_batched = self.vortex.forward_served(
            x_batched, H, position_ids=pos_batched
        )

        velocity_single = velocity_batched.squeeze(1)          # [B*n_heads, d_head]
        input_single = x_batched.squeeze(1)                    # [B*n_heads, d_head]
        H_new = self.memory.update_single_served(H, velocity_single, input_vec=input_single)

        deflected = deflected_batched.view(B, self.n_heads, 1, self.d_head).permute(0, 2, 1, 3).contiguous()
        deflected = deflected.view(B, 1, self.n_heads * self.d_head)
        deflected = self.output_proj(deflected)

        x_t = x_t + self.dropout(deflected) + self.ffn(x_norm)

        return x_t, H_new

    def forward_step_batched(
        self,
        x_t: torch.Tensor,
        H: torch.Tensor,
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single token for ALL sequences + heads in one batched call.

        This is the efficient batched path for multi-sequence generation.  Instead
        of looping over heads (as in forward_step), it uses the batched
        vortex injection and Hamiltonian memory modules to process all
        sequences and all heads simultaneously.

        When ``n_threads > 1``, the Hamiltonian state has shape
        ``[B*n_units, d_head, d_head]`` where ``n_units = n_heads * n_threads``.

        Parameters
        ----------
        x_t : torch.Tensor
            Single token embedding, shape [B, 1, d_model].
        H : torch.Tensor
            Batched Hamiltonian state, shape [B*n_units, d_head, d_head]
            (or [B*n_heads, d_head, d_head] when n_threads=1).
        position_idx : int, optional
            Position index for this token (for RoPE).

        Returns
        -------
        output : torch.Tensor
            Output for this token, shape [B, 1, d_model].
        H_new : torch.Tensor
            Updated Hamiltonian state, same shape as input H.
        """
        B, _, D = x_t.shape

        if position_idx is None:
            position_ids = torch.arange(1, device=x_t.device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.full((B, 1), int(position_idx), device=x_t.device, dtype=torch.long)

        # Single shared norm for parallel residual (matches forward())
        x_norm = self.norm(x_t)  # [B, 1, D]
        x_proj = self.input_proj(x_norm)  # [B, 1, D]
        x_heads = x_proj.view(B, 1, self.n_heads, self.d_head)  # [B, 1, n_heads, d_head]

        # Reshape to [B*n_heads, 1, d_head] for batched processing
        x_batched = x_heads.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, 1, self.d_head)

        # Expand position_ids for all heads: [B*n_heads, 1]
        pos_batched = position_ids.unsqueeze(1).expand(B, self.n_heads, 1).contiguous()
        pos_batched = pos_batched.view(B * self.n_heads, 1)

        # --- Batched vortex injection (all heads/threads at once) ---
        # When n_threads > 1: deflected is [B*n_heads, 1, D] (mixed),
        #                      velocity is [B*n_units, 1, D] (per-thread, unmixed)
        # When n_threads = 1: both are [B*n_heads, 1, D]
        deflected_batched, velocity_batched = self.vortex(
            x_batched, H, position_ids=pos_batched
        )

        # --- Batched Hamiltonian update ---
        # velocity_batched: [B*n_units, 1, d_head] → squeeze to [B*n_units, d_head]
        velocity_single = velocity_batched.squeeze(1)

        # Expand input for per-thread memory updates
        if self.n_threads > 1:
            # x_batched: [B*n_heads, 1, D] → expand to [B*n_units, 1, D]
            x_expanded = x_batched.view(B, self.n_heads, 1, self.d_head)
            x_expanded = x_expanded.unsqueeze(2).expand(
                B, self.n_heads, self.n_threads, 1, self.d_head
            )
            input_single = x_expanded.reshape(B * self.n_units, self.d_head)
        else:
            input_single = x_batched.squeeze(1)  # [B*n_heads, d_head]

        H_new = self.memory.update_single(H, velocity_single, input_vec=input_single)

        # --- Reshape back to [B, 1, D] ---
        deflected = deflected_batched.view(B, self.n_heads, 1, self.d_head).permute(0, 2, 1, 3).contiguous()
        deflected = deflected.view(B, 1, self.n_heads * self.d_head)  # [B, 1, D]
        deflected = self.output_proj(deflected)

        # Parallel residual: x = x + vortex(norm(x)) + ffn(norm(x))
        x_t = x_t + self.dropout(deflected) + self.ffn(x_norm)

        return x_t, H_new