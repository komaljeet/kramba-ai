"""
Batched Multi-Head Vortex Injection
====================================

A fully batched version of VortexInjectionLayer that processes ALL heads
simultaneously by stacking per-head weights into batched tensors.

Instead of looping over n_heads (each with its own nn.Linear), this module
uses a single batched matmul (torch.bmm) to compute all heads in parallel.

Shapes use BH = B * n_heads as the batch dimension, so all heads are
processed as one large batch.
"""

import torch
import torch.nn as nn
import math

from .vortex_injection import _apply_rotary_embeddings


class MultiHeadVortexInjection(nn.Module):
    """Batched multi-head vortex injection layer.

    Combines n_heads VortexInjectionLayer instances into a single module
    that processes all heads in parallel using batched operations.

    Parameters
    ----------
    d_head : int
        Per-head dimensionality.
    n_heads : int
        Number of parallel heads.
    """

    def __init__(self, d_head: int, n_heads: int):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        # --- Batched vorticity projection: ω_t = W_ω · x_t ---
        # Weight shape: [n_heads, d_head, d_head] — one matrix per head
        self.vorticity_weight = nn.Parameter(
            torch.empty(n_heads, d_head, d_head)
        )
        for h in range(n_heads):
            nn.init.normal_(self.vorticity_weight[h], mean=0.0, std=1.0 / math.sqrt(d_head))

        # --- Per-head vorticity frequency (Fourier-style specialisation) ---
        # Each head gets a distinct frequency scaling factor for the rotary
        # positional embeddings.  Heads with high frequency capture local
        # patterns; heads with low frequency capture global structure.
        # We initialise with a geometric series from high to low frequency,
        # analogous to how multi-head attention heads specialise.
        #
        #   head 0:           freq = 2^0  = 1.0   (base frequency)
        #   head 1:           freq = 2^1  = 2.0   (higher — more local)
        #   ...
        #   head n_heads-1:   freq = 2^(n-1)     (highest — most local)
        #
        # The parameter is learnable so the model can adjust during training.
        freq_init = torch.tensor(
            [2.0 ** h for h in range(n_heads)],
            dtype=torch.float32,
        )
        self.vorticity_frequency = nn.Parameter(freq_init)  # [n_heads]

        # --- Batched coupling constant Γ (one per head) ---
        self.gamma = nn.Parameter(torch.full((n_heads, 1), 1.0 / math.sqrt(d_head)))

        # --- Batched gate projection: g_t = σ(W_g · [x_t ; v_t]) ---
        # Weight: [n_heads, d_head, 2*d_head], Bias: [n_heads, d_head]
        self.gate_weight = nn.Parameter(torch.empty(n_heads, d_head, 2 * d_head))
        self.gate_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        for h in range(n_heads):
            nn.init.xavier_uniform_(self.gate_weight[h])

        # --- Batched output projection ---
        self.output_weight = nn.Parameter(torch.empty(n_heads, d_head, d_head))
        for h in range(n_heads):
            nn.init.xavier_uniform_(self.output_weight[h])

        # --- Batched LayerNorm (one per head, but LayerNorm doesn't batch easily) ---
        # We use a single GroupNorm-like approach: normalize over last dim per head
        self.ln_weight = nn.Parameter(torch.ones(n_heads, d_head))
        self.ln_bias = nn.Parameter(torch.zeros(n_heads, d_head))

    def forward(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vortex-velocity-deflected representations for all heads.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape [B*n_heads, S, d_head].
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        position_ids : torch.Tensor, optional
            Position IDs, shape [B*n_heads, S] or [B, S].

        Returns
        -------
        deflected : torch.Tensor
            Deflected representations, shape [B*n_heads, S, d_head].
        velocity : torch.Tensor
            Raw velocity vectors, shape [B*n_heads, S, d_head].
        """
        BH, S, D = x.shape  # BH = B * n_heads

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(BH, -1)
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
            position_ids = position_ids.expand(BH, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(BH, -1)
        else:
            position_ids = position_ids.to(device=x.device)

        # --- Step 1: Batched vorticity projection ---
        # ω_t = W_ω · x_t  for all heads at once
        # vorticity_weight: [n_heads, D, D] → tile for each batch element: [BH, D, D]
        # The block lays the batch out **batch-outer** (bh = b*n_heads + h, so
        # slot bh holds head bh % n_heads); per-head params must therefore be
        # **tiled** (repeat), which yields [h0..h(n-1)] repeated B times, matching
        # that ordering.  (Using repeat_interleave here is the Issue #46 bug — it
        # produces head-outer ordering h*B+b and mis-routes weights for
        # n_heads>1, B>1, so a batched forward != running each sequence alone.)
        B = BH // self.n_heads
        w_vort = self.vorticity_weight.repeat(B, 1, 1)  # [BH, D, D]
        vorticity = torch.bmm(x, w_vort.transpose(-1, -2))  # [BH, S, D]
        # Apply rotary embeddings with per-head frequency scaling
        # Each head uses a different frequency → different heads capture
        # different frequency patterns (local vs global, like Fourier features)
        freq_scale = self.vorticity_frequency.repeat(B)  # [BH]
        vorticity = _apply_rotary_embeddings(vorticity, position_ids, frequency_scale=freq_scale)

        # --- Step 2: Batched Biot-Savart velocity ---
        # v_t = Γ · (H · ω_t)
        velocity = torch.bmm(vorticity, hamiltonian.transpose(-1, -2))  # [BH, S, D]
        gamma_expanded = self.gamma.repeat(B, 1)  # [BH, 1]
        velocity = gamma_expanded.view(BH, 1, 1) * velocity  # [BH, S, D]

        # --- Step 3: Batched gated deflection ---
        # g_t = σ(W_g · [x_t ; v_t])
        gate_input = torch.cat([x, velocity], dim=-1)  # [BH, S, 2D]
        w_gate = self.gate_weight.repeat(B, 1, 1)  # [BH, D, 2D]
        b_gate = self.gate_bias.repeat(B, 1)  # [BH, D]
        gate = torch.bmm(gate_input, w_gate.transpose(-1, -2))  # [BH, S, D]
        gate = gate + b_gate.view(BH, 1, D)
        gate = torch.sigmoid(gate)  # [BH, S, D]

        # r̃_t = x_t + g_t ⊙ v_t
        deflected_raw = x + gate * velocity  # [BH, S, D]

        # Batched output projection
        w_out = self.output_weight.repeat(B, 1, 1)  # [BH, D, D]
        deflected = torch.bmm(deflected_raw, w_out.transpose(-1, -2))  # [BH, S, D]

        # Batched LayerNorm: normalize over last dim
        mean = deflected.mean(dim=-1, keepdim=True)
        var = deflected.var(dim=-1, keepdim=True, unbiased=False)
        deflected = (deflected - mean) / torch.sqrt(var + 1e-5)
        ln_w = self.ln_weight.repeat(B, 1)  # [BH, D]
        ln_b = self.ln_bias.repeat(B, 1)  # [BH, D]
        deflected = deflected * ln_w.view(BH, 1, D) + ln_b.view(BH, 1, D)

        return deflected, velocity

    # ------------------------------------------------------------------
    # Concurrent-serving decode step: grouped GEMM over streams (Issue #43)
    # ------------------------------------------------------------------

    @staticmethod
    def _grouped_proj(u: torch.Tensor, W: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Apply a per-head weight to a single-token batch as one grouped GEMM.

        This is the concurrent-serving reformulation of ``bmm(u, W_rep^T)``.
        The per-stream decode step feeds a length-1 sequence, so the naïve
        batched path degenerates to ``B*n_heads`` independent matrix-*vector*
        products (a GEMV — memory-bandwidth bound, tensor cores idle) and, worse,
        replicates the shared per-head weight ``B`` times via
        ``repeat_interleave``.  Here we instead stack the ``B`` concurrent streams
        as the columns of a GEMM: for each head ``h`` the shared weight
        ``W[h] ∈ ℝ^{Dout×Din}`` multiplies ``X_h ∈ ℝ^{Din×B}`` (all streams at
        once) — a single compute-bound GEMM per head, weights read once.

        ``u`` is laid out **batch-outer** (``bh = b*n_heads + h``), matching the
        block's ``permute(0, 2, 1, 3)`` reshape, so head ``h`` is ``bh % n_heads``.

        Parameters
        ----------
        u : torch.Tensor
            Single-token input, shape [B*n_heads, 1, Din].
        W : torch.Tensor
            Per-head weight, shape [n_heads, Dout, Din].
        n_heads : int
            Number of heads.

        Returns
        -------
        torch.Tensor, shape [B*n_heads, 1, Dout] — equals ``bmm(u, W_rep^T)``.
        """
        BH, one, Din = u.shape
        Dout = W.shape[1]
        B = BH // n_heads
        # [BH,1,Din] -> [B, n_heads, Din] -> [n_heads, Din, B]
        U = u.view(B, n_heads, Din).permute(1, 2, 0)          # [n_heads, Din, B]
        Y = torch.bmm(W, U)                                    # [n_heads, Dout, B]  (grouped GEMM)
        # [n_heads, Dout, B] -> [B, n_heads, Dout] -> [BH, 1, Dout]
        return Y.permute(2, 0, 1).reshape(BH, one, Dout)

    def forward_served(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grouped-GEMM decode step for concurrent serving (Issue #43).

        Numerically equivalent to :meth:`forward` for a single token (``S == 1``),
        but reformulates the three **shared-weight** projections (vorticity, gate,
        output) as grouped GEMMs across the concurrent-stream axis instead of
        ``B*n_heads`` per-stream GEMVs with a ``repeat_interleave``-replicated
        weight.  The per-stream Biot-Savart read ``H · ω`` stays a batched op
        (each stream owns its Hamiltonian — inherently block-diagonal).

        Uses **batch-outer** ordering (``bh = b*n_heads + h``) throughout, so the
        output of a concurrent batch matches running each stream on its own.  (The
        legacy :meth:`forward` expands per-head params with ``repeat_interleave``,
        i.e. head-outer ordering, which silently mismatches heads for
        ``n_heads>1, B>1`` — this path is the corrected reformulation.)

        Parameters
        ----------
        x : torch.Tensor
            Single-token input, shape [B*n_heads, 1, d_head].
        hamiltonian : torch.Tensor
            Per-stream Hamiltonians, shape [B*n_heads, d_head, d_head].
        position_ids : torch.Tensor, optional
            Position ids, shape [B*n_heads, 1] (batch-outer).

        Returns
        -------
        deflected : torch.Tensor, shape [B*n_heads, 1, d_head]
        velocity : torch.Tensor, shape [B*n_heads, 1, d_head]
        """
        BH, S, D = x.shape
        assert S == 1, "forward_served handles a single decode token"
        nh = self.n_heads
        B = BH // nh

        if position_ids is None:
            position_ids = torch.zeros(BH, 1, device=x.device, dtype=torch.long)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(BH, -1)

        # --- Step 1: Vorticity projection ω = W_ω · x (grouped GEMM over streams) ---
        vorticity = self._grouped_proj(x, self.vorticity_weight, nh)  # [BH, 1, D]
        freq_scale = self.vorticity_frequency.repeat(B)               # [BH] batch-outer tile
        vorticity = _apply_rotary_embeddings(vorticity, position_ids, frequency_scale=freq_scale)

        # --- Step 2: Biot-Savart velocity v = Γ · (H · ω) (per-stream, block-diagonal) ---
        velocity = torch.bmm(vorticity, hamiltonian.transpose(-1, -2))  # [BH, 1, D]
        gamma_e = self.gamma.repeat(B, 1)                              # [BH, 1] batch-outer tile
        velocity = gamma_e.view(BH, 1, 1) * velocity

        # --- Step 3: Gated deflection g = σ(W_g · [x; v]) (grouped GEMM) ---
        gate_input = torch.cat([x, velocity], dim=-1)                 # [BH, 1, 2D]
        gate = self._grouped_proj(gate_input, self.gate_weight, nh)  # [BH, 1, D]
        b_gate = self.gate_bias.repeat(B, 1)                          # [BH, D] batch-outer tile
        gate = torch.sigmoid(gate + b_gate.view(BH, 1, D))
        deflected_raw = x + gate * velocity

        # --- Output projection (grouped GEMM) + per-head LayerNorm ---
        deflected = self._grouped_proj(deflected_raw, self.output_weight, nh)  # [BH, 1, D]
        mean = deflected.mean(dim=-1, keepdim=True)
        var = deflected.var(dim=-1, keepdim=True, unbiased=False)
        deflected = (deflected - mean) / torch.sqrt(var + 1e-5)
        ln_w = self.ln_weight.repeat(B, 1)                            # [BH, D] batch-outer tile
        ln_b = self.ln_bias.repeat(B, 1)
        deflected = deflected * ln_w.view(BH, 1, D) + ln_b.view(BH, 1, D)

        return deflected, velocity


class MultiThreadVortexInjection(nn.Module):
    """Multi-Thread Vortex Injection (Issue #22).

    Extends :class:`MultiHeadVortexInjection` to support **K independent
    threads per head**.  Each thread has its own:
      - Input projection (W_proj_k: d_head → d_head) — different view of the input
      - Hamiltonian matrix H_k — independent memory
      - Velocity computation: v_k = Γ_k · (H_k · ω_k)

    The K thread velocities are combined via a learned mixing weight
    (like multi-head attention's output projection).

    When ``n_threads=1`` this module is functionally equivalent to
    :class:`MultiHeadVortexInjection` (backward-compatible).

    Parameters
    ----------
    d_head : int
        Per-head dimensionality.
    n_heads : int
        Number of parallel heads.
    n_threads : int
        Number of independent threads per head (K).
    """

    def __init__(self, d_head: int, n_heads: int, n_threads: int = 1):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.n_threads = n_threads
        self.n_units = n_heads * n_threads  # total (head, thread) pairs

        # --- Batched vorticity projection: ω_t = W_ω · x_t ---
        # One vorticity weight per (head, thread) pair.
        # Shape: [n_units, d_head, d_head]
        self.vorticity_weight = nn.Parameter(
            torch.empty(self.n_units, d_head, d_head)
        )
        for u in range(self.n_units):
            nn.init.normal_(self.vorticity_weight[u], mean=0.0, std=1.0 / math.sqrt(d_head))

        # --- Per-unit vorticity frequency (Fourier-style specialisation) ---
        # Each (head, thread) pair gets a distinct frequency.
        # Threads within a head get geometrically increasing frequencies so
        # they specialise on different scales (local vs global patterns).
        freq_init = torch.tensor(
            [2.0 ** (h * n_threads + k) for h in range(n_heads) for k in range(n_threads)],
            dtype=torch.float32,
        )
        self.vorticity_frequency = nn.Parameter(freq_init)  # [n_units]

        # --- Per-unit coupling constant Γ ---
        self.gamma = nn.Parameter(
            torch.full((self.n_units, 1), 1.0 / math.sqrt(d_head))
        )

        # --- Batched gate projection: g_t = σ(W_g · [x_t ; v_t]) ---
        self.gate_weight = nn.Parameter(
            torch.empty(self.n_units, d_head, 2 * d_head)
        )
        self.gate_bias = nn.Parameter(torch.zeros(self.n_units, d_head))
        for u in range(self.n_units):
            nn.init.xavier_uniform_(self.gate_weight[u])

        # --- Batched output projection ---
        self.output_weight = nn.Parameter(
            torch.empty(self.n_units, d_head, d_head)
        )
        for u in range(self.n_units):
            nn.init.xavier_uniform_(self.output_weight[u])

        # --- Batched LayerNorm ---
        self.ln_weight = nn.Parameter(torch.ones(self.n_units, d_head))
        self.ln_bias = nn.Parameter(torch.zeros(self.n_units, d_head))

        # --- Per-thread input projections: W_proj_k · x_t → ω_k ---
        # Each thread projects the input differently before vorticity.
        # Shape: [n_threads, d_head, d_head]
        if n_threads > 1:
            self.thread_proj = nn.Parameter(
                torch.empty(n_threads, d_head, d_head)
            )
            for k in range(n_threads):
                nn.init.xavier_uniform_(self.thread_proj[k])
        else:
            self.thread_proj = None

        # --- Thread mixing weights: combine K thread outputs per head ---
        # mix_weight[h, k] — how much thread k contributes to head h's output.
        # Shape: [n_heads, n_threads]
        if n_threads > 1:
            self.thread_mix = nn.Parameter(
                torch.ones(n_heads, n_threads) / n_threads
            )
        else:
            self.thread_mix = None

    def forward(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vortex-velocity-deflected representations for all threads.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape [B*n_heads, S, d_head].
        hamiltonian : torch.Tensor
            Per-unit Hamiltonian matrices, shape [B*n_units, d_head, d_head]
            where n_units = n_heads * n_threads.
        position_ids : torch.Tensor, optional
            Position IDs, shape [B*n_heads, S] or [B, S].

        Returns
        -------
        deflected : torch.Tensor
            Thread-mixed deflected representations, shape [B*n_heads, S, d_head].
        velocity : torch.Tensor
            Per-thread (unmixed) velocity vectors, shape [B*n_units, S, d_head].
            These are used by the memory module to update each thread's
            Hamiltonian independently.
        """
        BH, S, D = x.shape  # BH = B * n_heads
        B = BH // self.n_heads
        BU = B * self.n_units  # B * n_heads * n_threads

        # --- Expand inputs for threads ---
        # x: [BH, S, D] → [BU, S, D] by repeating each head's input n_threads times
        if self.n_threads > 1:
            # x: [B, n_heads, S, D] → [B, n_heads, n_threads, S, D] → [BU, S, D]
            x_expanded = x.view(B, self.n_heads, S, D)
            x_expanded = x_expanded.unsqueeze(2).expand(
                B, self.n_heads, self.n_threads, S, D
            )
            x_threaded = x_expanded.reshape(BU, S, D)

            # Apply per-thread input projection
            # thread_proj: [n_threads, D, D]
            # x_threaded: [BU, S, D] where BU = B * n_heads * n_threads
            # We need to apply thread k's projection to the corresponding slice.
            # Reshape: [B, n_heads, n_threads, S, D] → apply per-thread proj
            x_for_proj = x_threaded.view(B, self.n_heads, self.n_threads, S, D)
            x_projected = torch.einsum(
                "bhksd,kde->bhkse", x_for_proj, self.thread_proj
            )  # [B, n_heads, n_threads, S, D]
            x_vort = x_projected.reshape(BU, S, D)
        else:
            x_vort = x

        # --- Position IDs ---
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(BH, -1)
        # Expand position_ids for all threads: [BU, S]
        if self.n_threads > 1:
            pos_expanded = position_ids.view(B, self.n_heads, S)
            pos_expanded = pos_expanded.unsqueeze(2).expand(
                B, self.n_heads, self.n_threads, S
            )
            pos_threaded = pos_expanded.reshape(BU, S)
        else:
            pos_threaded = position_ids

        # --- Step 1: Batched vorticity projection ---
        # Per-unit params are laid out [n_units, ...] (unit = h*n_threads + k) and
        # the batch is **batch-outer** (bu = b*n_units + u), so tile (repeat), not
        # repeat_interleave — see the Issue #46 note in MultiHeadVortexInjection.
        w_vort = self.vorticity_weight.repeat(B, 1, 1)  # [BU, D, D]
        vorticity = torch.bmm(x_vort, w_vort.transpose(-1, -2))  # [BU, S, D]
        freq_scale = self.vorticity_frequency.repeat(B)  # [BU]
        vorticity = _apply_rotary_embeddings(
            vorticity, pos_threaded, frequency_scale=freq_scale
        )

        # --- Step 2: Batched Biot-Savart velocity ---
        velocity = torch.bmm(vorticity, hamiltonian.transpose(-1, -2))  # [BU, S, D]
        gamma_expanded = self.gamma.repeat(B, 1)  # [BU, 1]
        velocity = gamma_expanded.view(BU, 1, 1) * velocity  # [BU, S, D]

        # --- Step 3: Batched gated deflection ---
        gate_input = torch.cat([x_vort, velocity], dim=-1)  # [BU, S, 2D]
        w_gate = self.gate_weight.repeat(B, 1, 1)  # [BU, D, 2D]
        b_gate = self.gate_bias.repeat(B, 1)  # [BU, D]
        gate = torch.bmm(gate_input, w_gate.transpose(-1, -2))  # [BU, S, D]
        gate = gate + b_gate.view(BU, 1, D)
        gate = torch.sigmoid(gate)

        deflected_raw = x_vort + gate * velocity  # [BU, S, D]

        w_out = self.output_weight.repeat(B, 1, 1)  # [BU, D, D]
        deflected = torch.bmm(deflected_raw, w_out.transpose(-1, -2))  # [BU, S, D]

        # Batched LayerNorm
        mean = deflected.mean(dim=-1, keepdim=True)
        var = deflected.var(dim=-1, keepdim=True, unbiased=False)
        deflected = (deflected - mean) / torch.sqrt(var + 1e-5)
        ln_w = self.ln_weight.repeat(B, 1)
        ln_b = self.ln_bias.repeat(B, 1)
        deflected = deflected * ln_w.view(BU, 1, D) + ln_b.view(BU, 1, D)

        # --- Step 4: Mix thread outputs back to per-head ---
        if self.n_threads > 1:
            # deflected: [BU, S, D] → [B, n_heads, n_threads, S, D]
            deflected_reshaped = deflected.view(B, self.n_heads, self.n_threads, S, D)

            # thread_mix: [n_heads, n_threads]
            # mixed[b, h, s, d] = Σ_k thread_mix[h, k] * deflected[b, h, k, s, d]
            deflected_mixed = torch.einsum(
                "hk,bhksd->bhsd", self.thread_mix, deflected_reshaped
            )

            # Reshape to [BH, S, D]
            deflected = deflected_mixed.reshape(BH, S, D)

            # velocity stays unmixed: [BU, S, D] (for per-thread H updates)
            velocity = velocity  # already [BU, S, D]

        return deflected, velocity