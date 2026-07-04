"""
HamiltonianVortexNetwork — Full Model
======================================

Assembles the complete HVN language model:
    1. Token embedding layer
    2. N × HVNBlock (stacked)
    3. Final layer norm
    4. HVNDecoder → vocabulary logits

The model accepts input token IDs [batch, seq_len] and produces logits
[batch, seq_len, vocab_size].
"""

import torch
import torch.nn as nn
from .hvn_block import HVNBlock
from .deltanet import DeltaNetBlock
from .weightless import WeightlessDeltaNetBlock
from .decoder import HVNDecoder


class CUDAGraphRunner:
    """CUDA Graph capture/replay helper for incremental generation (Issue #33 Fix 6).

    Captures the per-token forward_step computation graph once, then replays
    it for each generated token — eliminating ~800 kernel launches per token
    and replacing them with a single graph launch.

    CUDA Graphs require:
      - Fixed tensor shapes (same d_model, n_heads, n_layers every step)
      - Static input/output tensors (data is swapped in-place via ``copy_``)
      - No dynamic control flow during the captured region

    Usage::

        runner = CUDAGraphRunner(model, batch_size=1, device="cuda")
        runner.capture()
        for token in tokens:
            runner.input_ids.copy_(token_tensor)
            logits = runner.replay()
    """

    def __init__(self, model: "HamiltonianVortexNetwork", batch_size: int = 1,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.model = model
        self.batch_size = batch_size
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # Static input/output tensors (reused across replays)
        self.input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        self.position_idx = 0  # tracked externally

        # Pre-allocate Hamiltonian states
        self.states = model.init_generation_state(batch_size, device=self.device, dtype=self.dtype)

        # Graph + replay buffers (populated by capture())
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        self._static_states: list | None = None
        self._captured = False

    def capture(self, position_idx: int = 0):
        """Capture the forward_step computation graph.

        Must be called on CUDA device with the model in eval mode.
        """
        assert self.device.type == "cuda", "CUDA Graphs require a CUDA device"
        self.model.eval()

        # --- Warmup (3 iterations to stabilise CUDA allocator) ---
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.input_ids.fill_(0)
                logits, new_states = self.model.forward_step(
                    self.input_ids, self.states, position_idx=position_idx
                )
        torch.cuda.current_stream().wait_stream(s)

        # --- Capture ---
        self._graph = torch.cuda.CUDAGraph()

        # Need static output tensors that persist across replays.
        # We capture into pre-allocated buffers.
        with torch.cuda.graph(self._graph):
            self._static_logits, self._static_states = self.model.forward_step(
                self.input_ids, self.states, position_idx=position_idx
            )

        self._captured = True

    def replay(self) -> torch.Tensor:
        """Replay the captured graph. Returns logits [B, 1, V].

        The caller must update ``self.input_ids`` via ``copy_`` before calling.
        """
        assert self._captured, "Must call capture() before replay()"
        self._graph.replay()
        # Update states to point to the static output states
        self.states = self._static_states
        return self._static_logits


class HamiltonianVortexNetwork(nn.Module):
    """Full Hamiltonian Vortex Network language model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    d_model : int
        Model dimensionality (embedding / hidden size).
    num_layers : int
        Number of stacked HVN blocks.
    bond_dimension : int, optional
        SVD truncation rank for quantum renormalization in each block.
        None = full rank (no truncation).
    dropout : float
        Dropout rate.
    use_mlp_decoder : bool
        If True, the decoder uses a small MLP instead of a single linear layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        n_heads: int = 4,
        bond_dimension: int | None = None,
        chunk_size: int | None = 64,
        dropout: float = 0.1,
        use_mlp_decoder: bool = False,
        rank_k: int = 1,
        use_nonlinear_update: bool = False,
        use_causal_reinjection: bool = False,
        use_cross_head_mixing: bool = False,
        use_swiglu: bool = False,
        use_per_token_eta: bool = False,
        n_threads: int = 1,
        use_parallel_scan: bool = False,
        use_deltanet: bool = False,
        deltanet_conv_kernel: int = 4,
        weightless: bool = False,
        weightless_seed: int = 1234,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.n_threads = n_threads
        self.use_parallel_scan = use_parallel_scan
        self.use_deltanet = use_deltanet
        # Issue #40: hybrid weightless variant — seed-regenerated (frozen) q/k/v/
        # out projections in the DeltaNet block; only meaningful with DeltaNet.
        self.weightless = weightless and use_deltanet
        self.weightless_seed = weightless_seed

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Stacked blocks.  When ``use_deltanet`` is set, use the Gated-DeltaNet
        # dynamics block (Issue #34) which fixes the readout/update dynamics;
        # otherwise use the original HVNBlock (default — backward compatible).
        if use_deltanet:
            # Issue #40: when weightless, the q/k/v/out projections are frozen
            # seed-regenerated random features (WeightlessDeltaNetBlock); the
            # data-dependent gates/conv stay learned.  Otherwise the standard
            # fully-trained DeltaNetBlock (#34).
            block_cls = WeightlessDeltaNetBlock if self.weightless else DeltaNetBlock
            self.blocks = nn.ModuleList([
                block_cls(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    conv_kernel=deltanet_conv_kernel,
                    use_swiglu=use_swiglu,
                    bond_dimension=bond_dimension,
                    chunk_size=chunk_size,
                    **({"seed": weightless_seed, "layer_idx": i} if self.weightless else {}),
                )
                for i in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                HVNBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    bond_dimension=bond_dimension,
                    chunk_size=chunk_size,
                    dropout=dropout,
                    rank_k=rank_k,
                    use_nonlinear_update=use_nonlinear_update,
                    use_causal_reinjection=use_causal_reinjection,
                    use_cross_head_mixing=use_cross_head_mixing,
                    use_swiglu=use_swiglu,
                    use_per_token_eta=use_per_token_eta,
                    n_threads=n_threads,
                    use_parallel_scan=use_parallel_scan,
                )
                for _ in range(num_layers)
            ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder = HVNDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            use_mlp=use_mlp_decoder,
            dropout=dropout,
        )

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass: token IDs → logits.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape [batch, seq_len], dtype long.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits, shape [batch, seq_len, vocab_size].
        """
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Embed tokens
        x = self.token_embedding(input_ids)  # [B, S, D]

        # Pass through HVN blocks
        for block in self.blocks:
            x = block(x, position_ids=position_ids)  # [B, S, D]

        # Final normalisation
        x = self.final_norm(x)  # [B, S, D]

        # Decode to logits
        logits = self.decoder(x)  # [B, S, V]

        return logits

    def forward_from_embeddings(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass from pre-computed embeddings (for the demo script).

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape [batch, seq_len, d_model].

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits, shape [batch, seq_len, vocab_size].
        """
        B, S, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        for block in self.blocks:
            x = block(x, position_ids=position_ids)
        x = self.final_norm(x)
        logits = self.decoder(x)
        return logits

    def forward_parallel(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        num_chunks: int = 4,
        correction_pass: bool = True,
    ) -> torch.Tensor:
        """Multi-chunk parallel forward pass (Issue #33 Fix 4).

        Uses ``HVNBlock.parallel_forward()`` to process chunks in parallel
        with barrier synchronization.  This provides K× training speedup
        where K = num_chunks.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs, shape [batch, seq_len].
        num_chunks : int
            Number of parallel chunks per block.
        correction_pass : bool
            If True, re-run vortex injection with merged H for accuracy.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits, shape [batch, seq_len, vocab_size].
        """
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block.parallel_forward(x, position_ids=position_ids,
                                       num_chunks=num_chunks, correction_pass=correction_pass)
        x = self.final_norm(x)
        logits = self.decoder(x)
        return logits

    def forward_multimodal(
        self,
        image_embeds: torch.Tensor,
        text_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for multimodal input: image patch embeddings + optional text.

        Image patch embeddings (from the vision encoder) are treated as vortex
        particles and processed through the same HVN blocks as text tokens.
        If text IDs are provided, text token embeddings are concatenated after
        the image patch embeddings, forming a combined sequence.

        Parameters
        ----------
        image_embeds : torch.Tensor
            Image patch embeddings, shape [batch, num_patches, d_model].
            Produced by ``ImageEncoder`` in ``hvn/vision.py``.
        text_ids : torch.Tensor, optional
            Text token IDs, shape [batch, seq_len_text], dtype long.
            If provided, text embeddings are appended after image patches.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits for the combined sequence,
            shape [batch, num_patches + seq_len_text, vocab_size].
        """
        if text_ids is not None:
            text_embeds = self.token_embedding(text_ids)  # [B, S_text, D]
            x = torch.cat([image_embeds, text_embeds], dim=1)  # [B, num_patches + S_text, D]
        else:
            x = image_embeds  # [B, num_patches, D]

        B, S, _ = x.shape
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        for block in self.blocks:
            x = block(x, position_ids=position_ids)

        x = self.final_norm(x)
        logits = self.decoder(x)
        return logits

    # ------------------------------------------------------------------
    # Incremental generation: O(1) per token via Hamiltonian state cache
    # ------------------------------------------------------------------

    def init_generation_state(
        self, batch_size: int, device=None, dtype=None
    ) -> list[list[torch.Tensor]]:
        """Initialize Hamiltonian states for all blocks (for incremental generation).

        Returns
        -------
        states : list[list[torch.Tensor]]
            One state per HVN block.  Each state is [H] where H is [B, D, D].
        """
        return [
            block.init_state(batch_size, device=device, dtype=dtype)
            for block in self.blocks
        ]

    def forward_step(
        self,
        input_ids: torch.Tensor,
        states: list[list[torch.Tensor]],
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """Process a single token incrementally — O(1) per token.

        This is the optimized inference path: instead of reprocessing the
        entire sequence, we maintain the Hamiltonian state and only feed
        the new token through.  Each block updates its own Hamiltonian.

        Parameters
        ----------
        input_ids : torch.Tensor
            Single token ID, shape [batch, 1], dtype long.
        states : list[list[torch.Tensor]]
            Hamiltonian states from previous tokens (one per block).

        Returns
        -------
        logits : torch.Tensor
            Logits for this token, shape [batch, 1, vocab_size].
        new_states : list[list[torch.Tensor]]
            Updated Hamiltonian states.
        """
        x = self.token_embedding(input_ids)  # [B, 1, D]

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block.forward_step(x, states[i], position_idx=position_idx)
            new_states.append(new_state)

        x = self.final_norm(x)
        logits = self.decoder(x)  # [B, 1, V]

        return logits, new_states

    # ------------------------------------------------------------------
    # Batched generation: multiple sequences with independent H states
    # ------------------------------------------------------------------

    def init_generation_state_batched(
        self, batch_size: int, device=None, dtype=None
    ) -> list[torch.Tensor]:
        """Initialize batched Hamiltonian states for all blocks.

        Each block returns a single tensor [B*n_heads, d_head, d_head]
        instead of a list of per-head states.  This is the efficient path
        for multi-sequence generation.

        Returns
        -------
        states : list[torch.Tensor]
            One batched H tensor per HVN block, each [B*n_heads, d_head, d_head].
        """
        return [
            block.init_state_batched(batch_size, device=device, dtype=dtype)
            for block in self.blocks
        ]

    def forward_step_batched(
        self,
        input_ids: torch.Tensor,
        states: list[torch.Tensor],
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process a single token for ALL sequences in one batched call.

        This is the batched multi-sequence inference path.  Multiple sequences
        (each with its own independent Hamiltonian state) are processed
        simultaneously, maximizing GPU utilization.

        Parameters
        ----------
        input_ids : torch.Tensor
            Single token IDs for all sequences, shape [B, 1], dtype long.
        states : list[torch.Tensor]
            Batched Hamiltonian states from previous tokens (one per block).
            Each tensor: [B*n_heads, d_head, d_head].
        position_idx : int, optional
            Position index for this token (for RoPE).

        Returns
        -------
        logits : torch.Tensor
            Logits for this token, shape [B, 1, vocab_size].
        new_states : list[torch.Tensor]
            Updated batched Hamiltonian states.
        """
        x = self.token_embedding(input_ids)  # [B, 1, D]

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block.forward_step_batched(x, states[i], position_idx=position_idx)
            new_states.append(new_state)

        x = self.final_norm(x)
        logits = self.decoder(x)  # [B, 1, V]

        return logits, new_states

    def supports_served(self) -> bool:
        """Whether all blocks expose the concurrent-serving decode path (#43)."""
        return not self.use_deltanet and all(
            hasattr(b, "supports_served") and b.supports_served() for b in self.blocks
        )

    def _served_can_fuse(self, device: torch.device) -> bool:
        """Whether the fused Triton serving backend is usable for this model.

        The fused single-launch step (Issue #33 ``FusedHVNStep``) needs a CUDA
        device, a working Triton, and ``d_head >= 16`` (the ``tl.dot`` tile
        floor).  When unavailable we fall back to the portable torch grouped-GEMM
        step, which is numerically tighter but launch-bound.
        """
        if device.type != "cuda":
            return False
        try:
            import triton  # noqa: F401
        except Exception:
            return False
        return all(getattr(b, "d_head", 0) >= 16 for b in self.blocks)

    def forward_step_served(
        self,
        input_ids: torch.Tensor,
        states: list[torch.Tensor],
        position_idx: int | None = None,
        use_fused: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Single-token step for N concurrent streams (Issue #43).

        Throughput-optimized batch-outer concurrent decode for the default HVN
        block, in two interchangeable backends (both keep every stream's output
        identical to running it alone).  Since Issue #46 fixed the per-head
        routing in the shared modules, :meth:`forward_step_batched` is *also*
        correct now (batch-outer tiling instead of head-outer
        ``repeat_interleave``); this path additionally avoids the per-stream weight
        replication for speed:

        * ``use_fused=True`` — the Issue #33 ``FusedHVNStep`` Triton kernel, one
          program per ``(stream, head)``, collapsing ~25 per-token PyTorch ops
          into 2 launches.  Decode is **launch-bound** at serving batch sizes
          (per-step wall-time is nearly flat in the stream count), so cutting
          launches is the real throughput lever — ~3× sustained tok/s.  Matches
          single-stream to the fused kernel's ~1e-2 numerics (same path as
          ``generate_incremental_fused``).
        * ``use_fused=False`` — a portable torch step that reformulates the three
          shared-weight projections as one grouped GEMM per head across the
          concurrent streams (no per-stream weight replication).  Matches
          single-stream to ~1e-6.  Used when Triton is unavailable or
          ``d_head < 16``.
        """
        x = self.token_embedding(input_ids)  # [B, 1, D]
        new_states = []
        for i, block in enumerate(self.blocks):
            if use_fused:
                x, new_state = block.forward_step_fused(x, states[i], position_idx=position_idx)
            else:
                x, new_state = block.forward_step_served(x, states[i], position_idx=position_idx)
            new_states.append(new_state)
        x = self.final_norm(x)
        logits = self.decoder(x)  # [B, 1, V]
        return logits, new_states

    @torch.no_grad()
    def generate_served(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device = None,
        backend: str = "auto",
    ) -> list[list[int]]:
        """Concurrent-serving batched generation (Issue #43).

        Same interface/semantics as :meth:`generate_batched`, but drives the
        correct concurrent decode step (:meth:`forward_step_served`) so N streams
        share the model efficiently while each stream's output stays identical to
        single-stream generation.  ``backend``: ``"fused"`` forces the Triton
        single-launch step (~3× tok/s, the serving win), ``"torch"`` forces the
        portable grouped-GEMM step, ``"auto"`` (default) uses fused when available
        (CUDA + Triton + ``d_head>=16``) else torch.  Falls back to
        :meth:`generate_batched` for blocks without the served path (DeltaNet /
        multi-thread memory).
        """
        if not self.supports_served():
            return self.generate_batched(
                prompt_ids_list, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, device=device,
            )

        if device is None:
            device = next(self.parameters()).device

        if backend == "fused":
            use_fused = True
        elif backend == "torch":
            use_fused = False
        else:
            use_fused = self._served_can_fuse(device)

        self.eval()
        B = len(prompt_ids_list)
        if B == 0:
            return []

        prompt_ids_list = [p if len(p) > 0 else [0] for p in prompt_ids_list]
        max_prompt_len = max(len(p) for p in prompt_ids_list)

        param_dtype = next(self.parameters()).dtype
        states = self.init_generation_state_batched(B, device=device, dtype=param_dtype)

        prompt_matrix = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        for b, prompt in enumerate(prompt_ids_list):
            prompt_matrix[b, : len(prompt)] = torch.tensor(
                prompt, dtype=torch.long, device=device
            )

        # Phase 1: prefill prompts (all streams in lock-step)
        for t in range(max_prompt_len):
            tok_ids = prompt_matrix[:, t : t + 1]
            logits_step, states = self.forward_step_served(
                tok_ids, states, position_idx=t, use_fused=use_fused
            )

        next_logits = logits_step[:, -1, :] / temperature  # [B, V]

        # Phase 2: incremental decode (sampling stays on-device)
        step_id_tensors: list[torch.Tensor] = []
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
            step_id_tensors.append(next_ids)
            logits_step, states = self.forward_step_served(
                next_ids, states, position_idx=max_prompt_len + step_idx, use_fused=use_fused
            )
            next_logits = logits_step[:, -1, :] / temperature

        generated_list = [list(p) for p in prompt_ids_list]
        if step_id_tensors:
            new_ids = torch.cat(step_id_tensors, dim=1).tolist()
            for b in range(B):
                generated_list[b].extend(new_ids[b])
        return generated_list

    @torch.no_grad()
    def generate_batched(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device = None,
    ) -> list[list[int]]:
        """Batched autoregressive generation for multiple sequences.

        Processes multiple sequences simultaneously, each with its own
        independent Hamiltonian state.  This maximizes GPU utilization
        compared to processing sequences one at a time.

        All sequences generate the same number of tokens (max_new_tokens).
        Shorter prompts are left-padded with a pad token (ID 0); the model
        processes the full prompt for each sequence to initialize its H state.

        Parameters
        ----------
        prompt_ids_list : list[list[int]]
            List of prompt token ID lists, one per sequence.
        max_new_tokens : int
            How many new tokens to generate per sequence.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on.

        Returns
        -------
        generated_list : list[list[int]]
            Full token lists (prompt + generated), one per sequence.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        B = len(prompt_ids_list)
        if B == 0:
            return []

        # Ensure all prompts have at least 1 token
        prompt_ids_list = [p if len(p) > 0 else [0] for p in prompt_ids_list]

        # --- Phase 1: Process prompts to initialize Hamiltonian states ---
        # Each sequence processes its prompt token-by-token through forward_step_batched.
        # We pad shorter prompts to the max prompt length so all sequences stay in sync.
        max_prompt_len = max(len(p) for p in prompt_ids_list)

        # Initialize batched states
        param_dtype = next(self.parameters()).dtype
        states = self.init_generation_state_batched(B, device=device, dtype=param_dtype)

        # Build the padded prompt matrix once on-device (pad id 0), then feed one
        # column per step — avoids rebuilding + host→device copying every token.
        prompt_matrix = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        for b, prompt in enumerate(prompt_ids_list):
            prompt_matrix[b, : len(prompt)] = torch.tensor(
                prompt, dtype=torch.long, device=device
            )

        # Process prompts token-by-token (all sequences in parallel)
        for t in range(max_prompt_len):
            tok_ids = prompt_matrix[:, t : t + 1]  # [B, 1]
            logits_step, states = self.forward_step_batched(tok_ids, states, position_idx=t)

        # Last logits for sampling: [B, 1, V] → [B, V]
        next_logits = logits_step[:, -1, :] / temperature  # [B, V]

        # --- Phase 2: Generate new tokens incrementally (all sequences in parallel) ---
        # Keep sampling on-device and collect ids as tensors; sync to host once.
        step_id_tensors: list[torch.Tensor] = []  # each [B, 1]
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)  # [B, V]
            probs = torch.softmax(logits, dim=-1)  # [B, V]
            next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
            step_id_tensors.append(next_ids)

            # Forward step for all sequences at once: O(d²) per sequence, batched
            logits_step, states = self.forward_step_batched(
                next_ids, states, position_idx=max_prompt_len + step_idx
            )
            next_logits = logits_step[:, -1, :] / temperature  # [B, V]

        # Single device→host sync: [B, max_new_tokens]
        generated_list = [list(p) for p in prompt_ids_list]
        if step_id_tensors:
            new_ids = torch.cat(step_id_tensors, dim=1).tolist()  # [B][max_new_tokens]
            for b in range(B):
                generated_list[b].extend(new_ids[b])

        return generated_list

    @torch.no_grad()
    def _filter_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering on the last dim, staying on-device.

        ``logits`` may be [V] (single sequence) or [B, V] (batched).  Returns a
        new tensor with sub-threshold logits set to -inf (no in-place mutation
        of the caller's tensor).
        """
        if not top_k or top_k <= 0:
            return logits
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        threshold = values[..., -1:].contiguous()
        return logits.masked_fill(logits < threshold, float("-inf"))

    @torch.no_grad()
    def prefill_states(
        self,
        prompt_tensor: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list]:
        """Prefill the recurrent state for a whole prompt, then seed sampling.

        Replaces the old prompt handling in incremental generation (Issue #36),
        which ran a full ``forward()`` whose result was thrown away and *then*
        re-processed the prompt token-by-token — double work.  Here the prompt is
        processed exactly once, and the returned state continues generation
        identically to the per-token recurrence used by ``forward_step``.

        Two exact paths, selected by block type:

        * **DeltaNet** — the chunked-parallel delta rule is algebraically exact
          (Issue #34, max|diff| ~5e-7 vs the sequential scan), so the whole
          prompt is prefilled in a single chunked pass (``block.forward`` with
          ``return_state=True``): O(chunks) kernel launches, and the captured
          recurrent state matches the token-by-token result.
        * **Default HVN** — the chunked forward is only a *within-chunk
          approximation* of the sequential Hamiltonian recurrence that
          ``forward_step`` uses at decode time, so prefilling through it would
          change generated output.  The prompt is instead absorbed token-by-token
          with the exact recurrence.  This is still a single pass — the redundant
          full ``forward()`` the old code discarded is gone.

        Parameters
        ----------
        prompt_tensor : torch.Tensor
            Prompt token IDs, shape [B, S], dtype long.
        position_offset : int
            Position index of the prompt's first token (for RoPE / positions).

        Returns
        -------
        last_logits : torch.Tensor
            Logits for the final prompt token, shape [B, 1, V].
        states : list
            Per-block recurrent states, ready for ``forward_step_batched``.
        """
        B, S = prompt_tensor.shape
        device = prompt_tensor.device
        dtype = next(self.parameters()).dtype

        if self.use_deltanet:
            position_ids = torch.arange(
                position_offset, position_offset + S, device=device
            ).unsqueeze(0).expand(B, -1)
            x = self.token_embedding(prompt_tensor)  # [B, S, D]
            states = []
            for block in self.blocks:
                x, state = block(x, position_ids=position_ids, return_state=True)
                states.append(state)
            # Only the last position is needed to seed sampling.
            last_logits = self.decoder(self.final_norm(x[:, -1:, :]))  # [B, 1, V]
            return last_logits, states

        # Default HVN path: exact sequential prefill (single pass).
        states = self.init_generation_state_batched(B, device=device, dtype=dtype)
        last_logits = None
        for t in range(S):
            last_logits, states = self.forward_step_batched(
                prompt_tensor[:, t : t + 1], states, position_idx=position_offset + t
            )
        return last_logits, states

    @torch.no_grad()
    def generate_incremental(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device = None,
    ) -> list[int]:
        """Fast autoregressive generation using O(1) incremental inference.

        Prefills the prompt in a single chunked/batched pass (``prefill_states``)
        — the same code path as training ``forward`` — then generates each new
        token with a single O(d²) ``forward_step_batched``.  No context
        reprocessing, no redundant prompt forward, and no per-token host↔device
        syncs in the decode loop (Issue #36).

        Parameters
        ----------
        prompt_ids : list[int]
            Token IDs for the prompt.
        max_new_tokens : int
            How many new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on.

        Returns
        -------
        generated : list[int]
            Full token list (prompt + generated).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        if len(prompt_ids) == 0:
            prompt_ids = [0]
        prompt_len = len(prompt_ids)

        # --- Phase 1: Prefill the prompt in one batched/chunked pass ---
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        last_logits, states = self.prefill_states(prompt_tensor)  # [1, 1, V]
        next_logits = last_logits[0, -1, :] / temperature  # [V]

        # --- Phase 2: Generate new tokens incrementally (O(1) each) ---
        # Keep sampling on-device: reuse one input buffer, collect sampled ids
        # as device tensors, and sync to host once at the end.
        cur = torch.empty(1, 1, dtype=torch.long, device=device)
        new_id_tensors: list[torch.Tensor] = []
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1], on device
            new_id_tensors.append(next_id)

            # Feed the sampled id back in-place (no per-token host→device alloc).
            cur.copy_(next_id.view(1, 1))
            logits_step, states = self.forward_step_batched(
                cur, states, position_idx=prompt_len + step_idx
            )
            next_logits = logits_step[0, -1, :] / temperature

        # Single device→host sync for the whole generation.
        new_ids = torch.cat(new_id_tensors).tolist() if new_id_tensors else []
        return list(prompt_ids) + new_ids

    # ------------------------------------------------------------------
    # CUDA Graph generation (Issue #33 Fix 6 — 2-3× speedup)
    # ------------------------------------------------------------------

    def enable_cuda_graph(self, batch_size: int = 1, device=None, dtype=None) -> CUDAGraphRunner:
        """Create a CUDA Graph runner for fast incremental generation.

        Captures the per-token ``forward_step`` computation graph so that
        each generated token requires only a single graph replay instead
        of ~800 individual kernel launches.

        Parameters
        ----------
        batch_size : int
            Batch size (number of sequences).  Must stay fixed.
        device : torch.device
            Device (must be CUDA).
        dtype : torch.dtype
            Data type for Hamiltonian states.

        Returns
        -------
        runner : CUDAGraphRunner
            Call ``runner.capture()`` then use ``runner.replay()`` per token.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        runner = CUDAGraphRunner(self, batch_size=batch_size, device=device, dtype=dtype)
        runner.capture()
        return runner

    @torch.no_grad()
    def generate_incremental_graph(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> list[int]:
        """Fast autoregressive generation using CUDA Graph replay.

        Like ``generate_incremental`` but uses CUDA Graph capture to eliminate
        kernel launch overhead — 2-3× faster for the generation phase.

        The prompt is processed normally (token-by-token via forward_step),
        then the generation phase uses a captured CUDA Graph for each new token.

        Parameters
        ----------
        prompt_ids : list[int]
            Token IDs for the prompt.
        max_new_tokens : int
            How many new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on (must be CUDA for graph capture).

        Returns
        -------
        generated : list[int]
            Full token list (prompt + generated).
        """
        if device is None:
            device = next(self.parameters()).device

        assert device.type == "cuda", "CUDA Graph generation requires a CUDA device"
        self.eval()
        B = 1

        if len(prompt_ids) == 0:
            prompt_ids = [0]

        param_dtype = next(self.parameters()).dtype

        # --- Phase 1: Process prompt token-by-token (no graph) ---
        states = self.init_generation_state(B, device=device, dtype=param_dtype)
        for t in range(len(prompt_ids)):
            tok = torch.tensor([[prompt_ids[t]]], dtype=torch.long, device=device)
            logits_step, states = self.forward_step(tok, states, position_idx=t)

        next_logits = logits_step[0, -1, :] / temperature

        # --- Phase 2: Capture graph and generate via replay ---
        # Build a runner with the current states
        runner = CUDAGraphRunner(self, batch_size=B, device=device, dtype=param_dtype)
        runner.states = states  # use the states from prompt processing
        runner.capture(position_idx=len(prompt_ids))

        # Keep sampling on-device; write the sampled id straight into the graph's
        # static input tensor; sync the generated ids to host once at the end.
        new_id_tensors: list[torch.Tensor] = []
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1], on device
            new_id_tensors.append(next_id)

            # Copy next token into the static input tensor and replay
            runner.input_ids.copy_(next_id.view(1, 1))
            logits_step = runner.replay()
            next_logits = logits_step[0, -1, :] / temperature

        new_ids = torch.cat(new_id_tensors).tolist() if new_id_tensors else []
        return list(prompt_ids) + new_ids

    # ------------------------------------------------------------------
    # Fused Triton kernel generation (Issue #33 Fix 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_incremental_fused(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> list[int]:
        """Fast autoregressive generation using fused Triton kernels.

        Uses the FusedHVNStep Triton kernel to combine vortex injection and
        Hamiltonian update into 2 kernel launches per token instead of ~25
        PyTorch ops.  This is the Fix 1 path from Issue #33.

        Parameters
        ----------
        prompt_ids : list[int]
            Token IDs for the prompt.
        max_new_tokens : int
            How many new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on (must be CUDA for Triton kernels).

        Returns
        -------
        generated : list[int]
            Full token list (prompt + generated).
        """
        if device is None:
            device = next(self.parameters()).device

        assert device.type == "cuda", "Fused Triton kernels require a CUDA device"
        self.eval()

        if len(prompt_ids) == 0:
            prompt_ids = [0]
        prompt_len = len(prompt_ids)

        # --- Phase 1: Prefill the prompt in one batched/chunked pass (Issue #36) ---
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        last_logits, states = self.prefill_states(prompt_tensor)
        next_logits = last_logits[0, -1, :] / temperature

        # --- Phase 2: Generate using fused Triton kernels ---
        # Keep sampling on-device; reuse one input buffer; sync ids once.
        cur = torch.empty(1, 1, dtype=torch.long, device=device)
        new_id_tensors: list[torch.Tensor] = []
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1], on device
            new_id_tensors.append(next_id)

            # Fused forward step: 2 Triton kernel launches per block
            cur.copy_(next_id.view(1, 1))
            x = self.token_embedding(cur)
            new_states = []
            for i, block in enumerate(self.blocks):
                x, new_state = block.forward_step_fused(x, states[i], position_idx=prompt_len + step_idx)
                new_states.append(new_state)
            states = new_states
            x = self.final_norm(x)
            logits_step = self.decoder(x)
            next_logits = logits_step[0, -1, :] / temperature

        new_ids = torch.cat(new_id_tensors).tolist() if new_id_tensors else []
        return list(prompt_ids) + new_ids

    # ------------------------------------------------------------------
    # Conversation state management (Issue #25 — Agentic Mode)
    # ------------------------------------------------------------------

    def init_conversation_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict:
        """Initialize conversation state for multi-turn chat.

        Returns a state dict containing:
          - ``hamiltonian_states``: per-block Hamiltonian states
          - ``position``: current position index (for RoPE)
          - ``history_ids``: list of all token IDs processed so far

        This enables multi-turn conversations where the Hamiltonian memory
        carries context across turns — O(1) per token, no KV cache.

        Parameters
        ----------
        device : torch.device
            Device for the Hamiltonian states.
        dtype : torch.dtype
            Data type for the Hamiltonian states.

        Returns
        -------
        conv_state : dict
            Conversation state to pass to ``chat_step``.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        return {
            "hamiltonian_states": self.init_generation_state(1, device=device, dtype=dtype),
            "position": 0,
            "history_ids": [],
        }

    @torch.no_grad()
    def chat_step(
        self,
        input_ids: list[int],
        conv_state: dict,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> tuple[list[int], dict]:
        """Process a conversation turn incrementally.

        Feeds the input tokens through the model using the existing
        Hamiltonian state (carrying context from previous turns), then
        generates a response.  The Hamiltonian state is updated in-place.

        This is the multi-turn chat equivalent of ``generate_incremental``
        — it preserves O(1) per-token inference by reusing the Hamiltonian
        state cache across conversation turns.

        Parameters
        ----------
        input_ids : list[int]
            Token IDs for this turn's user input.
        conv_state : dict
            Conversation state from ``init_conversation_state`` or previous
            ``chat_step`` call.
        max_new_tokens : int
            Maximum new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on.

        Returns
        -------
        response_ids : list[int]
            Generated response token IDs (new tokens only).
        conv_state : dict
            Updated conversation state.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        states = conv_state["hamiltonian_states"]
        pos = conv_state["position"]

        # Feed input tokens through the model (updating Hamiltonian state).
        # Multi-turn chat continues from the carried-over state, so it uses the
        # incremental forward_step path (prefill would reset the state).  Reuse
        # one input buffer instead of allocating a tensor per token.
        cur = torch.empty(1, 1, dtype=torch.long, device=device)
        if len(input_ids) > 0:
            input_row = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, T]
            for t in range(input_row.size(1)):
                logits_step, states = self.forward_step(
                    input_row[:, t : t + 1], states, position_idx=pos
                )
                pos += 1
        else:
            logits_step = None

        # Sample first response token
        next_logits = logits_step[0, -1, :] / temperature

        # Generate response tokens — keep sampling on-device, sync ids once.
        resp_id_tensors: list[torch.Tensor] = []
        for step_idx in range(max_new_tokens):
            logits = self._filter_top_k(next_logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1], on device
            resp_id_tensors.append(next_id)

            cur.copy_(next_id.view(1, 1))
            logits_step, states = self.forward_step(cur, states, position_idx=pos)
            pos += 1
            next_logits = logits_step[0, -1, :] / temperature

        response_ids = torch.cat(resp_id_tensors).tolist() if resp_id_tensors else []

        # Update conversation state
        conv_state["hamiltonian_states"] = states
        conv_state["position"] = pos
        conv_state["history_ids"].extend(input_ids)
        conv_state["history_ids"].extend(response_ids)

        return response_ids, conv_state

    # ------------------------------------------------------------------
    # Chat method (Issue #28 — Conversation Memory)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def chat(
        self,
        input_ids: list[int],
        conv_state: dict,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> tuple[list[int], dict]:
        """Alias for ``chat_step`` — process a conversation turn.

        This is a convenience method that delegates to ``chat_step``.
        It exists so users can call ``model.chat()`` directly (Issue #28).

        Parameters
        ----------
        input_ids : list[int]
            Token IDs for this turn's input.
        conv_state : dict
            Conversation state from ``init_conversation_state``.
        max_new_tokens : int
            Maximum new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on.

        Returns
        -------
        response_ids : list[int]
            Generated response token IDs (new tokens only).
        conv_state : dict
            Updated conversation state.
        """
        return self.chat_step(
            input_ids=input_ids,
            conv_state=conv_state,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

    # ------------------------------------------------------------------
    # Tool-calling generation (Issue #24)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_tools(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
        allowed_token_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> list[int]:
        """Autoregressive generation with tool-calling support.

        Extends ``generate_incremental`` with:
          - ``allowed_token_ids``: restrict sampling to a subset of tokens
            (for JSON-constrained decoding when inside [TOOL_CALL] blocks).
          - ``stop_token_ids``: stop generation when any of these tokens
            is emitted (e.g. ``[/TOOL_CALL]``).

        Both constraints are optional — if None, behaves like
        ``generate_incremental``.

        Parameters
        ----------
        prompt_ids : list[int]
            Token IDs for the prompt.
        max_new_tokens : int
            Maximum new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device
            Device to run on.
        allowed_token_ids : list[int], optional
            If provided, only these token IDs can be sampled.  Used for
            JSON-constrained decoding inside [TOOL_CALL] blocks.
        stop_token_ids : list[int], optional
            If provided, generation stops when any of these tokens is emitted.
            Used to detect [/TOOL_CALL] or [/TOOL_RESULT] end markers.

        Returns
        -------
        generated : list[int]
            Full token list (prompt + generated).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        if len(prompt_ids) == 0:
            prompt_ids = [0]
        prompt_len = len(prompt_ids)

        # Prefill the prompt in one batched/chunked pass (Issue #36).
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        last_logits, states = self.prefill_states(prompt_tensor)
        next_logits = last_logits[0, -1, :] / temperature

        generated = list(prompt_ids)
        stop_set = set(stop_token_ids) if stop_token_ids else set()

        # Build allowed mask if provided
        allowed_mask = None
        if allowed_token_ids is not None:
            allowed_mask = torch.full((self.vocab_size,), False, device=device)
            for tid in allowed_token_ids:
                if 0 <= tid < self.vocab_size:
                    allowed_mask[tid] = True

        # Reuse one input buffer.  The sampled id must be synced each step here
        # because stop / constrained-decoding decisions depend on its value.
        cur = torch.empty(1, 1, dtype=torch.long, device=device)
        for step_idx in range(max_new_tokens):
            # Apply allowed-token constraint (JSON-constrained decoding)
            if allowed_mask is not None:
                next_logits = next_logits.masked_fill(~allowed_mask, float("-inf"))

            next_logits = self._filter_top_k(next_logits, top_k)

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_id_t = torch.multinomial(probs, num_samples=1)  # [1]
            next_id = int(next_id_t.item())
            generated.append(next_id)

            # Check stop condition
            if next_id in stop_set:
                break

            # Forward step: O(d²) — no context reprocessing!
            cur.copy_(next_id_t.view(1, 1))
            logits_step, states = self.forward_step_batched(
                cur, states, position_idx=prompt_len + step_idx
            )
            next_logits = logits_step[0, -1, :] / temperature

        return generated

    # ------------------------------------------------------------------
    # LoRA / QLoRA support (Issue #29)
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
    ) -> "HamiltonianVortexNetwork":
        """Apply LoRA adapters to this model's linear layers.

        Freezes all base model weights and adds trainable low-rank adapters.
        Only LoRA parameters will have ``requires_grad=True`` after this call.

        Parameters
        ----------
        rank : int
            LoRA rank (r).  Default 8.
        alpha : float
            LoRA scaling factor.  Default 16.
        dropout : float
            LoRA dropout.  Default 0.
        target_modules : list[str], optional
            Module name patterns to target.  If None, targets all linear layers.

        Returns
        -------
        self : HamiltonianVortexNetwork
            The model with LoRA applied (in-place).
        """
        from ..lora import apply_lora as _apply_lora
        return _apply_lora(self, rank=rank, alpha=alpha, dropout=dropout,
                           target_modules=target_modules)

    def apply_qlora(
        self,
        config=None,
    ) -> "HamiltonianVortexNetwork":
        """Apply QLoRA: quantize base model to NF4 + add LoRA adapters.

        Parameters
        ----------
        config : QLoRAConfig, optional
            QLoRA configuration.  If None, uses defaults (rank=8, NF4).

        Returns
        -------
        self : HamiltonianVortexNetwork
            The model with QLoRA applied (in-place).
        """
        from ..lora import apply_qlora as _apply_qlora
        return _apply_qlora(self, config=config)

    def merge_lora(self) -> "HamiltonianVortexNetwork":
        """Merge all LoRA/QLoRA adapters back into the base model weights.

        After merging, the model has zero LoRA overhead and can be used
        for inference exactly like a standard model.

        Returns
        -------
        self : HamiltonianVortexNetwork
            The model with all adapters merged (in-place).
        """
        from ..lora import merge_lora_weights as _merge
        return _merge(self)

    def count_trainable_params(self) -> dict:
        """Count trainable vs. total parameters.

        Returns
        -------
        dict
            Dictionary with 'total', 'trainable', 'frozen', 'trainable_pct'.
        """
        from ..lora import count_lora_params
        return count_lora_params(self)

    def get_lora_parameters(self) -> list[torch.nn.Parameter]:
        """Return only the trainable (LoRA) parameters.

        Use this to configure the optimizer to only update LoRA weights.
        """
        from ..lora import get_lora_params
        return get_lora_params(self)

    def save_lora(self, path: str, extra_metadata: dict | None = None) -> None:
        """Save only the LoRA adapter weights to a checkpoint file.

        Parameters
        ----------
        path : str
            Save path for the LoRA checkpoint.
        extra_metadata : dict, optional
            Additional metadata to save (e.g. model config, training info).
        """
        from ..lora import save_lora_checkpoint
        save_lora_checkpoint(self, path, extra_metadata=extra_metadata)

    def load_lora(self, path: str, strict: bool = False) -> dict:
        """Load LoRA adapter weights from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the LoRA checkpoint.
        strict : bool
            If True, all LoRA keys must match exactly.

        Returns
        -------
        metadata : dict
            Metadata from the checkpoint (if any).
        """
        from ..lora import load_lora_checkpoint
        return load_lora_checkpoint(self, path, strict=strict)