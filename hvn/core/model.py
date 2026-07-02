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
from .decoder import HVNDecoder


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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.n_threads = n_threads

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Stacked HVN blocks (multi-head, multi-thread)
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

        # Process prompts token-by-token (all sequences in parallel)
        for t in range(max_prompt_len):
            # Build batch: use token t from each prompt, or pad token (0) if shorter
            tok_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
            for b, prompt in enumerate(prompt_ids_list):
                if t < len(prompt):
                    tok_ids[b, 0] = prompt[t]
                # else: pad token 0 (already initialized)

            logits_step, states = self.forward_step_batched(tok_ids, states, position_idx=t)

        # Last logits for sampling: [B, 1, V] → [B, V]
        next_logits = logits_step[:, -1, :] / temperature  # [B, V]

        # Track generated tokens
        generated_list = [list(p) for p in prompt_ids_list]

        # --- Phase 2: Generate new tokens incrementally (all sequences in parallel) ---
        for step_idx in range(max_new_tokens):
            # Top-k filtering (per sequence)
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k_val, dim=-1)  # [B, k]
                threshold = values[:, -1:]  # [B, 1] — k-th largest value
                next_logits = next_logits.clone()
                next_logits[next_logits < threshold] = float("-inf")

            # Sample (batched)
            probs = torch.softmax(next_logits, dim=-1)  # [B, V]
            next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to each sequence
            for b in range(B):
                generated_list[b].append(next_ids[b, 0].item())

            # Forward step for all sequences at once: O(d²) per sequence, batched
            logits_step, states = self.forward_step_batched(
                next_ids, states, position_idx=max_prompt_len + step_idx
            )
            next_logits = logits_step[:, -1, :] / temperature  # [B, V]

        return generated_list

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

        Processes the prompt in one forward pass, then generates each new
        token with a single O(d²) forward_step — no context reprocessing.

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
        B = 1

        # --- Phase 1: Process the prompt in one forward pass ---
        # This initializes the Hamiltonian state with the prompt context.
        if len(prompt_ids) == 0:
            prompt_ids = [0]

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = self.forward(prompt_tensor)  # [1, S, V]

        # Initialize states by re-processing the prompt through forward_step
        # (more efficient: we can get states from the full forward, but for
        # simplicity we reinitialize states and process prompt token-by-token)
        states = self.init_generation_state(B, device=device, dtype=next(self.parameters()).dtype)

        for t in range(len(prompt_ids)):
            tok = torch.tensor([[prompt_ids[t]]], dtype=torch.long, device=device)
            logits_step, states = self.forward_step(tok, states, position_idx=t)

        # Last logits for sampling
        next_logits = logits_step[0, -1, :] / temperature

        generated = list(prompt_ids)

        # --- Phase 2: Generate new tokens incrementally (O(1) each) ---
        for step_idx in range(max_new_tokens):
            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k_val)
                next_logits = next_logits.clone()
                next_logits[next_logits < values[-1]] = float("-inf")

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)

            # Forward step: O(d²) — no context reprocessing!
            tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits_step, states = self.forward_step(tok, states, position_idx=len(prompt_ids) + step_idx)
            next_logits = logits_step[0, -1, :] / temperature

        return generated