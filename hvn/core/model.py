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
        chunk_size: int = 16,
        dropout: float = 0.1,
        use_mlp_decoder: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Stacked HVN blocks (multi-head)
        self.blocks = nn.ModuleList([
            HVNBlock(
                d_model=d_model,
                n_heads=n_heads,
                bond_dimension=bond_dimension,
                chunk_size=chunk_size,
                dropout=dropout,
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        # Embed tokens
        x = self.token_embedding(input_ids)  # [B, S, D]

        # Pass through HVN blocks
        for block in self.blocks:
            x = block(x)  # [B, S, D]

        # Final normalisation
        x = self.final_norm(x)  # [B, S, D]

        # Decode to logits
        logits = self.decoder(x)  # [B, S, V]

        return logits

    def forward_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
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
        for block in self.blocks:
            x = block(x)
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
            x, new_state = block.forward_step(x, states[i])
            new_states.append(new_state)

        x = self.final_norm(x)
        logits = self.decoder(x)  # [B, 1, V]

        return logits, new_states

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
            logits_step, states = self.forward_step(tok, states)

        # Last logits for sampling
        next_logits = logits_step[0, -1, :] / temperature

        generated = list(prompt_ids)

        # --- Phase 2: Generate new tokens incrementally (O(1) each) ---
        for _ in range(max_new_tokens):
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
            logits_step, states = self.forward_step(tok, states)
            next_logits = logits_step[0, -1, :] / temperature

        return generated