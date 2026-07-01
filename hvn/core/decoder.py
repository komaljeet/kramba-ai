"""
HVNDecoder
==========

A shallow, lightweight feed-forward decoder that probes the final Hamiltonian
state tensor and projects it back to logit space over a vocabulary.

The decoder is intentionally minimal — a single linear projection from d_model
to vocab_size — because the heavy associative lifting is done by the HVN blocks.
An optional small MLP (one hidden layer with GELU) can be enabled for slightly
richer decoding at minimal cost.
"""

import torch
import torch.nn as nn


class HVNDecoder(nn.Module):
    """Shallow feed-forward decoder: d_model → vocab_size logits.

    Parameters
    ----------
    d_model : int
        Model dimensionality (input).
    vocab_size : int
        Size of the output vocabulary.
    use_mlp : bool
        If True, adds a single hidden layer (d_model → d_model → vocab_size)
        with GELU activation for slightly richer decoding.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_mlp: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, vocab_size),
            )
        else:
            self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Parameters
        ----------
        x : torch.Tensor
            Hidden states from the HVN blocks, shape [batch, seq_len, d_model].

        Returns
        -------
        logits : torch.Tensor
            Logits over vocabulary, shape [batch, seq_len, vocab_size].
        """
        return self.decoder(x)