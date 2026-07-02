"""Lightweight Hugging Face-style wrappers for the HVN model.

The repository does not require `transformers` to be installed for these helpers,
so the implementation is intentionally dependency-light while still exposing a
familiar save/load API and a `generate()` entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import HamiltonianVortexNetwork


class HVNConfig:
    """Configuration container with a JSON-serialisable shape."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
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
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.bond_dimension = bond_dimension
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.use_mlp_decoder = use_mlp_decoder
        self.rank_k = rank_k
        self.use_nonlinear_update = use_nonlinear_update
        self.use_causal_reinjection = use_causal_reinjection
        self.use_cross_head_mixing = use_cross_head_mixing
        self.use_swiglu = use_swiglu
        self.use_per_token_eta = use_per_token_eta
        self.n_threads = n_threads

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "n_heads": self.n_heads,
            "bond_dimension": self.bond_dimension,
            "chunk_size": self.chunk_size,
            "dropout": self.dropout,
            "use_mlp_decoder": self.use_mlp_decoder,
            "rank_k": self.rank_k,
            "use_nonlinear_update": self.use_nonlinear_update,
            "use_causal_reinjection": self.use_causal_reinjection,
            "use_cross_head_mixing": self.use_cross_head_mixing,
            "use_swiglu": self.use_swiglu,
            "use_per_token_eta": self.use_per_token_eta,
            "n_threads": self.n_threads,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HVNConfig":
        return cls(**data)


class HVNForCausalLM(nn.Module):
    """A Hugging Face-style wrapper around :class:`HamiltonianVortexNetwork`.

    The wrapper exposes:
    - ``save_pretrained`` / ``from_pretrained``
    - ``generate`` for simple autoregressive sampling
    - a ``forward`` method compatible with common training loops
    """

    def __init__(self, config: HVNConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = HVNConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword arguments, not both")

        self.config = config
        self.model = HamiltonianVortexNetwork(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            n_heads=self.config.n_heads,
            bond_dimension=self.config.bond_dimension,
            chunk_size=self.config.chunk_size,
            dropout=self.config.dropout,
            use_mlp_decoder=self.config.use_mlp_decoder,
            rank_k=self.config.rank_k,
            use_nonlinear_update=self.config.use_nonlinear_update,
            use_causal_reinjection=self.config.use_causal_reinjection,
            use_cross_head_mixing=self.config.use_cross_head_mixing,
            use_swiglu=self.config.use_swiglu,
            use_per_token_eta=self.config.use_per_token_eta,
            n_threads=self.config.n_threads,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | SimpleNamespace:
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if return_dict:
            return SimpleNamespace(loss=loss, logits=logits)
        if loss is not None:
            return loss, logits
        return logits

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        with open(save_path / "config.json", "w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any) -> "HVNForCausalLM":
        model_path = Path(pretrained_model_name_or_path)
        if not model_path.exists():
            raise FileNotFoundError(f"No saved HVN model found at {model_path}")

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)

        config = HVNConfig.from_dict(config_data)
        model = cls(config=config)
        state_dict_path = model_path / "pytorch_model.bin"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        return model

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 10,
        do_sample: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        """Generate tokens autoregressively using HVN's incremental path."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        prompt_ids = input_ids[0].tolist()
        generated_ids = self.model.generate_incremental(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=next(self.parameters()).device,
        )
        return torch.tensor([generated_ids], dtype=torch.long, device=next(self.parameters()).device)


__all__ = ["HVNConfig", "HVNForCausalLM"]
