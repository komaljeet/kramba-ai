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
from .vision import ImageEncoder, CrossModalProjector, MultimodalFusionHVN


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
        # Vision / multimodal (Issue #23)
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cross_modal_proj: bool = True,
        # Multimodal fusion (Issue #27)
        use_gated_fusion: bool = True,
        use_modality_embed: bool = True,
        use_retrieval_head: bool = True,
        retrieval_dim: int | None = None,
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
        # Vision / multimodal
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.use_cross_modal_proj = use_cross_modal_proj
        # Multimodal fusion (Issue #27)
        self.use_gated_fusion = use_gated_fusion
        self.use_modality_embed = use_modality_embed
        self.use_retrieval_head = use_retrieval_head
        self.retrieval_dim = retrieval_dim

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
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "use_cross_modal_proj": self.use_cross_modal_proj,
            "use_gated_fusion": self.use_gated_fusion,
            "use_modality_embed": self.use_modality_embed,
            "use_retrieval_head": self.use_retrieval_head,
            "retrieval_dim": self.retrieval_dim,
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

        # Vision encoder for multimodal support (Issue #23)
        self.vision_encoder = ImageEncoder(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            d_model=self.config.d_model,
        )
        if self.config.use_cross_modal_proj:
            self.image_proj = CrossModalProjector(
                self.config.d_model, dropout=self.config.dropout,
            )
        else:
            self.image_proj = None

        # Multimodal fusion components (Issue #27)
        self.multimodal_fusion: MultimodalFusionHVN | None = None
        if self.config.use_gated_fusion or self.config.use_modality_embed or self.config.use_retrieval_head:
            self.multimodal_fusion = MultimodalFusionHVN(
                vision_encoder=self.vision_encoder,
                hvn=self.model,
                use_cross_modal_proj=self.config.use_cross_modal_proj,
                use_gated_fusion=self.config.use_gated_fusion,
                use_modality_embed=self.config.use_modality_embed,
                use_retrieval_head=self.config.use_retrieval_head,
                retrieval_dim=self.config.retrieval_dim,
                dropout=self.config.dropout,
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

    def forward_multimodal(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | SimpleNamespace:
        """Multimodal forward: images (+ optional text) → logits.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape [B, C, H, W].
        text_ids : torch.Tensor, optional
            Text token IDs, shape [B, S_text].
        labels : torch.Tensor, optional
            Target token IDs for loss computation.
        return_dict : bool
            If True, return a SimpleNamespace with loss and logits.
        """
        img_embeds = self.vision_encoder(images)
        if self.image_proj is not None:
            img_embeds = self.image_proj(img_embeds)
        logits = self.model.forward_multimodal(img_embeds, text_ids)
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


    def generate_multimodal(
        self,
        images: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 10,
        **_: Any,
    ) -> list[int]:
        """Generate a caption for the given image (Issue #27).

        Uses the multimodal fusion model if available, otherwise falls back
        to the basic vision encoder + HVN.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]`` (B=1 for single caption).
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling.

        Returns
        -------
        generated : list[int]
            Generated token IDs.
        """
        device = next(self.parameters()).device
        if self.multimodal_fusion is not None:
            return self.multimodal_fusion.generate_caption(
                images, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, device=device,
            )
        # Fallback: use basic MultimodalHVN path
        from .vision import MultimodalHVN
        basic = MultimodalHVN(
            vision_encoder=self.vision_encoder,
            hvn=self.model,
            use_cross_modal_proj=self.config.use_cross_modal_proj,
        ).to(device)
        return basic.generate_caption(
            images, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, device=device,
        )

    def answer_question_multimodal(
        self,
        images: torch.Tensor,
        question_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 10,
        **_: Any,
    ) -> list[int]:
        """Visual question answering: image + question → answer (Issue #27).

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[1, C, H, W]``.
        question_ids : list[int]
            Question token IDs.
        max_new_tokens : int
            Maximum answer tokens.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling.

        Returns
        -------
        answer_ids : list[int]
            Generated answer token IDs.
        """
        device = next(self.parameters()).device
        if self.multimodal_fusion is not None:
            return self.multimodal_fusion.answer_question(
                images, question_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, device=device,
            )
        raise NotImplementedError("VQA requires multimodal fusion model (use_gated_fusion=True)")


__all__ = ["HVNConfig", "HVNForCausalLM"]
