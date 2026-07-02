"""
Vision: Image Encoding via Vortex-Projected Patch Embeddings
=============================================================

Treats image patches as vortex particles — the same way text tokens are treated.
Each image is split into a grid of patches, each patch is flattened and projected
into d_model dimensions, then processed sequentially through the existing HVN
blocks.

Architecture
------------
1. Patch Extraction: Split image into PxP patches (e.g. 16×16 pixels)
2. Patch Projection: Flatten each patch then linear projection to d_model (like ViT)
3. 2D Positional Encoding: Add learned 2D positional embeddings to patch vectors
4. Vortex Injection: Each patch embedding becomes a vortex particle; the
   Hamiltonian memory absorbs image structure the same way it absorbs text
5. Cross-modal fusion: Optional projection of image embeddings and text embeddings
   into a shared space before HVN processing

Key insight: HVN already processes arbitrary embedding vectors — patches are just
another embedding source.  The O(1) inference advantage applies to images too:
no quadratic attention over patches, and the Hamiltonian memory stores image-text
associations in fixed-size memory.

Issue #23 — https://gitlab.com/komalbarun/kramba-ai/-/work_items/23
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split an image into patches and project each to ``d_model`` dimensions.

    This is the ViT-style patch embedding adapted for HVN.  Instead of
    attention over patches, the resulting patch sequence is fed through
    HVN blocks where the Hamiltonian memory absorbs image structure in O(1)
    per patch.

    Parameters
    ----------
    image_size : int
        Height/width of the input image (assumed square).
    patch_size : int
        Height/width of each patch (assumed square).
    in_channels : int
        Number of input channels (3 for RGB, 1 for grayscale).
    d_model : int
        Output embedding dimensionality (must match HVN's d_model).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size  # patches per side

        # Single Conv2d projects all patches in one shot:
        # kernel=patch_size, stride=patch_size → one patch per output cell
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from an image batch.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape ``[B, C, H, W]``.

        Returns
        -------
        embeddings : torch.Tensor
            Patch embeddings, shape ``[B, num_patches, d_model]``.
        """
        B = x.shape[0]
        x = self.proj(x)                       # [B, d_model, grid, grid]
        x = x.flatten(2).transpose(1, 2)       # [B, num_patches, d_model]
        return x

    def get_patch_grid(self) -> tuple[int, int]:
        """Return the (rows, cols) grid layout of patches."""
        g = self.grid_size
        return g, g


class PositionalEmbedding2D(nn.Module):
    """Learned 2D positional embeddings for image patches.

    Each patch position ``(row, col)`` gets its own learned embedding vector
    of size ``d_model``.  This encodes spatial layout information that the
    sequential HVN processing would otherwise lose.

    Parameters
    ----------
    grid_size : int
        Number of patches per spatial side (image_size / patch_size).
    d_model : int
        Embedding dimensionality.
    """

    def __init__(self, grid_size: int, d_model: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size * grid_size, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to patch embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Patch embeddings, shape ``[B, num_patches, d_model]``.

        Returns
        -------
        torch.Tensor
            Patch embeddings + 2D positional info, same shape.
        """
        return x + self.pos_embed


class ImageEncoder(nn.Module):
    """Full image encoder: patches → projected embeddings → 2D positional encoding.

    This produces a sequence of patch embeddings that can be fed directly into
    HVN blocks as if they were token embeddings.  Each patch becomes a vortex
    particle whose vorticity interacts with the Hamiltonian memory.

    Parameters
    ----------
    image_size : int
        Height/width of input images (square).
    patch_size : int
        Height/width of each patch (square).
    in_channels : int
        Number of image channels.
    d_model : int
        Output embedding dimensionality (must match HVN's d_model).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        self.pos_embed = PositionalEmbedding2D(
            grid_size=self.patch_embed.grid_size,
            d_model=d_model,
        )
        self.num_patches = self.patch_embed.num_patches
        self.d_model = d_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch embedding sequences.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.

        Returns
        -------
        embeddings : torch.Tensor
            Patch embeddings with 2D positional info,
            shape ``[B, num_patches, d_model]``.
        """
        x = self.patch_embed(images)   # [B, num_patches, d_model]
        x = self.pos_embed(x)          # add 2D positional encoding
        return x


class CrossModalProjector(nn.Module):
    """Project image and text embeddings into a shared space before HVN processing.

    Optionally applies a linear projection + LayerNorm to align image patch
    embeddings with text token embeddings.  This is useful when the image and
    text modalities have different distributional properties.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings into the shared cross-modal space.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings (image or text), shape ``[B, S, D]``.

        Returns
        -------
        torch.Tensor
            Projected embeddings, same shape.
        """
        return self.dropout(self.norm(self.proj(x)))


class MultimodalHVN(nn.Module):
    """Multimodal HVN: processes images + text through shared HVN blocks.

    The image is encoded into patch embeddings (vortex particles), optionally
    concatenated with text token embeddings, and the combined sequence is
    processed through the existing HVN blocks.  The Hamiltonian memory absorbs
    both image structure and text context in fixed-size O(1) memory.

    Two modes are supported:

    1. **Image-only**: ``forward(images)`` → logits over image patches
    2. **Image + Text**: ``forward(images, text_ids)`` → logits over the
       concatenated image-patch + text-token sequence

    Parameters
    ----------
    vision_encoder : ImageEncoder
        Patch embedding + 2D positional encoding module.
    hvn : HamiltonianVortexNetwork
        The core HVN model (embedding + blocks + decoder).
    use_cross_modal_proj : bool
        If True, project image embeddings through CrossModalProjector
        before concatenation with text.
    dropout : float
        Dropout for the cross-modal projector.
    """

    def __init__(
        self,
        vision_encoder: ImageEncoder,
        hvn: "HamiltonianVortexNetwork",
        use_cross_modal_proj: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.hvn = hvn
        self.d_model = hvn.d_model
        self.use_cross_modal_proj = use_cross_modal_proj

        if use_cross_modal_proj:
            self.image_proj = CrossModalProjector(self.d_model, dropout=dropout)

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: images (optionally + text) → logits.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.
        text_ids : torch.Tensor, optional
            Text token IDs, shape ``[B, S_text]``.  If provided, text token
            embeddings are appended after the image patch embeddings.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits for the combined sequence,
            shape ``[B, num_patches + S_text, vocab_size]``.
        """
        # Encode images → patch embeddings [B, num_patches, D]
        img_embeds = self.vision_encoder(images)

        # Optionally project image embeddings into shared space
        if self.use_cross_modal_proj:
            img_embeds = self.image_proj(img_embeds)

        if text_ids is not None:
            # Embed text tokens [B, S_text, D]
            text_embeds = self.hvn.token_embedding(text_ids)
            # Concatenate: [image patches ... text tokens]
            x = torch.cat([img_embeds, text_embeds], dim=1)  # [B, num_patches + S_text, D]
        else:
            x = img_embeds  # [B, num_patches, D]

        # Generate position IDs for the combined sequence
        B, S, _ = x.shape
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        # Process through HVN blocks (same blocks for both modalities)
        for block in self.hvn.blocks:
            x = block(x, position_ids=position_ids)

        x = self.hvn.final_norm(x)
        logits = self.hvn.decoder(x)  # [B, S, V]

        return logits

    @torch.no_grad()
    def forward_step_image(
        self,
        images: torch.Tensor,
        states: list,
    ) -> tuple[torch.Tensor, list]:
        """Process a batch of images through HVN incrementally (one patch at a time).

        This initialises the Hamiltonian state with image patches, then returns
        the state for subsequent text generation.  Each patch is processed in
        O(d²) — constant in the number of patches.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.
        states : list
            Hamiltonian states (one per block) from ``init_generation_state``.

        Returns
        -------
        logits : torch.Tensor
            Logits after the last image patch, shape ``[B, 1, V]``.
        new_states : list
            Updated Hamiltonian states.
        """
        img_embeds = self.vision_encoder(images)  # [B, num_patches, D]
        if self.use_cross_modal_proj:
            img_embeds = self.image_proj(img_embeds)

        B, num_patches, D = img_embeds.shape
        new_states = states

        logits = None
        for t in range(num_patches):
            patch_t = img_embeds[:, t:t + 1, :]  # [B, 1, D]
            new_states = []
            for i, block in enumerate(self.hvn.blocks):
                patch_t, s = block.forward_step(patch_t, states[i], position_idx=t)
                new_states.append(s)
            states = new_states
            logits = self.hvn.decoder(self.hvn.final_norm(patch_t))  # [B, 1, V]

        return logits, new_states

    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> list[int]:
        """Image captioning: encode image → generate text tokens.

        The image patches are processed through HVN to initialise the Hamiltonian
        state, then text tokens are generated autoregressively using the
        incremental O(1) forward path.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]`` (B=1 for captioning).
        max_new_tokens : int
            Maximum text tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device, optional
            Device to run on.

        Returns
        -------
        generated : list[int]
            Generated token IDs.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        images = images.to(device)
        B = images.shape[0]

        # Initialise Hamiltonian states
        param_dtype = next(self.parameters()).dtype
        states = self.hvn.init_generation_state(B, device=device, dtype=param_dtype)

        # Process image patches through HVN (initialise H with image structure)
        logits, states = self.forward_step_image(images, states)

        # Start generating text tokens
        next_logits = logits[:, -1, :] / temperature  # [B, V]
        generated: list[int] = []

        num_patches = self.vision_encoder.num_patches

        for step_idx in range(max_new_tokens):
            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits[0], top_k_val)
                next_logits = next_logits.clone()
                next_logits[next_logits < values[-1]] = float("-inf")

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
            generated.append(next_id)

            # Forward step: O(d²) — no context reprocessing
            tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits_step, states = self.hvn.forward_step(
                tok, states, position_idx=num_patches + step_idx,
            )
            next_logits = logits_step[:, -1, :] / temperature

        return generated


__all__ = [
    "PatchEmbedding",
    "PositionalEmbedding2D",
    "ImageEncoder",
    "CrossModalProjector",
    "MultimodalHVN",
]