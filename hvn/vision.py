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
Issue #27 — https://gitlab.com/komalbarun/kramba-ai/-/work_items/27 (Multimodal Fusion)
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


# ---------------------------------------------------------------------------
# Issue #27: Multimodal Fusion — Joint Image-Text Training & Cross-Modal Retrieval
# ---------------------------------------------------------------------------


class GatedFusionLayer(nn.Module):
    """Gated cross-modal fusion layer.

    Fuses image and text representations using a learned gate that controls
    how much of each modality contributes to the fused output:

        fused = g * image_proj + (1 - g) * text_proj

    where ``g = sigmoid(W_g * [img; txt])`` is a per-position gate.

    This is a lightweight alternative to cross-attention — it preserves O(1)
    inference because the gate is computed from fixed-size projections, not
    from an N×N attention matrix.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        # Gate: takes concatenated [img; txt] → scalar gate per position
        self.gate = nn.Linear(d_model * 2, d_model)
        # Modality projections
        self.img_proj = nn.Linear(d_model, d_model)
        self.txt_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse image and text embeddings via gated combination.

        Parameters
        ----------
        image_embeds : torch.Tensor
            Image patch embeddings, shape ``[B, S_img, D]``.
        text_embeds : torch.Tensor
            Text token embeddings, shape ``[B, S_txt, D]``.

        Returns
        -------
        fused : torch.Tensor
            Fused embeddings, shape ``[B, S_img + S_txt, D]``.
            The image portion is gated-fused with a pooled text representation,
            and the text portion is gated-fused with a pooled image representation.
        """
        # Pool across the opposite modality for cross-modal context
        img_pooled = image_embeds.mean(dim=1, keepdim=True)  # [B, 1, D]
        txt_pooled = text_embeds.mean(dim=1, keepdim=True)    # [B, 1, D]

        # Gated fusion for image tokens: mix image with pooled text context
        img_gate = torch.sigmoid(
            self.gate(torch.cat([image_embeds, txt_pooled.expand(-1, image_embeds.size(1), -1)], dim=-1))
        )  # [B, S_img, D]
        fused_img = img_gate * self.img_proj(image_embeds) + (1 - img_gate) * self.txt_proj(
            txt_pooled.expand(-1, image_embeds.size(1), -1)
        )

        # Gated fusion for text tokens: mix text with pooled image context
        txt_gate = torch.sigmoid(
            self.gate(torch.cat([txt_pooled.expand(-1, text_embeds.size(1), -1), text_embeds], dim=-1))
        )  # [B, S_txt, D]
        fused_txt = txt_gate * self.txt_proj(text_embeds) + (1 - txt_gate) * self.img_proj(
            img_pooled.expand(-1, text_embeds.size(1), -1)
        )

        # Concatenate fused image + fused text
        fused = torch.cat([fused_img, fused_txt], dim=1)  # [B, S_img + S_txt, D]
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused


class ModalityEmbedding(nn.Module):
    """Learned modality type embeddings.

    Adds a learned ``[IMAGE]`` or ``[TEXT]`` embedding to each position so
    the model can distinguish which modality each token comes from.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # 2 modality types: 0=image, 1=text
        self.modality_embed = nn.Embedding(2, d_model)

    def forward(self, x: torch.Tensor, modality: int) -> torch.Tensor:
        """Add modality embedding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape ``[B, S, D]``.
        modality : int
            0 for image, 1 for text.

        Returns
        -------
        torch.Tensor
            Embeddings + modality type, same shape.
        """
        B, S, D = x.shape
        mod_ids = torch.full((B, S), modality, dtype=torch.long, device=x.device)
        return x + self.modality_embed(mod_ids)


class CrossModalRetriever(nn.Module):
    """Cross-modal retrieval via shared embedding space.

    Projects image and text representations into a shared retrieval space
    where cosine similarity measures cross-modal alignment.  Supports:
    - Text-to-image retrieval: given text query, find matching images
    - Image-to-text retrieval: given image, find matching text descriptions

    The projection is a simple linear layer followed by L2 normalization,
    preserving O(1) inference (no attention scan).

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    retrieval_dim : int
        Dimensionality of the shared retrieval space (default: d_model).
    """

    def __init__(self, d_model: int, retrieval_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.retrieval_dim = retrieval_dim or d_model
        self.img_proj = nn.Linear(d_model, self.retrieval_dim)
        self.txt_proj = nn.Linear(d_model, self.retrieval_dim)

    def encode_image(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Encode image patch embeddings into retrieval space.

        Parameters
        ----------
        image_embeds : torch.Tensor
            Image patch embeddings, shape ``[B, num_patches, D]``.

        Returns
        -------
        torch.Tensor
            Image vector in retrieval space, shape ``[B, retrieval_dim]``,
            L2-normalized.
        """
        # Pool patches → single vector
        pooled = image_embeds.mean(dim=1)  # [B, D]
        projected = self.img_proj(pooled)  # [B, retrieval_dim]
        return F.normalize(projected, dim=-1)

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Encode text token embeddings into retrieval space.

        Parameters
        ----------
        text_embeds : torch.Tensor
            Text token embeddings, shape ``[B, S, D]``.

        Returns
        -------
        torch.Tensor
            Text vector in retrieval space, shape ``[B, retrieval_dim]``,
            L2-normalized.
        """
        # Pool tokens → single vector
        pooled = text_embeds.mean(dim=1)  # [B, D]
        projected = self.txt_proj(pooled)  # [B, retrieval_dim]
        return F.normalize(projected, dim=-1)

    def retrieve(
        self,
        query_embeds: torch.Tensor,
        candidate_embeds: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k matching candidates for each query.

        Parameters
        ----------
        query_embeds : torch.Tensor
            Query vectors in retrieval space, shape ``[B_q, retrieval_dim]``.
        candidate_embeds : torch.Tensor
            Candidate vectors in retrieval space, shape ``[B_c, retrieval_dim]``.
        top_k : int
            Number of top matches to return.

        Returns
        -------
        indices : torch.Tensor
            Indices of top-k matches, shape ``[B_q, top_k]``.
        similarities : torch.Tensor
            Cosine similarities of top-k matches, shape ``[B_q, top_k]``.
        """
        # Cosine similarity (vectors are already L2-normalized)
        sim_matrix = query_embeds @ candidate_embeds.T  # [B_q, B_c]
        k = min(top_k, sim_matrix.size(-1))
        similarities, indices = torch.topk(sim_matrix, k, dim=-1)
        return indices, similarities

    def contrastive_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss for image-text pairs.

        For each image, the matching text should have higher similarity than
        all other texts (and vice versa).  This trains the shared embedding
        space for cross-modal retrieval.

        Parameters
        ----------
        image_embeds : torch.Tensor
            Image vectors in retrieval space, shape ``[B, retrieval_dim]``.
        text_embeds : torch.Tensor
            Text vectors in retrieval space, shape ``[B, retrieval_dim]``.
        temperature : float
            Temperature scaling for softmax.

        Returns
        -------
        loss : torch.Tensor
            Scalar contrastive loss.
        """
        # Both are already L2-normalized
        logits = image_embeds @ text_embeds.T / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class MultimodalFusionHVN(nn.Module):
    """Full multimodal fusion model: joint image-text training + cross-modal retrieval.

    Extends ``MultimodalHVN`` with:
    - **Gated cross-modal fusion**: a learned gate mixes image and text
      representations before HVN processing
    - **Modality embeddings**: ``[IMAGE]`` / ``[TEXT]`` type embeddings so the
      model can distinguish modalities in the interleaved sequence
    - **Cross-modal retrieval**: a ``CrossModalRetriever`` head for learning
      a shared image-text embedding space via contrastive loss
    - **VQA support**: image + question → answer generation

    All additions preserve O(1) inference — no attention scan, no KV cache.

    Parameters
    ----------
    vision_encoder : ImageEncoder
        Patch embedding + 2D positional encoding module.
    hvn : HamiltonianVortexNetwork
        The core HVN model.
    use_cross_modal_proj : bool
        If True, project image embeddings through CrossModalProjector.
    use_gated_fusion : bool
        If True, apply GatedFusionLayer to mix image and text before HVN.
    use_modality_embed : bool
        If True, add learned modality type embeddings.
    use_retrieval_head : bool
        If True, add CrossModalRetriever for contrastive retrieval training.
    retrieval_dim : int, optional
        Dimensionality of the retrieval space (default: d_model).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vision_encoder: ImageEncoder,
        hvn: "HamiltonianVortexNetwork",
        use_cross_modal_proj: bool = True,
        use_gated_fusion: bool = True,
        use_modality_embed: bool = True,
        use_retrieval_head: bool = True,
        retrieval_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.hvn = hvn
        self.d_model = hvn.d_model
        self.use_cross_modal_proj = use_cross_modal_proj
        self.use_gated_fusion = use_gated_fusion
        self.use_modality_embed = use_modality_embed
        self.use_retrieval_head = use_retrieval_head

        if use_cross_modal_proj:
            self.image_proj = CrossModalProjector(self.d_model, dropout=dropout)

        if use_gated_fusion:
            self.gated_fusion = GatedFusionLayer(self.d_model, dropout=dropout)

        if use_modality_embed:
            self.modality_embed = ModalityEmbedding(self.d_model)

        if use_retrieval_head:
            self.retriever = CrossModalRetriever(
                self.d_model, retrieval_dim=retrieval_dim,
            )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch embeddings with optional projection.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.

        Returns
        -------
        img_embeds : torch.Tensor
            Image patch embeddings, shape ``[B, num_patches, D]``.
        """
        img_embeds = self.vision_encoder(images)
        if self.use_cross_modal_proj:
            img_embeds = self.image_proj(img_embeds)
        if self.use_modality_embed:
            img_embeds = self.modality_embed(img_embeds, modality=0)
        return img_embeds

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text token IDs into embeddings with modality embedding.

        Parameters
        ----------
        text_ids : torch.Tensor
            Text token IDs, shape ``[B, S_text]``.

        Returns
        -------
        text_embeds : torch.Tensor
            Text token embeddings, shape ``[B, S_text, D]``.
        """
        text_embeds = self.hvn.token_embedding(text_ids)
        if self.use_modality_embed:
            text_embeds = self.modality_embed(text_embeds, modality=1)
        return text_embeds

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
            Text token IDs, shape ``[B, S_text]``.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits for the combined sequence,
            shape ``[B, num_patches + S_text, vocab_size]``.
        """
        img_embeds = self.encode_images(images)

        if text_ids is not None:
            text_embeds = self.encode_text(text_ids)

            if self.use_gated_fusion:
                x = self.gated_fusion(img_embeds, text_embeds)
            else:
                x = torch.cat([img_embeds, text_embeds], dim=1)
        else:
            x = img_embeds

        B, S, _ = x.shape
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        for block in self.hvn.blocks:
            x = block(x, position_ids=position_ids)

        x = self.hvn.final_norm(x)
        logits = self.hvn.decoder(x)
        return logits

    def forward_with_retrieval(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass that also computes retrieval embeddings.

        Used for joint training with language modeling + contrastive loss.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.
        text_ids : torch.Tensor
            Text token IDs, shape ``[B, S_text]``.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary logits, shape ``[B, num_patches + S_text, V]``.
        img_retrieval : torch.Tensor
            Image retrieval vector, shape ``[B, retrieval_dim]``.
        txt_retrieval : torch.Tensor
            Text retrieval vector, shape ``[B, retrieval_dim]``.
        """
        img_embeds = self.encode_images(images)
        text_embeds = self.encode_text(text_ids)

        # Compute retrieval embeddings from pre-HVN embeddings
        if self.use_retrieval_head:
            img_retrieval = self.retriever.encode_image(img_embeds)
            txt_retrieval = self.retriever.encode_text(text_embeds)
        else:
            img_retrieval = None
            txt_retrieval = None

        # Fusion + HVN processing
        if self.use_gated_fusion:
            x = self.gated_fusion(img_embeds, text_embeds)
        else:
            x = torch.cat([img_embeds, text_embeds], dim=1)

        B, S, _ = x.shape
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        for block in self.hvn.blocks:
            x = block(x, position_ids=position_ids)

        x = self.hvn.final_norm(x)
        logits = self.hvn.decoder(x)

        return logits, img_retrieval, txt_retrieval

    def compute_loss(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        retrieval_weight: float = 0.5,
        temperature: float = 0.07,
    ) -> dict[str, torch.Tensor]:
        """Joint multimodal training loss: LM + contrastive retrieval.

        Combines:
        1. **Language modeling loss** (cross-entropy on text tokens)
        2. **Contrastive retrieval loss** (InfoNCE on image-text pairs)

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.
        text_ids : torch.Tensor
            Text token IDs, shape ``[B, S_text]``.
        retrieval_weight : float
            Weight for the contrastive loss (0 = LM only, 1 = retrieval only).
        temperature : float
            Temperature for contrastive loss.

        Returns
        -------
        loss_dict : dict[str, torch.Tensor]
            Dictionary with 'total', 'lm', and 'retrieval' loss values.
        """
        logits, img_ret, txt_ret = self.forward_with_retrieval(images, text_ids)

        # LM loss: predict next text token (shift by 1, only on text portion)
        num_patches = self.vision_encoder.num_patches
        text_logits = logits[:, num_patches:, :]  # [B, S_text, V]
        # Shift: predict token t+1 from token t
        if text_logits.size(1) > 1:
            lm_logits = text_logits[:, :-1, :].contiguous()   # [B, S_text-1, V]
            lm_targets = text_ids[:, 1:].contiguous()           # [B, S_text-1]
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                lm_targets.view(-1),
            )
        else:
            lm_loss = torch.tensor(0.0, device=logits.device)

        # Retrieval loss
        if self.use_retrieval_head and img_ret is not None:
            ret_loss = self.retriever.contrastive_loss(img_ret, txt_ret, temperature)
        else:
            ret_loss = torch.tensor(0.0, device=logits.device)

        total = lm_loss + retrieval_weight * ret_loss
        return {"total": total, "lm": lm_loss, "retrieval": ret_loss}

    @torch.no_grad()
    def forward_step_image(
        self,
        images: torch.Tensor,
        states: list,
    ) -> tuple[torch.Tensor, list]:
        """Process images through HVN incrementally (one patch at a time).

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]``.
        states : list
            Hamiltonian states from ``init_generation_state``.

        Returns
        -------
        logits : torch.Tensor
            Logits after the last image patch, shape ``[B, 1, V]``.
        new_states : list
            Updated Hamiltonian states.
        """
        img_embeds = self.encode_images(images)

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
            logits = self.hvn.decoder(self.hvn.final_norm(patch_t))

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

        param_dtype = next(self.parameters()).dtype
        states = self.hvn.init_generation_state(B, device=device, dtype=param_dtype)

        logits, states = self.forward_step_image(images, states)

        next_logits = logits[:, -1, :] / temperature
        generated: list[int] = []
        num_patches = self.vision_encoder.num_patches

        for step_idx in range(max_new_tokens):
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits[0], top_k_val)
                next_logits = next_logits.clone()
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
            generated.append(next_id)

            tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits_step, states = self.hvn.forward_step(
                tok, states, position_idx=num_patches + step_idx,
            )
            next_logits = logits_step[:, -1, :] / temperature

        return generated

    @torch.no_grad()
    def answer_question(
        self,
        images: torch.Tensor,
        question_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> list[int]:
        """Visual Question Answering: image + question → answer.

        Processes the image through HVN to initialise the Hamiltonian state,
        then feeds the question tokens through the model, and generates
        an answer — all in O(1) per token.

        Parameters
        ----------
        images : torch.Tensor
            Input images, shape ``[B, C, H, W]`` (B=1 for VQA).
        question_ids : list[int]
            Token IDs for the question.
        max_new_tokens : int
            Maximum answer tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).
        device : torch.device, optional
            Device to run on.

        Returns
        -------
        answer_ids : list[int]
            Generated answer token IDs.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        images = images.to(device)
        B = images.shape[0]

        param_dtype = next(self.parameters()).dtype
        states = self.hvn.init_generation_state(B, device=device, dtype=param_dtype)

        # Phase 1: Process image patches through HVN
        logits, states = self.forward_step_image(images, states)
        num_patches = self.vision_encoder.num_patches

        # Phase 2: Feed question tokens through the model
        pos = num_patches
        for qid in question_ids:
            tok = torch.tensor([[qid]], dtype=torch.long, device=device)
            logits_step, states = self.hvn.forward_step(tok, states, position_idx=pos)
            pos += 1

        # Phase 3: Generate answer tokens
        next_logits = logits_step[:, -1, :] / temperature
        answer_ids: list[int] = []

        for step_idx in range(max_new_tokens):
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits[0], top_k_val)
                next_logits = next_logits.clone()
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
            answer_ids.append(next_id)

            tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits_step, states = self.hvn.forward_step(tok, states, position_idx=pos)
            pos += 1
            next_logits = logits_step[:, -1, :] / temperature

        return answer_ids

    @torch.no_grad()
    def retrieve_images(
        self,
        text_query_ids: list[int],
        image_candidates: torch.Tensor,
        top_k: int = 5,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-to-image retrieval: given text query, find matching images.

        Parameters
        ----------
        text_query_ids : list[int]
            Token IDs for the text query.
        image_candidates : torch.Tensor
            Candidate images, shape ``[B_c, C, H, W]``.
        top_k : int
            Number of top matches to return.
        device : torch.device, optional
            Device to run on.

        Returns
        -------
        indices : torch.Tensor
            Indices of top-k matching images, shape ``[top_k]``.
        similarities : torch.Tensor
            Cosine similarities, shape ``[top_k]``.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        if not self.use_retrieval_head:
            raise ValueError("Retrieval head not enabled — set use_retrieval_head=True")

        # Encode text query
        query_tensor = torch.tensor([text_query_ids], dtype=torch.long, device=device)
        text_embeds = self.encode_text(query_tensor)
        txt_vec = self.retriever.encode_text(text_embeds)  # [1, retrieval_dim]

        # Encode candidate images
        img_embeds = self.encode_images(image_candidates.to(device))
        img_vecs = self.retriever.encode_image(img_embeds)  # [B_c, retrieval_dim]

        # Retrieve
        indices, similarities = self.retriever.retrieve(txt_vec, img_vecs, top_k=top_k)
        return indices[0], similarities[0]

    @torch.no_grad()
    def retrieve_text(
        self,
        query_image: torch.Tensor,
        text_candidates: list[list[int]],
        top_k: int = 5,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Image-to-text retrieval: given image, find matching text descriptions.

        Parameters
        ----------
        query_image : torch.Tensor
            Query image, shape ``[1, C, H, W]``.
        text_candidates : list[list[int]]
            List of text candidate token ID lists.
        top_k : int
            Number of top matches to return.
        device : torch.device, optional
            Device to run on.

        Returns
        -------
        indices : torch.Tensor
            Indices of top-k matching texts, shape ``[top_k]``.
        similarities : torch.Tensor
            Cosine similarities, shape ``[top_k]``.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        if not self.use_retrieval_head:
            raise ValueError("Retrieval head not enabled — set use_retrieval_head=True")

        # Encode query image
        img_embeds = self.encode_images(query_image.to(device))
        img_vec = self.retriever.encode_image(img_embeds)  # [1, retrieval_dim]

        # Encode candidate texts
        txt_vecs = []
        for cand_ids in text_candidates:
            cand_tensor = torch.tensor([cand_ids], dtype=torch.long, device=device)
            text_embeds = self.encode_text(cand_tensor)
            txt_vec = self.retriever.encode_text(text_embeds)  # [1, retrieval_dim]
            txt_vecs.append(txt_vec)
        txt_vecs = torch.cat(txt_vecs, dim=0)  # [B_c, retrieval_dim]

        # Retrieve
        indices, similarities = self.retriever.retrieve(img_vec, txt_vecs, top_k=top_k)
        return indices[0], similarities[0]


# ---------------------------------------------------------------------------
# Image preprocessing utilities
# ---------------------------------------------------------------------------


class ImagePreprocessor:
    """Simple image preprocessing for HVN multimodal training.

    Provides resize, normalize, and tensor conversion utilities without
    requiring torchvision (uses F.interpolate for resizing).

    Parameters
    ----------
    image_size : int
        Target image size (square).
    in_channels : int
        Number of channels (3 for RGB, 1 for grayscale).
    normalize_mean : tuple
        Per-channel mean for normalization.
    normalize_std : tuple
        Per-channel std for normalization.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        normalize_mean: tuple[float, ...] = (0.5,),
        normalize_std: tuple[float, ...] = (0.5,),
    ) -> None:
        self.image_size = image_size
        self.in_channels = in_channels
        self.normalize_mean = torch.tensor(normalize_mean).view(1, in_channels, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(1, in_channels, 1, 1)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess a single image tensor.

        Parameters
        ----------
        image : torch.Tensor
            Input image, shape ``[C, H, W]`` or ``[H, W, C]``.

        Returns
        -------
        torch.Tensor
            Preprocessed image, shape ``[C, image_size, image_size]``.
        """
        # Handle HWC → CHW
        if image.dim() == 3 and image.size(-1) in (1, 3) and image.size(0) not in (1, 3):
            image = image.permute(2, 0, 1)  # HWC → CHW

        # Ensure correct number of channels
        if image.size(0) != self.in_channels:
            if image.size(0) == 1 and self.in_channels == 3:
                image = image.expand(3, -1, -1)
            elif image.size(0) == 3 and self.in_channels == 1:
                image = image[:1]

        # Add batch dim for interpolate
        image = image.unsqueeze(0)  # [1, C, H, W]
        image = F.interpolate(
            image, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        # Normalize
        mean = self.normalize_mean.to(image.device)
        std = self.normalize_std.to(image.device)
        image = (image - mean) / std

        return image.squeeze(0)  # [C, H, W]

    def preprocess_batch(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Preprocess a list of images into a batch.

        Parameters
        ----------
        images : list[torch.Tensor]
            List of image tensors, each ``[C, H, W]`` or ``[H, W, C]``.

        Returns
        -------
        torch.Tensor
            Batch tensor, shape ``[B, C, image_size, image_size]``.
        """
        processed = [self.preprocess(img) for img in images]
        return torch.stack(processed, dim=0)


# ---------------------------------------------------------------------------
# Synthetic image-text pair dataset generator
# ---------------------------------------------------------------------------


def generate_synthetic_image_text_pairs(
    num_pairs: int = 32,
    image_size: int = 64,
    in_channels: int = 3,
    text_vocab_size: int = 100,
    text_seq_len: int = 16,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic image-text pairs for multimodal training.

    Creates images with simple geometric patterns (circles, rectangles, gradients)
    paired with synthetic "caption" token sequences.  The patterns are correlated
    with the text so the model can learn cross-modal associations.

    Parameters
    ----------
    num_pairs : int
        Number of image-text pairs to generate.
    image_size : int
        Size of each image (square).
    in_channels : int
        Number of image channels.
    text_vocab_size : int
        Size of the synthetic text vocabulary.
    text_seq_len : int
        Length of each text sequence.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    images : torch.Tensor
        Image batch, shape ``[num_pairs, C, image_size, image_size]``.
    text_ids : torch.Tensor
        Text token IDs, shape ``[num_pairs, text_seq_len]``.
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = torch.Generator()

    images = torch.zeros(num_pairs, in_channels, image_size, image_size)
    text_ids = torch.zeros(num_pairs, text_seq_len, dtype=torch.long)

    for i in range(num_pairs):
        # Choose a pattern type (0-3)
        pattern = i % 4

        if pattern == 0:
            # Circle
            cx, cy = image_size // 2, image_size // 2
            radius = image_size // 4
            y_coords, x_coords = torch.meshgrid(
                torch.arange(image_size), torch.arange(image_size), indexing="ij"
            )
            mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < radius ** 2
            color = torch.rand(in_channels, generator=gen)
            images[i] = color.view(in_channels, 1, 1) * mask.float().unsqueeze(0)
            # Caption starts with token 1 (circle)
            text_ids[i, 0] = 1
        elif pattern == 1:
            # Rectangle
            rect_h = image_size // 2
            rect_w = image_size // 2
            color = torch.rand(in_channels, 1, 1, generator=gen)
            images[i, :, image_size // 4:3 * image_size // 4,
                   image_size // 4:3 * image_size // 4] = color.expand(in_channels, rect_h, rect_w)
            text_ids[i, 0] = 2
        elif pattern == 2:
            # Horizontal gradient
            gradient = torch.linspace(0, 1, image_size).view(1, 1, -1).expand(in_channels, image_size, -1)
            images[i] = gradient * torch.rand(in_channels, generator=gen).view(in_channels, 1, 1)
            text_ids[i, 0] = 3
        elif pattern == 3:
            # Random noise
            images[i] = torch.rand(in_channels, image_size, image_size, generator=gen)
            text_ids[i, 0] = 4

        # Fill rest of text with random tokens (but correlated with pattern)
        text_ids[i, 1:] = torch.randint(
            5, text_vocab_size, (text_seq_len - 1,), generator=gen,
        )

    return images, text_ids


__all__ = [
    # Issue #23
    "PatchEmbedding",
    "PositionalEmbedding2D",
    "ImageEncoder",
    "CrossModalProjector",
    "MultimodalHVN",
    # Issue #27
    "GatedFusionLayer",
    "ModalityEmbedding",
    "CrossModalRetriever",
    "MultimodalFusionHVN",
    "ImagePreprocessor",
    "generate_synthetic_image_text_pairs",
]