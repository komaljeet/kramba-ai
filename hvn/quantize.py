"""
Hamiltonian Quantization for Inference
=======================================

INT8 and 4-bit quantization for the Hamiltonian matrices — the main memory
consumers in HVN.  Quantizing them from fp32 to INT8 (4× smaller) or 4-bit
(8× smaller) dramatically reduces inference memory and can speed up inference
by reducing memory bandwidth pressure.

Two quantization schemes are provided:

1. **INT8 symmetric quantization** — per-channel or per-tensor.
   Each fp32 weight is mapped to an int8 value in [-127, 127] using a scale
   factor ``s = max(|w|) / 127``.  Dequantization: ``w ≈ q * s``.

2. **NF4 (NormalFloat 4-bit) quantization** — block-wise.
   Weights are divided into blocks of ``block_size`` elements (default 64).
   Each block is normalized to unit standard deviation, then quantized to one
   of 16 fixed levels chosen to match the normal distribution of weights
   (the same distribution used by QLoRA / bitsandbytes NF4).

The :class:`QuantizedHamiltonian` wrapper stores the quantized representation
and provides a :meth:`dequantize` method that reconstructs the fp32 matrix
on-the-fly for use in the Biot-Savart velocity computation and the
entanglement update.

A convenience function :func:`quantize_model_hamiltonians` walks an
``HamiltonianVortexNetwork`` and replaces every
``MultiHeadHamiltonianMemory`` / ``HamiltonianMemoryBlock`` with a quantized
variant for inference.

All quantization is **inference-only** — training should use full precision.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

# ---------------------------------------------------------------------------
# INT8 symmetric quantization
# ---------------------------------------------------------------------------


def quantize_int8(
    w: torch.Tensor,
    per_channel: bool = False,
    channel_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization.

    Parameters
    ----------
    w : torch.Tensor
        Float tensor to quantize (any shape).
    per_channel : bool
        If ``True``, compute a separate scale per slice along *channel_dim*.
    channel_dim : int
        Dimension along which per-channel scales are computed.

    Returns
    -------
    q : torch.Tensor
        Quantized int8 tensor, same shape as *w*.
    scale : torch.Tensor
        Float scale tensor.  Shape ``()`` for per-tensor or
        ``(w.shape[:channel_dim] + (1,) * (w.ndim - channel_dim - 1))``
        for per-channel (broadcastable to *w*).
    """
    orig_dtype = w.dtype
    w_f32 = w.to(torch.float32)

    if per_channel:
        reduce_dims = [d for d in range(w_f32.ndim) if d != channel_dim]
        max_abs = w_f32.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    else:
        max_abs = w_f32.abs().amax().clamp(min=1e-8)

    scale = max_abs / 127.0
    q = torch.round(w_f32 / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(orig_dtype)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an INT8 tensor back to float.

    Parameters
    ----------
    q : torch.Tensor
        int8 quantized tensor.
    scale : torch.Tensor
        Scale tensor (broadcastable to *q*).

    Returns
    -------
    w : torch.Tensor
        Dequantized float tensor (same dtype as *scale*).
    """
    return q.to(scale.dtype) * scale


# ---------------------------------------------------------------------------
# NF4 (NormalFloat 4-bit) quantization
# ---------------------------------------------------------------------------

# The 16 NF4 codebook values — quantile levels of a standard normal distribution.
# These are the exact values from the QLoRA paper (Dettmers et al., 2023).
_NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444141149520874,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


def _get_nf4_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the NF4 codebook on the requested device/dtype."""
    return _NF4_CODEBOOK.to(device=device, dtype=dtype)


def quantize_nf4(
    w: torch.Tensor,
    block_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Block-wise NF4 (4-bit NormalFloat) quantization.

    The input tensor is reshaped into blocks of *block_size* elements.  Each
    block is normalized by its own absmax, quantized to one of 16 NF4 levels,
    and stored as packed int8 (two 4-bit values per byte).

    Parameters
    ----------
    w : torch.Tensor
        Float tensor to quantize (any shape, will be flattened).
    block_size : int
        Number of elements per quantization block (default 64, as in QLoRA).

    Returns
    -------
    dict with keys:
        ``"codes"``      — packed int8 tensor (two 4-bit codes per byte).
        ``"scales"``     — float absmax scale per block, shape [num_blocks].
        ``"packed_shape"`` — original tensor shape (for reconstruction).
        ``"block_size"`` — block size used.
        ``"num_blocks"`` — number of blocks.
    """
    orig_shape = w.shape
    orig_dtype = w.dtype
    device = w.device
    w_f32 = w.to(torch.float32).reshape(-1)

    num_elements = w_f32.numel()
    # Pad to a multiple of block_size
    pad_len = (block_size - (num_elements % block_size)) % block_size
    if pad_len > 0:
        w_f32 = torch.nn.functional.pad(w_f32, (0, pad_len))

    num_blocks = w_f32.numel() // block_size
    blocks = w_f32.view(num_blocks, block_size)  # [num_blocks, block_size]

    # Per-block absmax scale
    scales = blocks.abs().amax(dim=1).clamp(min=1e-8)  # [num_blocks]
    normalized = blocks / scales.unsqueeze(1)  # [-1, 1] range

    # Quantize: find nearest NF4 code for each element
    codebook = _get_nf4_codebook(device, torch.float32)  # [16]
    # Compute distances: [num_blocks, block_size, 16]
    diff = (normalized.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)).abs()
    codes = diff.argmin(dim=-1).to(torch.uint8)  # [num_blocks, block_size], values 0-15

    # Pack two 4-bit codes into one int8 byte
    codes_flat = codes.reshape(-1)  # [num_blocks * block_size]
    assert codes_flat.numel() % 2 == 0, "Total codes must be even for packing"
    even = codes_flat[0::2].to(torch.uint8)
    odd = codes_flat[1::2].to(torch.uint8)
    packed = (even << 4) | odd  # int8 (but values 0-255, stored as uint8)

    return {
        "codes": packed.view(torch.int8),  # store as int8 for torch.save compatibility
        "scales": scales.to(orig_dtype),
        "packed_shape": orig_shape,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "num_elements": num_elements,
    }


def dequantize_nf4(qdict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Dequantize an NF4-quantized tensor back to float.

    Parameters
    ----------
    qdict : dict
        Dictionary returned by :func:`quantize_nf4`.

    Returns
    -------
    w : torch.Tensor
        Dequantized float tensor with the original shape.
    """
    packed = qdict["codes"].view(torch.uint8)  # [num_packed_bytes]
    scales = qdict["scales"].to(torch.float32)
    block_size = qdict["block_size"]
    num_blocks = qdict["num_blocks"]
    num_elements = qdict["num_elements"]
    orig_shape = qdict["packed_shape"]
    device = packed.device

    # Unpack: two 4-bit codes per byte
    even = (packed >> 4) & 0x0F  # high nibble
    odd = packed & 0x0F  # low nibble
    codes_flat = torch.stack([even, odd], dim=1).reshape(-1).to(torch.int64)  # [num_blocks * block_size]

    # Lookup in NF4 codebook
    codebook = _get_nf4_codebook(device, torch.float32)  # [16]
    dequant_flat = codebook[codes_flat]  # [num_blocks * block_size]

    # Reshape to blocks and rescale
    blocks = dequant_flat.view(num_blocks, block_size)
    dequant = blocks * scales.unsqueeze(1)  # [num_blocks, block_size]
    dequant = dequant.reshape(-1)[:num_elements]

    return dequant.view(orig_shape)


# ---------------------------------------------------------------------------
# Quantized Hamiltonian wrapper
# ---------------------------------------------------------------------------


class QuantizedHamiltonian:
    """Stores a Hamiltonian matrix in quantized form and dequantizes on demand.

    This is a lightweight container (not an ``nn.Module``) intended for
    inference.  It holds the quantized codes + scales and provides a
    :meth:`dequantize` method that reconstructs the fp32 (or bf16) matrix.

    Parameters
    ----------
    quantized : torch.Tensor or dict
        Either:
        - INT8: a tuple of (q_int8, scale) — pass via *int8_data*.
        - NF4:  a dict from :func:`quantize_nf4` — pass via *nf4_data*.
    bits : int
        8 for INT8, 4 for NF4.
    """

    def __init__(
        self,
        int8_data: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        nf4_data: Optional[dict[str, torch.Tensor]] = None,
        bits: int = 8,
    ):
        self.bits = bits
        if bits == 8:
            assert int8_data is not None, "INT8 quantization requires int8_data"
            self.q, self.scale = int8_data
        elif bits == 4:
            assert nf4_data is not None, "NF4 quantization requires nf4_data"
            self.qdict = nf4_data
        else:
            raise ValueError(f"Unsupported bits: {bits}. Use 8 or 4.")

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct the Hamiltonian matrix as a float tensor.

        Parameters
        ----------
        dtype : torch.dtype
            Output dtype (default fp32).  Use bf16/fp16 for faster inference.

        Returns
        -------
        H : torch.Tensor
            Dequantized Hamiltonian matrix.
        """
        if self.bits == 8:
            return dequantize_int8(self.q, self.scale.to(dtype))
        else:
            return dequantize_nf4(self.qdict).to(dtype)

    def memory_bytes(self) -> int:
        """Return the memory footprint of the quantized representation in bytes."""
        if self.bits == 8:
            return self.q.numel() + self.scale.numel() * 4  # int8 + fp32 scale
        else:
            return self.qdict["codes"].numel() + self.qdict["scales"].numel() * 4

    @property
    def shape(self) -> tuple[int, ...]:
        """Original shape of the quantized tensor."""
        if self.bits == 8:
            return tuple(self.q.shape)
        else:
            return tuple(self.qdict["packed_shape"])


# ---------------------------------------------------------------------------
# Model-level quantization utilities
# ---------------------------------------------------------------------------


def quantize_hamiltonian(
    H: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    block_size: int = 64,
) -> QuantizedHamiltonian:
    """Quantize a Hamiltonian matrix tensor.

    Parameters
    ----------
    H : torch.Tensor
        Hamiltonian matrix, shape ``[B*n_heads, d_head, d_head]`` or
        ``[batch, d_model, d_model]``.
    bits : int
        8 for INT8, 4 for NF4 4-bit.
    per_channel : bool
        For INT8: use per-channel quantization along the last dimension.
    block_size : int
        For NF4: block size for block-wise quantization.

    Returns
    -------
    QuantizedHamiltonian
        Wrapper holding the quantized representation.
    """
    if bits == 8:
        q, scale = quantize_int8(H, per_channel=per_channel, channel_dim=-1)
        return QuantizedHamiltonian(int8_data=(q, scale), bits=8)
    elif bits == 4:
        qdict = quantize_nf4(H, block_size=block_size)
        return QuantizedHamiltonian(nf4_data=qdict, bits=4)
    else:
        raise ValueError(f"Unsupported bits: {bits}. Use 8 or 4.")


def quantize_model_hamiltonians(
    model: nn.Module,
    bits: int = 8,
    per_channel: bool = False,
    block_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict[str, dict]:
    """Quantize all Hamiltonian-related parameters in an HVN model.

    This function scans the model for ``MultiHeadHamiltonianMemory`` and
    ``HamiltonianMemoryBlock`` modules and quantizes their learnable
    parameters (``eta``, ``memory_gate``, and ``eta_proj`` weights) to the
    specified precision.  The original fp32 parameters are replaced in-place
    with quantized versions, and a metadata dict is returned for restoration.

    .. note::
        The Hamiltonian *state tensors* (the ``[B*n_heads, d_head, d_head]``
        matrices) are created at runtime by ``init_hamiltonian()`` and are
        not stored as model parameters.  To quantize the runtime Hamiltonian
        states during inference, use :func:`quantize_hamiltonian` on the
        state tensors directly, or use the :class:`QuantizedHVNInference`
        wrapper which handles this automatically.

    Parameters
    ----------
    model : nn.Module
        An ``HamiltonianVortexNetwork`` (or ``HVNForCausalLM``).
    bits : int
        8 for INT8, 4 for NF4.
    per_channel : bool
        Per-channel INT8 quantization.
    block_size : int
        NF4 block size.
    device : torch.device, optional
        Device for quantized tensors.  Defaults to the model's device.

    Returns
    -------
    metadata : dict
        Dictionary mapping parameter names to their quantized representations,
        for analysis or restoration.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    metadata: dict[str, dict] = {}

    for name, module in model.named_modules():
        if isinstance(module, (MultiHeadHamiltonianMemory, HamiltonianMemoryBlock)):
            # Quantize the learnable parameters of the memory module
            for pname, param in module.named_parameters():
                full_name = f"{name}.{pname}" if name else pname
                if bits == 8:
                    q, scale = quantize_int8(param.data, per_channel=per_channel)
                    deq = dequantize_int8(q, scale)
                else:
                    qdict = quantize_nf4(param.data, block_size=block_size)
                    deq = dequantize_nf4(qdict)

                # Replace the parameter data with dequantized version (still fp32,
                # but the metadata stores the quantized form for memory analysis)
                param.data = deq.to(device)
                metadata[full_name] = {
                    "bits": bits,
                    "orig_bytes": param.numel() * 4,
                    "quant_bytes": (
                        q.numel() + scale.numel() * 4 if bits == 8
                        else qdict["codes"].numel() + qdict["scales"].numel() * 4
                    ),
                    "max_abs_error": (deq.to(torch.float32) - param.data.to(torch.float32)).abs().max().item(),
                }

    return metadata


def quantize_hamiltonian_states(
    states: list[torch.Tensor] | list[list[list[torch.Tensor]]],
    bits: int = 8,
    per_channel: bool = False,
    block_size: int = 64,
) -> list:
    """Quantize runtime Hamiltonian state tensors for inference.

    Takes the Hamiltonian states (as produced by ``init_generation_state``
    or ``init_generation_state_batched``) and returns quantized versions.

    Parameters
    ----------
    states : list
        Either a list of batched tensors ``[B*n_heads, d_head, d_head]``
        (from ``init_generation_state_batched``) or a nested list of
        per-head states (from ``init_generation_state``).
    bits : int
        8 for INT8, 4 for NF4.
    per_channel : bool
        Per-channel INT8 quantization.
    block_size : int
        NF4 block size.

    Returns
    -------
    list
        Same structure as *states* but with :class:`QuantizedHamiltonian`
        objects replacing the raw tensors.
    """
    quantized_states = []
    for state in states:
        if isinstance(state, torch.Tensor):
            # Batched state: [B*n_heads, d_head, d_head]
            qh = quantize_hamiltonian(state, bits=bits, per_channel=per_channel, block_size=block_size)
            quantized_states.append(qh)
        elif isinstance(state, list):
            # Per-head state: [[H_0], [H_1], ...]
            inner = []
            for h_state in state:
                if isinstance(h_state, torch.Tensor):
                    inner.append(
                        quantize_hamiltonian(h_state, bits=bits, per_channel=per_channel, block_size=block_size)
                    )
                else:
                    inner.append(h_state)
            quantized_states.append(inner)
        else:
            quantized_states.append(state)
    return quantized_states


def dequantize_hamiltonian_states(
    quantized_states: list,
    dtype: torch.dtype = torch.float32,
) -> list:
    """Dequantize Hamiltonian states back to float tensors.

    Parameters
    ----------
    quantized_states : list
        States quantized by :func:`quantize_hamiltonian_states`.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    list
        Same structure but with float tensors.
    """
    deq_states = []
    for state in quantized_states:
        if isinstance(state, QuantizedHamiltonian):
            deq_states.append(state.dequantize(dtype=dtype))
        elif isinstance(state, list):
            inner = []
            for h_state in state:
                if isinstance(h_state, QuantizedHamiltonian):
                    inner.append(h_state.dequantize(dtype=dtype))
                else:
                    inner.append(h_state)
            deq_states.append(inner)
        else:
            deq_states.append(state)
    return deq_states


# ---------------------------------------------------------------------------
# Full quantized inference wrapper
# ---------------------------------------------------------------------------


class QuantizedHVNInference:
    """Wrapper for quantized HVN inference.

    Wraps a trained ``HamiltonianVortexNetwork`` and quantizes the Hamiltonian
    states on-the-fly during incremental generation.  This reduces the memory
    footprint of the Hamiltonian cache by 4× (INT8) or 8× (NF4) compared to
    fp32, with minimal quality loss.

    The model's learnable parameters (embeddings, projections, FFN) remain
    in their original precision — only the runtime Hamiltonian state tensors
    are quantized.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        Trained HVN model.
    bits : int
        8 for INT8, 4 for NF4 4-bit.
    per_channel : bool
        Per-channel INT8 quantization.
    block_size : int
        NF4 block size.
    dequant_dtype : torch.dtype
        Dtype to use when dequantizing Hamiltonian states for computation.
        Default ``torch.float32``.  Use ``torch.bfloat16`` for faster
        inference on supported GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        per_channel: bool = False,
        block_size: int = 64,
        dequant_dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.bits = bits
        self.per_channel = per_channel
        self.block_size = block_size
        self.dequant_dtype = dequant_dtype
        self.device = next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate_incremental(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
    ) -> list[int]:
        """Incremental generation with quantized Hamiltonian states.

        After each token, the Hamiltonian state is quantized to ``bits``-bit
        precision.  Before the next forward step, the state is dequantized
        back to float for computation.  This simulates the memory savings of
        a fully-quantized inference path while keeping the computation in
        higher precision for numerical stability.

        Parameters
        ----------
        prompt_ids : list[int]
            Token IDs for the prompt.
        max_new_tokens : int
            Number of new tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).

        Returns
        -------
        generated : list[int]
            Full token list (prompt + generated).
        """
        if len(prompt_ids) == 0:
            prompt_ids = [0]

        B = 1
        param_dtype = next(self.model.parameters()).dtype
        states = self.model.init_generation_state(B, device=self.device, dtype=param_dtype)

        # Process prompt token-by-token, quantizing H after each step
        for t in range(len(prompt_ids)):
            tok = torch.tensor([[prompt_ids[t]]], dtype=torch.long, device=self.device)
            # Dequantize states for computation
            states = dequantize_hamiltonian_states(states, dtype=param_dtype)
            logits_step, states = self.model.forward_step(tok, states, position_idx=t)
            # Quantize states for storage
            states = quantize_hamiltonian_states(
                states, bits=self.bits, per_channel=self.per_channel, block_size=self.block_size
            )

        next_logits = logits_step[0, -1, :] / temperature
        generated = list(prompt_ids)

        for step_idx in range(max_new_tokens):
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k_val)
                next_logits = next_logits.clone()
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)

            tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            # Dequantize → compute → quantize
            states = dequantize_hamiltonian_states(states, dtype=param_dtype)
            logits_step, states = self.model.forward_step(
                tok, states, position_idx=len(prompt_ids) + step_idx
            )
            states = quantize_hamiltonian_states(
                states, bits=self.bits, per_channel=self.per_channel, block_size=self.block_size
            )
            next_logits = logits_step[0, -1, :] / temperature

        return generated

    @torch.no_grad()
    def generate_batched(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
    ) -> list[list[int]]:
        """Batched incremental generation with quantized Hamiltonian states.

        Parameters
        ----------
        prompt_ids_list : list[list[int]]
            List of prompt token ID lists, one per sequence.
        max_new_tokens : int
            Number of new tokens to generate per sequence.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k sampling (0 = disabled).

        Returns
        -------
        generated_list : list[list[int]]
            Full token lists (prompt + generated), one per sequence.
        """
        B = len(prompt_ids_list)
        if B == 0:
            return []

        prompt_ids_list = [p if len(p) > 0 else [0] for p in prompt_ids_list]
        max_prompt_len = max(len(p) for p in prompt_ids_list)

        param_dtype = next(self.model.parameters()).dtype
        states = self.model.init_generation_state_batched(B, device=self.device, dtype=param_dtype)

        # Process prompts
        for t in range(max_prompt_len):
            tok_ids = torch.zeros(B, 1, dtype=torch.long, device=self.device)
            for b, prompt in enumerate(prompt_ids_list):
                if t < len(prompt):
                    tok_ids[b, 0] = prompt[t]

            states = dequantize_hamiltonian_states(states, dtype=param_dtype)
            logits_step, states = self.model.forward_step_batched(tok_ids, states, position_idx=t)
            states = quantize_hamiltonian_states(
                states, bits=self.bits, per_channel=self.per_channel, block_size=self.block_size
            )

        next_logits = logits_step[:, -1, :] / temperature
        generated_list = [list(p) for p in prompt_ids_list]

        for step_idx in range(max_new_tokens):
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k_val, dim=-1)
                threshold = values[:, -1:]
                next_logits = next_logits.clone()
                next_logits[next_logits < threshold] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)

            for b in range(B):
                generated_list[b].append(next_ids[b, 0].item())

            states = dequantize_hamiltonian_states(states, dtype=param_dtype)
            logits_step, states = self.model.forward_step_batched(
                next_ids, states, position_idx=max_prompt_len + step_idx
            )
            states = quantize_hamiltonian_states(
                states, bits=self.bits, per_channel=self.per_channel, block_size=self.block_size
            )
            next_logits = logits_step[:, -1, :] / temperature

        return generated_list

    def hamiltonian_memory_bytes(self, states: list) -> int:
        """Compute the memory footprint of quantized Hamiltonian states.

        Parameters
        ----------
        states : list
            Quantized Hamiltonian states (from
            :func:`quantize_hamiltonian_states`).

        Returns
        -------
        int
            Total bytes used by the quantized states.
        """
        total = 0
        for state in states:
            if isinstance(state, QuantizedHamiltonian):
                total += state.memory_bytes()
            elif isinstance(state, list):
                for h_state in state:
                    if isinstance(h_state, QuantizedHamiltonian):
                        total += h_state.memory_bytes()
        return total


# ---------------------------------------------------------------------------
# Convenience: estimate memory savings
# ---------------------------------------------------------------------------


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int = 1,
    bits: int = 8,
) -> dict[str, float]:
    """Estimate the memory savings from quantizing Hamiltonian states.

    Parameters
    ----------
    model : nn.Module
        HVN model.
    batch_size : int
        Inference batch size.
    bits : int
        8 for INT8, 4 for NF4.

    Returns
    -------
    dict with keys:
        ``"fp32_bytes"``    — Hamiltonian state memory in fp32.
        ``"quant_bytes"``   — Hamiltonian state memory quantized.
        ``"compression"``   — Compression ratio (fp32 / quant).
        ``"savings_pct"``   — Percentage memory saved.
    """
    # Compute total Hamiltonian state size
    total_elements = 0
    for module in model.modules():
        if isinstance(module, MultiHeadHamiltonianMemory):
            # State: [B*n_heads, d_head, d_head]
            total_elements += batch_size * module.n_heads * module.d_head * module.d_head
        elif isinstance(module, HamiltonianMemoryBlock):
            total_elements += batch_size * module.d_model * module.d_model

    fp32_bytes = total_elements * 4  # fp32 = 4 bytes

    if bits == 8:
        # INT8: 1 byte per element + small scale overhead
        quant_bytes = total_elements * 1 + total_elements // 64 * 4  # ~1.06 bytes/element
    elif bits == 4:
        # NF4: 0.5 bytes per element + scales (1 fp32 per 64 elements)
        quant_bytes = total_elements // 2 + total_elements // 64 * 4  # ~0.56 bytes/element
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return {
        "fp32_bytes": fp32_bytes,
        "quant_bytes": quant_bytes,
        "compression": fp32_bytes / max(quant_bytes, 1),
        "savings_pct": (1 - quant_bytes / max(fp32_bytes, 1)) * 100,
    }


# Late imports for isinstance checks (avoid circular imports)
from .core.batched_memory import MultiHeadHamiltonianMemory
from .core.hamiltonian_memory import HamiltonianMemoryBlock