"""
LoRA / QLoRA — Parameter-Efficient Fine-Tuning for HVN (Issue #29)
==================================================================

Implements Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) for the
Hamiltonian Vortex Network.  LoRA freezes the base model weights and trains
small low-rank matrices (A, B) on top of each linear layer, reducing the
number of trainable parameters to 1–10% of the full model.  QLoRA goes
further by quantizing the frozen base model to NF4 (4-bit), enabling
fine-tuning of larger models on 6 GB VRAM.

Key components
--------------
``LoRAAdapter``         — A single low-rank adapter: ``ΔW = B @ A`` where
                          A ∈ ℝ^{r×in}, B ∈ ℝ^{out×r}.  Added to a frozen
                          ``nn.Linear`` layer.
``LoRALinear``          — Wraps a frozen ``nn.Linear`` with a ``LoRAAdapter``
                          and computes ``y = Wx + B(Ax)``.
``apply_lora``          — Walks an ``HamiltonianVortexNetwork`` and replaces
                          every ``nn.Linear`` with a ``LoRALinear`` wrapper.
``QLoRAConfig``         — Configuration container for QLoRA (quantization +
                          LoRA hyper-parameters).
``apply_qlora``         — Quantizes the base model to NF4, then applies LoRA
                          adapters on top.
``merge_lora_weights``  — Folds LoRA deltas back into the base model's
                          weight matrices (for inference with zero overhead).
``freeze_base_model``   — Sets ``requires_grad=False`` on all non-LoRA
                          parameters.
``get_lora_params``     — Returns only the trainable LoRA parameters (for
                          the optimizer).
``count_lora_params``   — Reports trainable vs. total parameter counts.

All LoRA operations preserve O(1) inference — the low-rank matrices add
O(r·d) per layer, which is dominated by the O(d²) Hamiltonian computation.
After merging, there is zero inference overhead.

References
----------
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from .quantize import quantize_nf4, dequantize_nf4


# ---------------------------------------------------------------------------
# 1. LoRA Adapter — the core low-rank decomposition
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """A single LoRA low-rank adapter: ΔW = B @ A.

    The adapter computes ``ΔW @ x`` where ``ΔW = B @ A`` with
    ``A ∈ ℝ^{r×in_features}`` and ``B ∈ ℝ^{out_features×r}``.

    Initialisation follows the LoRA paper:
      - A: Kaiming uniform (or zeros for the first call)
      - B: Zeros (so ΔW = 0 at init → no change to the base model initially)
      - Scaling: ``α / r`` applied to the output

    Parameters
    ----------
    in_features : int
        Input dimension of the adapted layer.
    out_features : int
        Output dimension of the adapted layer.
    rank : int
        LoRA rank (r).  Typical values: 4, 8, 16, 32.
    alpha : float
        LoRA scaling factor.  The effective learning rate is scaled by
        ``alpha / rank``.  Common choice: ``alpha = 2 * rank``.
    dropout : float
        Dropout applied to the LoRA input (helps regularisation).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices: A [r, in], B [out, r]
        # A initialised with Kaiming uniform, B initialised to zero
        # → ΔW = B @ A = 0 at start (no change to base model)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA delta: scaling * B(A(dropout(x))).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape [..., in_features].

        Returns
        -------
        delta : torch.Tensor
            LoRA delta, shape [..., out_features].
        """
        return self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

    def merged_weight(self) -> torch.Tensor:
        """Return the merged delta weight ΔW = scaling * B @ A.

        Shape: [out_features, in_features]
        """
        return self.scaling * (self.lora_B.weight @ self.lora_A.weight)


# ---------------------------------------------------------------------------
# 2. LoRALinear — frozen Linear + LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a LoRA adapter.

    Computes ``y = W_frozen(x) + lora(x)`` where W_frozen has
    ``requires_grad=False`` and only the LoRA parameters are trainable.

    Parameters
    ----------
    base_layer : nn.Linear
        The original linear layer (will be frozen).
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    dropout : float
        LoRA dropout.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRAAdapter(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Move LoRA adapter to the same device as the base layer
        base_device = next(self.base_layer.parameters()).device
        self.lora = self.lora.to(base_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = W_frozen(x) + lora(x)."""
        return self.base_layer(x) + self.lora(x)

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base layer and return a plain Linear.

        After merging, the returned Linear has the combined weights and
        no LoRA overhead.  The original LoRALinear should be replaced by
        the returned module in the model.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged.weight.data = self.base_layer.weight.data + self.lora.merged_weight().to(
            self.base_layer.weight.dtype
        )
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        merged = merged.to(self.base_layer.weight.device)
        return merged


# ---------------------------------------------------------------------------
# 3. apply_lora — inject LoRA adapters into an HVN model
# ---------------------------------------------------------------------------

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to all nn.Linear layers in the model.

    Replaces every ``nn.Linear`` (that is not already a LoRALinear) with a
    ``LoRALinear`` wrapper.  The base model weights are frozen; only LoRA
    parameters are trainable.

    Parameters
    ----------
    model : nn.Module
        The HVN model (HamiltonianVortexNetwork or HVNForCausalLM).
    rank : int
        LoRA rank (r).  Default 8.
    alpha : float
        LoRA scaling factor.  Default 16 (→ scaling = 16/8 = 2).
    dropout : float
        LoRA dropout.  Default 0.
    target_modules : list[str], optional
        If provided, only apply LoRA to layers whose *parent module name*
        contains one of these strings.  E.g. ``["vortex", "decoder"]``.
        If None, applies to all nn.Linear layers.

    Returns
    -------
    model : nn.Module
        The same model with LoRA adapters applied (in-place modification).
    """
    # Default: target all nn.Linear layers
    if target_modules is None:
        target_modules = ["vortex", "memory", "decoder", "block", "output", "gate",
                          "eta_proj", "cross_head", "swiglu", "fusion", "retrieval"]

    modified_count = 0

    def _replace_linears(parent: nn.Module, prefix: str = ""):
        nonlocal modified_count
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this module name matches any target
            should_target = any(t in full_name.lower() for t in target_modules) if target_modules else True

            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and should_target:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, name, lora_layer)
                modified_count += 1
            elif isinstance(child, nn.Module) and not isinstance(child, LoRALinear):
                _replace_linears(child, full_name)

    _replace_linears(model)

    # Freeze all non-LoRA parameters
    freeze_base_model(model)

    print(f"LoRA applied: {modified_count} layers adapted (rank={rank}, alpha={alpha})")
    return model


# ---------------------------------------------------------------------------
# 4. QLoRA — quantize base model + apply LoRA
# ---------------------------------------------------------------------------

class QLoRAConfig:
    """Configuration for QLoRA fine-tuning.

    Parameters
    ----------
    lora_rank : int
        LoRA rank (r).  Default 8.
    lora_alpha : float
        LoRA scaling factor.  Default 16.
    lora_dropout : float
        LoRA dropout.  Default 0.
    quant_bits : int
        Quantization bits for the base model: 4 (NF4) or 8 (INT8).
        Default 4 (NF4, as in the QLoRA paper).
    block_size : int
        NF4 block size.  Default 64.
    target_modules : list[str], optional
        Module name patterns to target for LoRA.  If None, targets all
        linear layers.
    """

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        quant_bits: int = 4,
        block_size: int = 64,
        target_modules: Optional[list[str]] = None,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quant_bits = quant_bits
        self.block_size = block_size
        self.target_modules = target_modules

    def __repr__(self) -> str:
        return (f"QLoRAConfig(rank={self.lora_rank}, alpha={self.lora_alpha}, "
                f"bits={self.quant_bits}, block={self.block_size})")


def apply_qlora(
    model: nn.Module,
    config: Optional[QLoRAConfig] = None,
) -> nn.Module:
    """Apply QLoRA: quantize base model to NF4, then add LoRA adapters.

    This is the QLoRA recipe from Dettmers et al. (2023):
      1. Quantize all base model weights to NF4 (4-bit) — reduces memory ~8×
      2. Apply LoRA adapters on top — only LoRA params are trainable
      3. During forward pass, base weights are dequantized on-the-fly

    The base model's Linear layers are replaced with QLoRALinear layers
    that store quantized weights and dequantize during the forward pass.

    Parameters
    ----------
    model : nn.Module
        The HVN model to apply QLoRA to.
    config : QLoRAConfig, optional
        QLoRA configuration.  If None, uses defaults (rank=8, NF4).

    Returns
    -------
    model : nn.Module
        The model with QLoRA applied.
    """
    if config is None:
        config = QLoRAConfig()

    target_modules = config.target_modules or [
        "vortex", "memory", "decoder", "block", "output", "gate",
        "eta_proj", "cross_head", "swiglu", "fusion", "retrieval",
    ]

    modified_count = 0

    def _replace_linears(parent: nn.Module, prefix: str = ""):
        nonlocal modified_count
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            should_target = any(t in full_name.lower() for t in target_modules) if target_modules else True

            if isinstance(child, nn.Linear) and not isinstance(child, (LoRALinear, QLoRALinear)) and should_target:
                qlora_layer = QLoRALinear(
                    child,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                    quant_bits=config.quant_bits,
                    block_size=config.block_size,
                )
                setattr(parent, name, qlora_layer)
                modified_count += 1
            elif isinstance(child, nn.Module) and not isinstance(child, (LoRALinear, QLoRALinear)):
                _replace_linears(child, full_name)

    _replace_linears(model)
    freeze_base_model(model)

    print(f"QLoRA applied: {modified_count} layers adapted "
          f"(rank={config.lora_rank}, bits={config.quant_bits}, alpha={config.lora_alpha})")
    return model


class QLoRALinear(nn.Module):
    """Quantized Linear + LoRA adapter (QLoRA).

    Stores the base layer's weight in NF4 (4-bit) or INT8 quantized form.
    During the forward pass, the weight is dequantized on-the-fly, then
    the LoRA delta is added.

    y = dequant(W_quant) @ x + lora(x)

    Parameters
    ----------
    base_layer : nn.Linear
        The original linear layer (will be quantized and frozen).
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    dropout : float
        LoRA dropout.
    quant_bits : int
        4 for NF4, 8 for INT8.
    block_size : int
        NF4 block size.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        quant_bits: int = 4,
        block_size: int = 64,
    ):
        super().__init__()
        self.quant_bits = quant_bits
        self.block_size = block_size
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.has_bias = base_layer.bias is not None

        # Quantize the base weight
        weight = base_layer.weight.data
        if quant_bits == 4:
            self.weight_quant = quantize_nf4(weight, block_size=block_size)
            # Move quantized data to the same device as the original weight
            self.weight_quant["codes"] = self.weight_quant["codes"].to(weight.device)
            self.weight_quant["scales"] = self.weight_quant["scales"].to(weight.device)
        elif quant_bits == 8:
            from .quantize import quantize_int8, dequantize_int8
            q, scale = quantize_int8(weight, per_channel=False)
            self.weight_quant = {"int8_q": q.to(weight.device), "int8_scale": scale.to(weight.device)}
        else:
            raise ValueError(f"Unsupported quant_bits: {quant_bits}. Use 4 or 8.")

        # Store bias (not quantized — biases are small)
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.bias = None

        # LoRA adapter (trainable)
        self.lora = LoRAAdapter(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Move LoRA adapter to the same device as the base weight
        base_device = weight.device
        self.lora = self.lora.to(base_device)

    def _dequantize_weight(self, target_device: torch.device | None = None) -> torch.Tensor:
        """Dequantize the stored weight to fp32.

        Parameters
        ----------
        target_device : torch.device, optional
            If provided, move the dequantized weight to this device.
        """
        if self.quant_bits == 4:
            qdict = self.weight_quant
            # Ensure all quantized data is on the target device
            codes = qdict["codes"]
            scales = qdict["scales"]
            if target_device is not None:
                codes = codes.to(target_device)
                scales = scales.to(target_device)
            # Reconstruct dict with device-corrected tensors
            corrected = {**qdict, "codes": codes, "scales": scales}
            return dequantize_nf4(corrected)
        else:
            from .quantize import dequantize_int8
            q = self.weight_quant["int8_q"]
            scale = self.weight_quant["int8_scale"]
            if target_device is not None:
                q = q.to(target_device)
                scale = scale.to(target_device)
            return dequantize_int8(q, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = dequant(W) @ x + lora(x) + bias."""
        weight = self._dequantize_weight(target_device=x.device).to(x.dtype)
        out = torch.nn.functional.linear(x, weight, self.bias)
        out = out + self.lora(x)
        return out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the dequantized base and return a plain Linear.

        After merging, the returned Linear has full-precision weights (dequantized
        base + LoRA delta) and no quantization or LoRA overhead.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.has_bias,
        )
        lora_device = next(self.lora.parameters()).device
        base_weight = self._dequantize_weight(target_device=lora_device)
        merged.weight.data = (base_weight + self.lora.merged_weight()).to(torch.float32)
        if self.bias is not None:
            merged.bias.data = self.bias.data.clone()
        merged = merged.to(lora_device)
        return merged


# ---------------------------------------------------------------------------
# 5. Utility functions
# ---------------------------------------------------------------------------

def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters that are not part of LoRA/QLoRA adapters.

    Sets ``requires_grad=False`` on all parameters except those in
    ``LoRAAdapter`` modules.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters.

    Use this to configure the optimizer to only update LoRA weights.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_lora_params(model: nn.Module) -> dict:
    """Count trainable vs. total parameters.

    Returns
    -------
    dict with keys:
        ``total`` : int — total parameter count
        ``trainable`` : int — trainable parameter count (LoRA only)
        ``frozen`` : int — frozen parameter count (base model)
        ``trainable_pct`` : float — trainable / total * 100
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = (trainable / total * 100) if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": pct,
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA/QLoRA adapters back into the base model.

    Replaces every ``LoRALinear`` / ``QLoRALinear`` with a plain ``nn.Linear``
    containing the merged weights.  After merging, the model has zero LoRA
    overhead and can be used for inference exactly like a standard model.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA/QLoRA adapters applied.

    Returns
    -------
    model : nn.Module
        Model with all adapters merged (in-place modification).
    """
    merged_count = 0

    def _merge_in_parent(parent: nn.Module):
        nonlocal merged_count
        for name, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, QLoRALinear)):
                merged = child.merge()
                setattr(parent, name, merged)
                merged_count += 1
            elif isinstance(child, nn.Module):
                _merge_in_parent(child)

    _merge_in_parent(model)
    print(f"LoRA merged: {merged_count} adapters folded into base weights")
    return model


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    extra_metadata: Optional[dict] = None,
) -> None:
    """Save only the LoRA adapter weights (not the frozen base model).

    This produces a small checkpoint containing only the trainable LoRA
    parameters, which can be loaded on top of a base model later.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA adapters applied.
    path : str
        Save path for the LoRA checkpoint.
    extra_metadata : dict, optional
        Additional metadata to save (e.g. model config, training info).
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.clone()

    save_dict = {"lora_state_dict": lora_state}
    if extra_metadata:
        save_dict["metadata"] = extra_metadata

    torch.save(save_dict, path)
    print(f"✓ LoRA checkpoint saved to {path} ({len(lora_state)} trainable tensors)")


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = False,
) -> dict:
    """Load LoRA adapter weights into a model that already has LoRA applied.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA adapters already applied (same architecture).
    path : str
        Path to the LoRA checkpoint.
    strict : bool
        If True, all LoRA keys must match exactly.

    Returns
    -------
    metadata : dict
        Metadata from the checkpoint (if any).
    """
    ckpt = torch.load(path, weights_only=False)
    lora_state = ckpt["lora_state_dict"]

    model_state = model.state_dict()
    loaded = 0
    for name, param in model.named_parameters():
        if name in lora_state:
            param.data = lora_state[name].to(param.device)
            loaded += 1

    metadata = ckpt.get("metadata", {})
    print(f"✓ LoRA checkpoint loaded from {path} ({loaded}/{len(lora_state)} tensors loaded)")
    return metadata


# ---------------------------------------------------------------------------
# 6. Instruction tuning data utilities
# ---------------------------------------------------------------------------

def generate_instruction_training_data(
    num_examples: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic instruction-tuning examples.

    Each example has the format:
        {
            "instruction": str,
            "input": str (optional, may be empty),
            "output": str,
        }

    The synthetic examples cover simple tasks like:
      - Summarisation
      - Translation (pig latin)
      - Question answering
      - Text classification
      - Code generation

    Parameters
    ----------
    num_examples : int
        Number of examples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    examples : list[dict]
        Instruction-formatted training examples.
    """
    import random
    rng = random.Random(seed)

    templates = [
        # Summarisation
        {
            "instruction": "Summarise the following text in one sentence.",
            "input_generator": lambda: rng.choice([
                "The Hamiltonian Vortex Network uses fluid dynamics and quantum mechanics for sequence modeling. "
                "It achieves O(1) inference by compressing history into a fixed-size matrix.",
                "LoRA reduces the number of trainable parameters by decomposing weight updates into low-rank matrices. "
                "This enables efficient fine-tuning on consumer GPUs.",
                "The Biot-Savart law describes how a vortex particle's velocity is induced by the surrounding vorticity field. "
                "HVN uses this as a local approximation to attention.",
            ]),
            "output_generator": lambda text: rng.choice([
                text.split(".")[0].strip() + ".",
                "It uses a fixed-size matrix for efficient inference.",
                "The method enables efficient computation.",
            ]),
        },
        # Translation (pig latin)
        {
            "instruction": "Translate the following text to pig latin.",
            "input_generator": lambda: rng.choice([
                "hello world",
                "the vortex network",
                "quantum entanglement",
                "hamiltonian memory",
            ]),
            "output_generator": lambda text: " ".join(
                word[1:] + word[0] + "ay" if word else ""
                for word in text.split()
            ),
        },
        # Question answering
        {
            "instruction": "Answer the question.",
            "input_generator": lambda: rng.choice([
                "What does HVN stand for?",
                "What is the inference complexity of HVN?",
                "What does LoRA stand for?",
                "What quantization does QLoRA use?",
            ]),
            "output_generator": lambda q: {
                "What does HVN stand for?": "Hamiltonian Vortex Network.",
                "What is the inference complexity of HVN?": "O(1) per token, constant in sequence length.",
                "What does LoRA stand for?": "Low-Rank Adaptation.",
                "What quantization does QLoRA use?": "NF4 4-bit quantization.",
            }.get(q, "I do not know."),
        },
        # Text classification
        {
            "instruction": "Classify the sentiment (positive/negative/neutral).",
            "input_generator": lambda: rng.choice([
                "This architecture is amazing and efficient!",
                "The model quality is poor and needs improvement.",
                "The paper describes a novel approach.",
                "I love how fast the inference is!",
                "The perplexity is too high for practical use.",
            ]),
            "output_generator": lambda text: (
                "positive" if any(w in text.lower() for w in ["amazing", "love", "great", "excellent"]) else
                "negative" if any(w in text.lower() for w in ["poor", "too high", "bad", "terrible"]) else
                "neutral"
            ),
        },
        # Code generation
        {
            "instruction": "Generate Python code for the following task.",
            "input_generator": lambda: rng.choice([
                "Create a function that adds two numbers.",
                "Create a function that reverses a string.",
                "Create a function that checks if a number is even.",
            ]),
            "output_generator": lambda task: {
                "Create a function that adds two numbers.": "def add(a, b):\n    return a + b",
                "Create a function that reverses a string.": "def reverse(s):\n    return s[::-1]",
                "Create a function that checks if a number is even.": "def is_even(n):\n    return n % 2 == 0",
            }.get(task, "def f():\n    pass"),
        },
        # Text transformation
        {
            "instruction": "Convert the following text to uppercase.",
            "input_generator": lambda: rng.choice([
                "hamiltonian vortex",
                "low rank adaptation",
                "quantized fine tuning",
            ]),
            "output_generator": lambda text: text.upper(),
        },
    ]

    examples = []
    for i in range(num_examples):
        template = templates[i % len(templates)]
        inp = template["input_generator"]()
        out = template["output_generator"](inp)
        examples.append({
            "instruction": template["instruction"],
            "input": inp,
            "output": out,
        })

    return examples


def format_instruction_example(
    example: dict,
    tokenizer=None,
) -> str:
    """Format an instruction example as a training string.

    Uses the Alpaca prompt format:

        Below is an instruction that describes a task. Write a response.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    If the input is empty, the ### Input section is omitted.

    Parameters
    ----------
    example : dict
        Instruction example with keys "instruction", "input", "output".
    tokenizer : optional
        Unused (kept for API compatibility).  The string is returned raw.

    Returns
    -------
    text : str
        Formatted instruction string.
    """
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp:
        return (
            f"Below is an instruction that describes a task. Write a response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        return (
            f"Below is an instruction that describes a task. Write a response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )