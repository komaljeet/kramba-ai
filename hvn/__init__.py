from .core import (
    VortexInjectionLayer,
    HamiltonianMemoryBlock,
    HVNBlock,
    HVNDecoder,
    HamiltonianVortexNetwork,
)
from .hf_model import HVNConfig, HVNForCausalLM
from .tokenizer import BPETokenizer
from .vision import (
    PatchEmbedding,
    PositionalEmbedding2D,
    ImageEncoder,
    CrossModalProjector,
    MultimodalHVN,
)
from .quantize import (
    quantize_int8,
    dequantize_int8,
    quantize_nf4,
    dequantize_nf4,
    quantize_hamiltonian,
    quantize_model_hamiltonians,
    quantize_hamiltonian_states,
    dequantize_hamiltonian_states,
    QuantizedHamiltonian,
    QuantizedHVNInference,
    estimate_memory_savings,
)
from .tools import (
    ToolRegistry,
    ToolCallParser,
    ToolExecutor,
    ConstrainedJSONDecoder,
    ToolCallingSession,
    TOOL_TOKENS,
    SPECIAL_TOKEN_STRINGS,
    create_demo_registry,
    generate_tool_training_data,
)

__all__ = [
    "VortexInjectionLayer",
    "HamiltonianMemoryBlock",
    "HVNBlock",
    "HVNDecoder",
    "HamiltonianVortexNetwork",
    "HVNConfig",
    "HVNForCausalLM",
    "BPETokenizer",
    # Vision / multimodal (Issue #23)
    "PatchEmbedding",
    "PositionalEmbedding2D",
    "ImageEncoder",
    "CrossModalProjector",
    "MultimodalHVN",
    # Quantization
    "quantize_int8",
    "dequantize_int8",
    "quantize_nf4",
    "dequantize_nf4",
    "quantize_hamiltonian",
    "quantize_model_hamiltonians",
    "quantize_hamiltonian_states",
    "dequantize_hamiltonian_states",
    "QuantizedHamiltonian",
    "QuantizedHVNInference",
    "estimate_memory_savings",
    # Tool calling (Issue #24)
    "ToolRegistry",
    "ToolCallParser",
    "ToolExecutor",
    "ConstrainedJSONDecoder",
    "ToolCallingSession",
    "TOOL_TOKENS",
    "SPECIAL_TOKEN_STRINGS",
    "create_demo_registry",
    "generate_tool_training_data",
]
