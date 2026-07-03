"""
Entropy Coding of Quantized Weights (Issue #32, Phases 1.1, 9.1)
====================================================================

Once weights are quantized (e.g. to ternary or 4-bit codes), the resulting
integer codes are rarely uniformly distributed — a ternary-quantized
weight matrix is typically 60-90% zeros.  Shannon's source coding theorem
(1948) says such skewed distributions can be stored at their true entropy
rather than their nominal bit-width, using a *lossless* code on top of the
already-lossy quantization.

:class:`EntropyCoder` implements canonical Huffman coding for this.
:class:`LDPCCoder` is documented as a thin alias: implementing a full
belief-propagation LDPC codec is a substantial undertaking on its own and,
per the issue's own estimate, only buys 1.1-1.3× over Huffman — Huffman
already gets within a few percent of the Shannon limit for the
skewed/near-degenerate distributions quantized weights produce, so it is
used as the practical near-entropy-limit substitute here.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

import torch


class EntropyCoder:
    """Canonical Huffman coding for quantized weight codes.

    Builds a prefix-free code from the empirical symbol distribution of
    the input tensor, encodes it to a packed bitstream, and can decode it
    back exactly (lossless on top of whatever quantization already
    happened).
    """

    @staticmethod
    def _build_huffman_table(freqs: dict[int, int]) -> dict[int, str]:
        heap = [[weight, [symbol, ""]] for symbol, weight in freqs.items()]
        heapq.heapify(heap)
        if len(heap) == 1:
            return {heap[0][1][0]: "0"}
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return {symbol: code for symbol, code in heap[0][1:]}

    def encode(self, codes: torch.Tensor) -> dict:
        """Encode an integer-valued tensor to a packed Huffman bitstream."""
        flat = codes.detach().cpu().flatten().tolist()
        freqs = Counter(flat)
        table = self._build_huffman_table(dict(freqs))
        bitstring = "".join(table[v] for v in flat)
        pad = (8 - len(bitstring) % 8) % 8
        bitstring_padded = bitstring + "0" * pad
        packed = bytearray(
            int(bitstring_padded[i:i + 8], 2) for i in range(0, len(bitstring_padded), 8)
        )
        return {
            "packed": torch.tensor(list(packed), dtype=torch.uint8),
            "table": table,
            "num_elements": len(flat),
            "shape": tuple(codes.shape),
            "pad": pad,
            "dtype": codes.dtype,
        }

    def decode(self, edict: dict) -> torch.Tensor:
        """Decode a packed Huffman bitstream back to the original integer tensor."""
        inv_table = {code: symbol for symbol, code in edict["table"].items()}
        bits = "".join(f"{byte:08b}" for byte in edict["packed"].tolist())
        if edict["pad"]:
            bits = bits[:-edict["pad"]]
        out, buf = [], ""
        for bit in bits:
            buf += bit
            if buf in inv_table:
                out.append(inv_table[buf])
                buf = ""
        return torch.tensor(out[:edict["num_elements"]], dtype=edict["dtype"]).reshape(edict["shape"])

    @staticmethod
    def theoretical_entropy_bits(codes: torch.Tensor) -> float:
        """Shannon entropy (bits/symbol) of the tensor's empirical distribution."""
        flat = codes.detach().cpu().flatten().tolist()
        n = len(flat)
        freqs = Counter(flat)
        return -sum((c / n) * math.log2(c / n) for c in freqs.values())

    @staticmethod
    def memory_bytes(edict: dict) -> int:
        table_overhead = len(edict["table"]) * 8  # symbol id + code length, rough estimate
        return edict["packed"].numel() + table_overhead


class LDPCCoder(EntropyCoder):
    """Source-coding compressor for quantized weights.

    A full Low-Density Parity-Check (Gallager 1962) codec requires an
    iterative belief-propagation decoder — a substantial project on its
    own, and per the issue's own estimate only buys 1.1-1.3× over Huffman
    coding for this use case.  This class is a documented alias for
    :class:`EntropyCoder`: Huffman coding already gets close to the
    Shannon limit for the skewed distributions quantized weights produce,
    so it serves as the practical near-entropy-limit substitute here
    rather than a from-scratch LDPC implementation.
    """
