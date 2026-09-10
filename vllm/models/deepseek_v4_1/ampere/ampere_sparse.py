# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/RTX 3090).

Mirrors ``vllm.models.deepseek_v4.ampere.ampere_sparse``: reuses the V4.1
ROCm Triton sparse-MLA implementation wholesale. Its kernels, ragged
metadata builders (topk + SWA), and bf16 o_proj reference path are plain
Triton/torch; the aiter-only preshuffle GEMMs and the fused aiter q/kv
norm+quant self-disable off ROCm (``rocm_aiter_ops.is_enabled()`` and
``is_linear_fp8_enabled()`` are False on CUDA), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.

The SWA backend is inherited unchanged from the ROCm layer: its builder adds
the ragged SWA indices the shared Triton decode kernels consume, and the
generic ``DeepseekSparseSWABackend`` has no compute-capability gate.
"""

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
