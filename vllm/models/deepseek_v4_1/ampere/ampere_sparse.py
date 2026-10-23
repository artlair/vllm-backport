# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/A6000).

Reuses the ROCm Triton sparse-MLA implementation wholesale, exactly like the
V4 Ampere layer: its kernels, ragged metadata builders and bf16 o_proj
reference path are plain Triton/torch (the aiter-only preshuffle GEMMs and the
fused aiter norm+quant self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLAMetadataBuilder
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import KVCacheSpec


def _adaptive_verification_enabled(vllm_config: VllmConfig) -> bool:
    spec_config = vllm_config.speculative_config
    return spec_config is not None and spec_config.enable_adaptive_verification


class DeepseekV41AmpereMLASparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """SM8x compressed-KV metadata, with varlen decode graphs when adaptive.

    Every decode bound this builder emits comes from the device query start
    locations and seq_lens (row map, compressed slot map) into persistent
    buffers whose tails are rewritten each step, so a FULL graph replayed for a
    device-trimmed batch stays correct. That argument only covers the trimmed
    path: the standard path keeps the shared UNIFORM_BATCH contract, which is
    also what every other user of the base builder keeps.
    """

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        if _adaptive_verification_enabled(vllm_config):
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)


class DeepseekV41AmpereSparseSWAMetadataBuilder(
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder
):
    """SM8x sliding-window metadata, with varlen decode graphs when adaptive.

    Same device-driven decode path as the compressed-KV builder: the per-token
    window indices, lens and ragged (indices, indptr) are all recomputed from
    the device query start locations, and padded rows are zeroed or set to -1.
    """

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        if _adaptive_verification_enabled(vllm_config):
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41AmpereMLASparseMetadataBuilder]:
        return DeepseekV41AmpereMLASparseMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereSparseSWABackend(DeepseekV41ROCMAiterSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41AmpereSparseSWAMetadataBuilder]:
        return DeepseekV41AmpereSparseSWAMetadataBuilder


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    swa_backend_cls = DeepseekV41AmpereSparseSWABackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm layer routes wo_a through the ordinary MXFP8 linear kernel
        # (is_bmm=False) because its bf16 einsum dequantizes the raw fp8 weight
        # itself. On CUDA that ordinary list is Marlin, whose repack would turn
        # the weight into packed garbage for the einsum. Keep the grouped-BMM
        # kernel list instead: on SM8x it resolves to the emulation kernel,
        # which dequantizes wo_a to a plain bf16 [g*r, d] weight at load time;
        # _get_cached_wo_a_bf16 then only views it (no second dequant).
        self.wo_a.is_bmm = True
