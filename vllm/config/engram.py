# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
}


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Each rank offloads its assigned hash heads, so host table storage and
    lookup traffic scale with the number of heads assigned to the rank."""

    # dsv41 engram-mmap: third storage mode. "auto" keeps the fork behaviour
    # (cpu_offload picks pinned host memory or HBM). "mmap" maps each rank's
    # row shard straight from the safetensors shard file (read-only,
    # MAP_SHARED, page-cache backed, MADV_RANDOM), gathers the fp8 rows plus
    # their ue8m0 scales on the host inside an eager cudagraph break, and
    # dequantises them on the GPU with the same kernel as the other modes.
    # No pinned memory, so the 94 GiB real tables fit hosts that cannot pin
    # them; see docs/dsv41-engram-mmap.md.
    table_mode: Literal["auto", "pinned", "resident", "mmap"] = "auto"
    """Where the n-gram tables live: "pinned" (host, UVA reads), "resident"
    (HBM), "mmap" (host page cache, gathered on the CPU) or "auto", which
    follows `cpu_offload` (True = pinned, False = resident)."""

    @property
    def resolved_table_mode(self) -> str:
        """`table_mode` with "auto" folded into the `cpu_offload` choice."""
        if self.table_mode == "auto":
            return "pinned" if self.cpu_offload else "resident"
        return self.table_mode

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings. Currently only the CUDA DeepSeek V4.1 "
                f"implementation with non-empty {field or 'engram_layer_ids'} "
                "is supported."
            )

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
