# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

logger = init_logger(__name__)

# Architecture -> the hf_text_config field naming the layers that own Engram
# tables. A model is only configurable here if it actually has such layers.
_ENGRAM_LAYER_FIELDS = {
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
    "DeepseekV41ForCausalLM": "engram_layer_ids",
}


def _default_cpu_offload() -> bool:
    """Honor the legacy environment variable only when the field is omitted."""
    if envs.is_set("VLLM_PLE_CPU_OFFLOAD"):
        logger.warning_once(
            "VLLM_PLE_CPU_OFFLOAD is a legacy setting and may be removed in a "
            "future release. Use --engram-config.cpu_offload instead."
        )
    return envs.VLLM_PLE_CPU_OFFLOAD


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = Field(default_factory=_default_cpu_offload)
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Defaults to False, or VLLM_PLE_CPU_OFFLOAD when set for compatibility.
    An explicit value takes precedence over the legacy environment variable.
    For DeepSeek V4.1 each rank offloads its assigned hash heads, so host
    table storage and lookup traffic scale with the heads assigned to it."""

    gpu_resident_heads: int = Field(default=0, ge=0)
    """With `cpu_offload`, keep the first this-many of each rank's hash heads
    in GPU memory and pin only the rest in host memory. Trades HBM for host
    RAM: for DeepSeek V4.1 Flash one head is 3.93 GiB of each table. Only
    honored by the DeepSeek V4.1 implementation."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica.
    Only honored by the Qwen4Exp PLE implementation."""

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without supported embeddings."""
        from vllm.platforms import current_platform

        field = (
            _ENGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            supported = ", ".join(sorted(_ENGRAM_LAYER_FIELDS))
            layer_field = field or "ple_layer_ids / engram_layer_ids"
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                f"embeddings. Currently only the CUDA implementations of "
                f"{supported} with non-empty {layer_field} are supported."
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def verify_host_memory(self, node_bytes: int, rank_bytes: int) -> None:
        """Refuse to pin more host memory for Engram than this node can hold.

        Offloaded tables are pinned, so they can be neither reclaimed nor
        swapped, and every rank on a node pins its own shard: the node needs
        the sum. Under PP only the stages owning an Engram layer pin anything,
        so `node_bytes` is the sum over the ranks actually on this node.

        Args:
            node_bytes: Pinned Engram bytes of all ranks on this node.
            rank_bytes: Pinned Engram bytes of this rank.

        Raises:
            ValueError: If `node_bytes` exceeds physical RAM or the cgroup
                memory limit.
        """
        if not self.cpu_offload or node_bytes == 0:
            return
        import psutil

        from vllm.utils.cpu_resource_utils import get_cgroup_memory_limit

        limit, source = psutil.virtual_memory().total, "physical RAM"
        hint = (
            "use a host with more memory, or put the Engram layers on "
            "pipeline stages of different nodes"
        )
        cgroup_limit, _ = get_cgroup_memory_limit()
        if cgroup_limit is not None and cgroup_limit < limit:
            limit, source = cgroup_limit, "container memory limit"
            hint = "raise the container memory limit"
        gib = 1 << 30
        if node_bytes > limit:
            raise ValueError(
                f"Engram tables need {node_bytes / gib:.1f} GiB of pinned host "
                f"memory on this node, but the {source} is {limit / gib:.1f} "
                "GiB. Pinned memory cannot be reclaimed or swapped, so this "
                "allocation cannot succeed. Keep the tables in GPU memory with "
                f"--engram-config '{{\"cpu_offload\": false}}', or {hint}."
            )
        log = logger.warning if node_bytes > 0.85 * limit else logger.info
        log(
            "Engram pins %.1f GiB of host memory on this rank and %.1f GiB on "
            "this node, leaving %.1f GiB of the %.1f GiB %s for everything else.",
            rank_bytes / gib,
            node_bytes / gib,
            (limit - node_bytes) / gib,
            limit / gib,
            source,
        )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
