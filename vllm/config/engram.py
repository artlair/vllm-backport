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

    # dsv41 engram-warm: boot-time page-cache warmup of the mmap slices. The
    # page cache starts cold after every boot (the weight stream evicts it)
    # and a cold row is one NVMe read; with DSpark's rejected drafts hashing
    # to never-seen n-grams the first benches decayed 10x while pages warmed.
    # "sync" blocks weight loading until the slices are cached, "async" reads
    # them in a background thread while the runner profiles and captures.
    # Ignored unless the resolved table mode is "mmap"; a dummy load (anonymous
    # tables) skips it with a log line.
    mmap_warm: Literal["none", "async", "sync"] = "none"
    """Read each rank's mmap slices of both tables into the page cache after
    the weights load: "none", "async" (background thread) or "sync" (block
    until cached). Only meaningful with `table_mode = "mmap"`."""

    # dsv41 engram-warm: the weight stream leaves the shard pages it read in
    # the page cache (~110 GiB per x299 host) although nothing reads them
    # again (the weights are in VRAM); they compete with the mmap tables for
    # the cache and evicted a freshly warmed table on x299 (45% resident
    # after boot). When on, the safetensors loader drops the page cache of
    # every shard it streamed (posix_fadvise DONTNEED) once the weights are
    # loaded, never a shard holding an mmap engram slice. None follows the
    # table mode: on for "mmap", off otherwise, so other models are untouched.
    drop_weight_pages: bool | None = None
    """Drop the page cache of the streamed weight shards after loading
    (never the shards backing mmap engram slices): True, False, or None
    (True with `table_mode = "mmap"`, False otherwise)."""

    @property
    def resolved_table_mode(self) -> str:
        """`table_mode` with "auto" folded into the `cpu_offload` choice."""
        if self.table_mode == "auto":
            return "pinned" if self.cpu_offload else "resident"
        return self.table_mode

    @property
    def resolved_drop_weight_pages(self) -> bool:
        """dsv41 engram-warm: `drop_weight_pages` with None folded into the
        table mode (drop with "mmap" tables, keep otherwise)."""
        if self.drop_weight_pages is None:
            return self.resolved_table_mode == "mmap"
        return self.drop_weight_pages

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
        # dsv41 engram-warm: the warmup and the page-cache drop change no
        # computation.
        return hash_factors(get_hash_factors(self, {"mmap_warm", "drop_weight_pages"}))
