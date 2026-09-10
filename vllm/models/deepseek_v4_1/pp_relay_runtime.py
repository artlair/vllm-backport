# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 pp-relay: runtime side of the V4.1 shared-cache relay.

``DeepseekV41PPRelay`` is owned by ``DeepseekV4Model`` when
``VLLM_DSV41_PP_KV_RELAY=1`` and PP > 1. It builds this stage's mirrors, owns
the send buffers, fills the receive side of ``make_empty_intermediate_tensors``
and provides the three forward hooks (``consume`` before the first local layer,
``after_layer`` snapshots, ``outgoing`` for the returned tensor dict). The
planning behind it lives in ``pp_relay.py``; the rationale in
``docs/dsv41-pp-kv-relay.md``.
"""

from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v4_1.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
    _indexer_k_cache_head_dim,
    _resolve_dsv4_kv_cache_dtype,
    compressed_kv_cache_spec,
)
from vllm.models.deepseek_v4_1.common.ops import indexer_k_norm_rope_store
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v4_1.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4_1.pp_relay import (
    CAND,
    LATENT,
    TOPK,
    RelayPayload,
    StageRelayPlan,
    V41Topology,
    plan_pp_kv_relay,
    stage_bounds,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


class DeepseekV41RelayCompressedCache(nn.Module, AttentionLayerBase):
    """Stand-in for a kv source's attention layer on a stage without it.

    Registered in the static forward context under the source's own prefix
    (``<layers.N>.attn``) with the source's spec and backend, so the KV-cache
    planner, the metadata builders and ``bind_kv_cache`` treat it exactly like
    the source; consumers resolve it through ``_compressed_kv_cache``.
    """

    def __init__(
        self,
        prefix: str,
        backend_cls: type[AttentionBackend],
        head_dim: int,
        compress_ratio: int,
        kv_cache_dtype: str,
        kv_cache_torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.backend_cls = backend_cls
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = kv_cache_torch_dtype
        self.kv_cache = torch.tensor([])
        static_forward_context = (
            get_current_vllm_config().compilation_config.static_forward_context
        )
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C], as DeepseekV4Attention does.
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return compressed_kv_cache_spec(
            vllm_config,
            self.head_dim,
            self.compress_ratio,
            self.kv_cache_dtype,
            self.kv_cache_torch_dtype,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def forward(self): ...


class DeepseekV41PPRelayMirror(nn.Module):
    """Rebuilds one kv source's shared caches from its relayed latent.

    Runs the same two kernels the source runs after its compressor:
    ``rope_quant_insert`` into the mirrored compressed-KV cache and, when a
    local non-kv index source borrows the K cache, ``wk`` + k_norm + RoPE +
    quant into the mirrored indexer K cache. Slots come from this rank's own
    attention metadata for the mirrored layer names.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_cls: type[DeepseekV4Attention],
        kv_source: int,
        attn_prefix: str,
        need_k_cache: bool,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        assert cache_config is not None
        self.kv_source = kv_source
        self.compress_ratio = int(config.compress_ratios[kv_source])
        self.head_dim = int(config.head_dim)
        # Same resolution as the source layer (writes fp8_ds_mla back onto
        # cache_config exactly as the source would on its own rank).
        kv_cache_dtype, kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            attn_cls.use_fp8_ds_mla_layout, cache_config.cache_dtype, cache_config
        )
        if kv_cache_torch_dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                "VLLM_DSV41_PP_KV_RELAY cannot mirror plain per-tensor fp8 "
                "compressed-KV rows: the insert needs the source layer's "
                "FlashInfer kv scale. Use the fp8_ds_mla or bf16 layouts."
            )
        self.compressed_cache = DeepseekV41RelayCompressedCache(
            attn_prefix,
            attn_cls.backend_cls,
            self.head_dim,
            self.compress_ratio,
            kv_cache_dtype,
            kv_cache_torch_dtype,
        )
        # The source's compress RoPE (positions are rounded to the group).
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.k_cache: DeepseekV4IndexerCache | None = None
        if need_k_cache:
            self.use_fp4_kv = dsa_indexer_uses_fp4(vllm_config)
            self.k_cache = DeepseekV4IndexerCache(
                head_dim=_indexer_k_cache_head_dim(
                    config.index_head_dim, self.use_fp4_kv
                ),
                dtype=torch.uint8,
                prefix=f"{attn_prefix}.indexer.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
            # The source indexer's wk / k_norm (bf16, unquantized in the
            # checkpoint); loaded through DeepseekV4Model's name redirect.
            self.wk = ReplicatedLinear(
                self.head_dim,
                config.index_head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{attn_prefix}.pp_relay_mirror.wk",
            )
            self.k_norm = RMSNorm(config.index_head_dim, config.rms_norm_eps)

    def checkpoint_params(self) -> Iterator[tuple[str, str]]:
        """(checkpoint name relative to the model, param name relative to this
        mirror) for every weight the mirror loads."""
        if self.k_cache is None:
            return
        source_prefix = f"layers.{self.kv_source}.attn.indexer"
        for name, _ in self.named_parameters():
            yield f"{source_prefix}.{name}", name

    def write(self, latent: torch.Tensor, positions: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile run: caches are not bound and the source emitted nothing.
            return
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        kv_metadata = cast(Any, attn_metadata[self.compressed_cache.prefix])
        rope_quant_insert(
            latent,
            positions,
            cos_sin_cache,
            self.compressed_cache.kv_cache,
            kv_metadata.slot_mapping,
            self.compress_ratio,
        )
        if self.k_cache is None:
            return
        k_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        # ReplicatedLinear returns (output, bias); bias is None.
        k_pre, _ = self.wk(latent)
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            cos_sin_cache,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            self.k_cache.kv_cache,
            k_metadata.slot_mapping,
            self.compress_ratio,
            self.use_fp4_kv,
        )


class DeepseekV41PPRelay(nn.Module):
    """This PP stage's share of the relay: plan, mirrors, buffers, hooks."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_cls: type[DeepseekV4Attention],
        layers_prefix: str,
        topk_indices_buffer: torch.Tensor,
        candidate_block_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.topology = V41Topology.from_hf_config(config)
        pp_group = get_pp_group()
        bounds = stage_bounds(config.num_hidden_layers, pp_group.world_size)
        plans = plan_pp_kv_relay(self.topology, bounds)
        self.plan: StageRelayPlan = plans[pp_group.rank_in_group]
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer

        self.mirrors = nn.ModuleDict(
            {
                str(mirror.kv_source): DeepseekV41PPRelayMirror(
                    vllm_config,
                    attn_cls,
                    mirror.kv_source,
                    f"{layers_prefix}.{mirror.kv_source}.attn",
                    mirror.need_k_cache,
                )
                for mirror in self.plan.mirrors
            }
        )

        # Send buffers for the payloads produced on this stage; forwarded
        # payloads reuse the runner-owned receive buffers. Plain attributes
        # (not registered buffers) like the model's topk_indices_buffer.
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.send_buffers: dict[str, torch.Tensor] = {
            payload.key: self._new_buffer(payload, max_tokens, device=None)
            for payload in self.plan.produced
        }
        # Layer id -> payloads snapshotted right after that layer runs.
        self.snapshots: dict[int, list[RelayPayload]] = {}
        for payload in self.plan.produced:
            if payload.kind == LATENT:
                continue
            self.snapshots.setdefault(payload.source, []).append(payload)
        for payload in (*self.plan.recv, *self.plan.send):
            if payload.kind == TOPK:
                assert topk_indices_buffer.shape[1] == payload.row_width(self.topology)
            elif payload.kind == CAND:
                assert candidate_block_buffer is not None
                assert candidate_block_buffer.shape[1] == payload.row_width(
                    self.topology
                )

        logger.info(
            "dsv41 pp-relay: rank %d layers [%d, %d) recv=%s (%d B/token) "
            "send=%s (%d B/token) mirrors=%s",
            self.plan.pp_rank,
            self.plan.start_layer,
            self.plan.end_layer,
            [p.key for p in self.plan.recv],
            self.plan.recv_bytes_per_token(self.topology),
            [p.key for p in self.plan.send],
            self.plan.send_bytes_per_token(self.topology),
            [(m.kv_source, m.need_k_cache) for m in self.plan.mirrors],
        )

    def _new_buffer(
        self, payload: RelayPayload, rows: int, device: torch.device | None
    ) -> torch.Tensor:
        shape = (rows, payload.row_width(self.topology))
        if payload.kind == LATENT:
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        # -1 is the "no token" / "no block" sentinel of both index buffers, so
        # dummy runs on receiving ranks see empty rows rather than index 0.
        return torch.full(shape, -1, dtype=torch.int32, device=device)

    def attach_sources(self, layers: nn.ModuleList) -> None:
        """Point local kv sources at their latent send buffers."""
        for payload in self.plan.produced:
            if payload.kind == LATENT:
                # The offloader may wrap the decoder layer; its attn is still
                # the kv-source DeepseekV4Attention.
                attn = layers[payload.source].attn
                assert getattr(attn, "is_kv_source", False), payload
                attn.pp_relay_latent_out = self.send_buffers[payload.key]

    def checkpoint_params(self, relay_attr: str) -> dict[str, str]:
        """Checkpoint name (model-relative) -> model-relative param name."""
        redirect = {}
        for kv_source, mirror in self.mirrors.items():
            for ckpt_name, param_name in mirror.checkpoint_params():
                redirect[ckpt_name] = f"{relay_attr}.mirrors.{kv_source}.{param_name}"
        return redirect

    def make_empty_tensors(
        self, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            payload.key: self._new_buffer(payload, batch_size, device)
            for payload in self.plan.recv
        }

    @property
    def all_gather_tensors(self) -> dict[str, bool]:
        """Per-key override of the PP send's TP all-gather split.

        The split assumes a tensor is replicated across TP ranks. Latents
        and top-k rows always are; candidate blocks are row-sharded across
        TP when the indexer query shard is on, so they must travel whole.
        """
        if not envs.VLLM_INDEXER_QUERY_SHARD:
            return {}
        return {
            payload.key: False
            for payload in (*self.plan.recv, *self.plan.send)
            if payload.kind == CAND
        }

    def consume(
        self, intermediate_tensors: IntermediateTensors, positions: torch.Tensor
    ) -> None:
        """Rebuild the relayed state before the first local layer runs."""
        for payload in self.plan.consumed:
            tensor = intermediate_tensors[payload.key]
            num_tokens = tensor.shape[0]
            if payload.kind == LATENT:
                self.mirrors[str(payload.source)].write(tensor, positions)
            elif payload.kind == TOPK:
                self.topk_indices_buffer[:num_tokens].copy_(tensor)
            else:
                assert self.candidate_block_buffer is not None
                self.candidate_block_buffer[:num_tokens].copy_(tensor)

    def after_layer(self, layer_id: int, num_tokens: int) -> None:
        """Snapshot the rows an index / candidate source just published,
        before a later local index source overwrites the shared buffer."""
        for payload in self.snapshots.get(layer_id, ()):
            if payload.kind == TOPK:
                src = self.topk_indices_buffer
            else:
                assert self.candidate_block_buffer is not None
                src = self.candidate_block_buffer
            self.send_buffers[payload.key][:num_tokens].copy_(src[:num_tokens])

    def outgoing(
        self, intermediate_tensors: IntermediateTensors | None, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        """Relay entries of the outgoing IntermediateTensors: produced payloads
        from the send buffers, forwarded ones as received (already ``[:T]``)."""
        out: dict[str, torch.Tensor] = {}
        for payload in self.plan.send:
            if payload in self.plan.produced:
                out[payload.key] = self.send_buffers[payload.key][:num_tokens]
            else:
                assert intermediate_tensors is not None
                out[payload.key] = intermediate_tensors[payload.key]
        return out
