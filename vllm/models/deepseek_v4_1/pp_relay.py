# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 pp-relay: planning for relaying V4.1 shared caches across PP stages.

DeepSeek V4.1 shares three things across layers: a kv source's compressed-KV
and indexer K caches (read by the layers up to the next kv source), an index
source's top-k rows (read by the layers up to the next index source) and the
candidate source's candidate blocks (read by every later index source). When
``VLLM_PP_LAYER_PARTITION`` places a consumer on a later stage than its
source, the source stage ships the per-token rows it produced this step in
its ``IntermediateTensors`` and the receiving stage rebuilds the shared state
locally (see ``docs/dsv41-pp-kv-relay.md``).

This module is deliberately free of torch / kernel imports: it only decides,
per stage, which payloads are received, sent, consumed and mirrored, so the
plan can be unit-tested on a CPU and so both ends of every hop derive their
tensor dict from the same function.
"""

from dataclasses import dataclass
from typing import Any

# Payload kinds, in the order their keys are emitted.
LATENT = "latent"  # kv source's bf16 compressor latent, [T, head_dim]
TOPK = "topk"  # index source's top-k rows, [T, index_topk] int32
CAND = "cand"  # candidate source's candidate block ids, [T, topk_blocks] int32
_KIND_ORDER = {LATENT: 0, TOPK: 1, CAND: 2}


@dataclass(frozen=True)
class V41Topology:
    """The parts of the V4.1 config that decide who shares what."""

    num_hidden_layers: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    index_topk: int = 0
    head_dim: int = 512

    @classmethod
    def from_hf_config(cls, config: Any) -> "V41Topology":
        ratios = tuple(int(r) for r in (getattr(config, "compress_ratios", None) or ()))
        return cls(
            num_hidden_layers=int(config.num_hidden_layers),
            compress_ratios=ratios,
            kv_source_layer_ids=tuple(
                int(s) for s in (getattr(config, "kv_source_layer_ids", None) or ())
            ),
            index_source_layer_ids=tuple(
                int(s) for s in (getattr(config, "index_source_layer_ids", None) or ())
            ),
            candidate_source_layer_id=int(
                getattr(config, "candidate_source_layer_id", -1)
            ),
            candidate_topk_blocks=int(getattr(config, "candidate_topk_blocks", 0)),
            index_topk=int(getattr(config, "index_topk", 0)),
            head_dim=int(getattr(config, "head_dim", 512)),
        )

    def compress_ratio(self, layer_id: int) -> int:
        # Mirrors DeepseekV4Attention: layers past the list are pure SWA.
        if layer_id < len(self.compress_ratios):
            return self.compress_ratios[layer_id]
        return 0

    def is_backbone(self, layer_id: int) -> bool:
        return layer_id < self.num_hidden_layers

    def is_kv_source(self, layer_id: int) -> bool:
        return self.is_backbone(layer_id) and layer_id in self.kv_source_layer_ids

    def is_index_source(self, layer_id: int) -> bool:
        return self.is_backbone(layer_id) and layer_id in self.index_source_layer_ids

    def kv_source_of(self, layer_id: int) -> int | None:
        """The kv source whose compressed cache ``layer_id`` reads (self for
        kv sources); None for sliding-window-only layers."""
        if self.compress_ratio(layer_id) <= 0:
            return None
        return max(s for s in self.kv_source_layer_ids if s <= layer_id)

    def index_source_of(self, layer_id: int) -> int | None:
        if self.compress_ratio(layer_id) <= 0:
            return None
        return max(s for s in self.index_source_layer_ids if s <= layer_id)

    def borrows_k_cache(self, layer_id: int) -> bool:
        """Non-kv index sources read the kv source's indexer K cache."""
        return self.is_index_source(layer_id) and not self.is_kv_source(layer_id)

    def uses_candidates(self, layer_id: int) -> bool:
        return (
            self.candidate_topk_blocks > 0
            and self.is_index_source(layer_id)
            and 0 <= self.candidate_source_layer_id < layer_id
        )


@dataclass(frozen=True, order=True)
class RelayPayload:
    """One relayed tensor: ``kind`` produced by backbone layer ``source``."""

    kind: str
    source: int

    @property
    def key(self) -> str:
        return f"dsv41_relay_{self.kind}_{self.source}"

    def row_width(self, topology: V41Topology) -> int:
        """Columns of the ``[T, width]`` payload."""
        if self.kind == LATENT:
            return topology.head_dim
        if self.kind == TOPK:
            return topology.index_topk
        if self.kind == CAND:
            return topology.candidate_topk_blocks
        raise ValueError(f"unknown relay payload kind {self.kind!r}")

    def row_bytes(self, topology: V41Topology) -> int:
        itemsize = 2 if self.kind == LATENT else 4
        return self.row_width(topology) * itemsize


@dataclass(frozen=True)
class MirrorSpec:
    """A kv source whose caches a stage must register and rebuild locally."""

    kv_source: int
    need_k_cache: bool


@dataclass(frozen=True)
class StageRelayPlan:
    pp_rank: int
    start_layer: int
    end_layer: int
    # Payloads received from the previous stage (provider stage < rank <= last
    # consumer stage), in key order.
    recv: tuple[RelayPayload, ...]
    # Payloads sent to the next stage: exactly the next stage's ``recv``.
    send: tuple[RelayPayload, ...]
    # Subset of ``recv`` that has a consumer on this stage (the rest is only
    # forwarded).
    consumed: tuple[RelayPayload, ...]
    # Subset of ``send`` whose source layer lives on this stage (the rest is
    # forwarded from ``recv``).
    produced: tuple[RelayPayload, ...]
    mirrors: tuple[MirrorSpec, ...]

    @property
    def is_active(self) -> bool:
        return bool(self.recv or self.send)

    def recv_bytes_per_token(self, topology: V41Topology) -> int:
        return sum(p.row_bytes(topology) for p in self.recv)

    def send_bytes_per_token(self, topology: V41Topology) -> int:
        return sum(p.row_bytes(topology) for p in self.send)


def stage_bounds_from_partition(partition: list[int]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for size in partition:
        bounds.append((start, start + size))
        start += size
    return bounds


def stage_bounds(num_hidden_layers: int, pp_size: int) -> list[tuple[int, int]]:
    """The ``(start, end)`` every rank's ``make_layers`` will use (honours
    ``VLLM_PP_LAYER_PARTITION`` exactly like the model does)."""
    from vllm.distributed.utils import get_pp_indices

    return [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]


def _payload_consumers(topology: V41Topology, payload: RelayPayload) -> tuple[int, ...]:
    """Backbone layers that read ``payload`` and are not its source."""
    layers = range(topology.num_hidden_layers)
    if payload.kind == LATENT:
        return tuple(
            layer
            for layer in layers
            if layer != payload.source
            and topology.kv_source_of(layer) == payload.source
        )
    if payload.kind == TOPK:
        # Index sources compute their own top-k; only the layers in between
        # read the published rows.
        return tuple(
            layer
            for layer in layers
            if layer != payload.source
            and not topology.is_index_source(layer)
            and topology.index_source_of(layer) == payload.source
        )
    if payload.kind == CAND:
        return tuple(layer for layer in layers if topology.uses_candidates(layer))
    raise ValueError(f"unknown relay payload kind {payload.kind!r}")


def _all_payloads(topology: V41Topology) -> list[RelayPayload]:
    payloads = [RelayPayload(LATENT, s) for s in topology.kv_source_layer_ids]
    payloads += [RelayPayload(TOPK, s) for s in topology.index_source_layer_ids]
    if topology.candidate_topk_blocks > 0 and topology.candidate_source_layer_id >= 0:
        payloads.append(RelayPayload(CAND, topology.candidate_source_layer_id))
    return sorted(payloads, key=lambda p: (_KIND_ORDER[p.kind], p.source))


def plan_pp_kv_relay(
    topology: V41Topology, bounds: list[tuple[int, int]]
) -> list[StageRelayPlan]:
    """Per-stage relay plan for the given contiguous stage bounds.

    ``bounds[r] = (start, end)`` of the backbone layers on rank ``r``; they
    must tile ``[0, num_hidden_layers)`` in order (what ``get_pp_indices``
    produces for any valid ``VLLM_PP_LAYER_PARTITION``).
    """
    expected = 0
    for start, end in bounds:
        if start != expected or end < start:
            raise ValueError(f"stage bounds {bounds} do not tile the layers in order")
        expected = end
    if expected != topology.num_hidden_layers:
        raise ValueError(
            f"stage bounds {bounds} cover {expected} layers, the model has "
            f"{topology.num_hidden_layers}"
        )
    pp_size = len(bounds)

    def stage_of(layer_id: int) -> int:
        for r, (start, end) in enumerate(bounds):
            if start <= layer_id < end:
                return r
        raise ValueError(f"layer {layer_id} is on no stage")

    recv: list[list[RelayPayload]] = [[] for _ in range(pp_size)]
    consumed: list[list[RelayPayload]] = [[] for _ in range(pp_size)]
    produced: list[list[RelayPayload]] = [[] for _ in range(pp_size)]
    for payload in _all_payloads(topology):
        consumers = _payload_consumers(topology, payload)
        if not consumers:
            continue
        provider = stage_of(payload.source)
        consumer_stages = {stage_of(layer) for layer in consumers}
        last = max(consumer_stages)
        if last <= provider:
            continue
        for r in range(provider + 1, last + 1):
            recv[r].append(payload)
            if r in consumer_stages:
                consumed[r].append(payload)
        produced[provider].append(payload)

    plans = []
    for r, (start, end) in enumerate(bounds):
        send = tuple(recv[r + 1]) if r + 1 < pp_size else ()
        mirrors = []
        for payload in consumed[r]:
            if payload.kind != LATENT:
                continue
            need_k_cache = any(
                topology.borrows_k_cache(layer)
                and topology.kv_source_of(layer) == payload.source
                for layer in range(start, end)
            )
            mirrors.append(MirrorSpec(payload.source, need_k_cache))
        plans.append(
            StageRelayPlan(
                pp_rank=r,
                start_layer=start,
                end_layer=end,
                recv=tuple(recv[r]),
                send=send,
                consumed=tuple(consumed[r]),
                produced=tuple(p for p in produced[r] if p in send),
                mirrors=tuple(mirrors),
            )
        )
    return plans
