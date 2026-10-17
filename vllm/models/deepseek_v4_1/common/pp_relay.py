# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel relay of DeepSeek V4.1 cross-layer attention sources.

V4.1 layers share state with earlier "source" layers: the compressed-KV and
index-K caches of ``kv_source_layer_ids``, the top-k indices of
``index_source_layer_ids`` and the candidate blocks of
``candidate_source_layer_id``. When a PP stage boundary separates a source
from its consumers, the upstream stage relays the source layer's attention
input through the intermediate tensors and the downstream stage runs a
"shadow" of the source's compressor and indexer on it, rebuilding the same
caches and buffers locally. The shadow uses the source's layer name, so its
caches join the source's KV cache group and share its block table.
"""

from collections.abc import Iterable

RELAY_KEY_PREFIX = "relay_attn_input."


def relay_key(layer_id: int) -> str:
    """Intermediate-tensor key of a source layer's relayed attention input."""
    return f"{RELAY_KEY_PREFIX}{layer_id}"


def _compress_ratio(config, layer_id: int) -> int:
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_id >= len(compress_ratios):
        return 0
    return int(compress_ratios[layer_id])


def source_reads(config, layer_id: int) -> set[int]:
    """Earlier source layers whose shared state ``layer_id``'s attention reads.

    Mirrors the source resolution in ``DeepseekV4Attention.__init__``: the
    latest kv source and index source at or below the layer, plus the
    candidate source for index sources above it.
    """
    if layer_id >= config.num_hidden_layers or _compress_ratio(config, layer_id) == 0:
        return set()
    kv_sources = getattr(config, "kv_source_layer_ids", None) or ()
    index_sources = getattr(config, "index_source_layer_ids", None) or ()
    if not kv_sources or not index_sources:
        return set()
    reads = {
        max(s for s in kv_sources if s <= layer_id),
        max(s for s in index_sources if s <= layer_id),
    }
    candidate_source = getattr(config, "candidate_source_layer_id", -1)
    if (
        layer_id in index_sources
        and getattr(config, "candidate_topk_blocks", 0) > 0
        and 0 <= candidate_source < layer_id
    ):
        reads.add(candidate_source)
    reads.discard(layer_id)
    return reads


def _sources_below(config, layer_ids: Iterable[int], boundary: int) -> tuple[int, ...]:
    pending = [s for layer_id in layer_ids for s in source_reads(config, layer_id)]
    seen: set[int] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        pending.extend(source_reads(config, source))
    return tuple(sorted(s for s in seen if s < boundary))


def relay_source_ids(config, boundary: int) -> tuple[int, ...]:
    """Sources below ``boundary`` that any layer at or above it depends on.

    These are the attention inputs relayed across a stage boundary at
    ``boundary``, whether the downstream stage shadows them or passes them on.
    """
    return _sources_below(config, range(boundary, config.num_hidden_layers), boundary)


def shadow_source_ids(config, start_layer: int, end_layer: int) -> tuple[int, ...]:
    """Sources below ``start_layer`` that the stage's own layers depend on.

    Returned in ascending order, which is the order the shadows must run in:
    a shadow only reads sources below itself, and the last shadow index
    source is the one whose top-k the stage's leading consumers read.
    """
    return _sources_below(config, range(start_layer, end_layer), start_layer)
