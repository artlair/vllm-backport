# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41: KV-cache grouping for the DeepSeek V4.1 layer mix on sm8x.

V4.1 registers, per decoder layer, a sliding-window MLA cache (every layer,
block 32) plus, on ``kv_source_layer_ids`` only, a compressed-KV
``MLAAttentionSpec`` (block 128, ``tokens_per_state`` = compress ratio), an
indexer K cache (same block, 132B rows) and, for ratio > 1, a
``CircularBufferSpec`` compressor ring that is exactly one block per request.
The DSpark draft layers are pure sliding window. These tests build that mix
from bare specs (no model, no GPU) and check the packed planner places every
layer, keeps the rings in a one-block-per-request group and leaves the
scheduler block sizes sane.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    group_and_unify_kv_cache_specs,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)

# sparse_mla.py: 64 on Hopper, 128 elsewhere (sm8x).
MLA_BLOCK = 128
# attention.py hardcodes the SWA cache block.
SWA_BLOCK = 32
WINDOW = 128
# compressor.py: rows_per_step = num_speculative_tokens + 2 = 5 -> 8.
RING_BLOCK = 8
HEAD_DIM = 512
INDEXER_ROW_BYTES = 132  # 128 fp8 + 4B fp32 scale
RING_STATE_DIM = 2 * HEAD_DIM  # kv_state + score_state
# fp8_ds_mla rows: 448B NoPE + 128B RoPE + 8B scale, 576B page alignment.
DS_MLA_ROW_BYTES = 584
DS_MLA_ALIGNMENT = 576

MAX_MODEL_LEN = 4096


def mla_kv_spec(ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=MLA_BLOCK,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.uint8,
        tokens_per_state=ratio,
        cache_dtype_str="fp8_ds_mla",
        alignment=DS_MLA_ALIGNMENT,
        model_version="deepseek_v4",
        kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
        state_content_bytes=DS_MLA_ROW_BYTES,
    )


def indexer_spec(ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=MLA_BLOCK,
        num_kv_heads=1,
        head_size=INDEXER_ROW_BYTES,
        dtype=torch.uint8,
        tokens_per_state=ratio,
        alignment=DS_MLA_ALIGNMENT,
    )


def swa_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=SWA_BLOCK,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.uint8,
        sliding_window=WINDOW,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=DS_MLA_ROW_BYTES,
        alignment=DS_MLA_ALIGNMENT,
        model_version="deepseek_v4",
        kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
    )


def ring_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=RING_BLOCK,
        num_kv_heads=1,
        head_size=RING_STATE_DIM,
        head_size_v=0,
        dtype=torch.float32,
    )


def v41_specs(
    compress_ratios: list[int],
    kv_source_layer_ids: list[int],
    index_source_layer_ids: list[int],
    num_draft_layers: int,
) -> dict[str, KVCacheSpec]:
    """Specs in static_forward_context registration order.

    Mirrors ``DeepseekV4Attention.__init__``: the attention layer registers
    itself first (only kv sources own a compressed cache), then the indexer K
    cache (only kv sources that are also index sources), then the SWA cache,
    then the compressor state ring (kv sources with ratio > 1). Draft layers
    past ``compress_ratios`` are pure sliding window.
    """
    specs: dict[str, KVCacheSpec] = {}
    num_backbone = len(compress_ratios)
    for layer_id in range(num_backbone + num_draft_layers):
        prefix = f"model.layers.{layer_id}.attn"
        ratio = compress_ratios[layer_id] if layer_id < num_backbone else 0
        is_kv_source = layer_id < num_backbone and layer_id in kv_source_layer_ids
        is_index_source = layer_id < num_backbone and layer_id in index_source_layer_ids
        if is_kv_source:
            specs[prefix] = mla_kv_spec(ratio)
        if is_index_source and is_kv_source:
            specs[f"{prefix}.indexer.k_cache"] = indexer_spec(ratio)
        specs[f"{prefix}.swa_cache"] = swa_spec()
        if is_kv_source and ratio > 1:
            specs[f"{prefix}.compressor.state_cache"] = ring_spec()
    return specs


def real_flash_specs() -> dict[str, KVCacheSpec]:
    """The shipped DeepSeek-V4.1-Flash topology with 3 DSpark draft layers."""
    return v41_specs(
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        num_draft_layers=3,
    )


def small_specs() -> dict[str, KVCacheSpec]:
    """A handful of layers covering every V4.1 cache kind."""
    return v41_specs(
        # 0/1 pure SWA, 2 and 4 ratio-2 kv+index sources, 3 a ratio-2
        # consumer, 5 a ratio-1 kv+index source, 6 a ratio-1 consumer that is
        # also a non-owning index source.
        compress_ratios=[0, 0, 2, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4, 5],
        index_source_layer_ids=[2, 4, 5, 6],
        num_draft_layers=1,
    )


def make_vllm_config(layout: str = "BLHNC", use_eagle: bool = True):
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = layout
    cache_config.num_gpu_blocks_override = None
    cache_config.enable_prefix_caching = True
    speculative_config = (
        SimpleNamespace(
            method="dspark",
            num_speculative_tokens=3,
            use_eagle=lambda: True,
            use_multi_module_mtp=lambda: False,
        )
        if use_eagle
        else None
    )
    return SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=speculative_config,
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN, original_max_model_len=MAX_MODEL_LEN
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None,
        max_in_flight_tokens=MLA_BLOCK,
    )


def layer_specs_of(group: KVCacheGroupSpec) -> list[KVCacheSpec]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def ring_groups_of(groups: list[KVCacheGroupSpec]) -> list[KVCacheGroupSpec]:
    return [
        g
        for g in groups
        if all(isinstance(s, CircularBufferSpec) for s in layer_specs_of(g))
    ]


def test_page_sizes_match_the_sm8x_layout():
    # The concrete per-layer page sizes the real config yields on sm8x.
    assert mla_kv_spec(2).page_size_bytes == 37440  # 64 states * 584B -> 576B
    assert mla_kv_spec(1).page_size_bytes == 74880  # 128 states * 584B
    assert indexer_spec(2).page_size_bytes == 8640  # 64 * 132B
    assert indexer_spec(1).page_size_bytes == 17280  # 128 * 132B
    assert swa_spec().page_size_bytes == 19008  # 32 * 584B
    assert ring_spec().page_size_bytes == 32768  # 8 * 1024 * fp32

    specs = real_flash_specs()
    counts: dict[tuple[str, int], int] = {}
    for spec in specs.values():
        key = (type(spec).__name__, spec.page_size_bytes)
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("SlidingWindowMLASpec", 19008): 43,
        ("MLAAttentionSpec", 37440): 3,
        ("MLAAttentionSpec", 74880): 1,
        ("MLAAttentionSpec", 8640): 3,
        ("MLAAttentionSpec", 17280): 1,
        ("CircularBufferSpec", 32768): 3,
    }


@pytest.mark.parametrize("specs_fn", [small_specs, real_flash_specs])
def test_groups_cover_every_layer_and_rings_get_one_block(specs_fn):
    vllm_config = make_vllm_config()
    specs = specs_fn()
    groups = get_kv_cache_groups(vllm_config, specs)

    assigned = [name for g in groups for name in g.layer_names]
    assert sorted(assigned) == sorted(specs)
    assert len(assigned) == len(set(assigned))

    ring_groups = ring_groups_of(groups)
    assert len(ring_groups) == 1
    ring_group = ring_groups[0]
    assert sorted(ring_group.layer_names) == sorted(
        name for name, s in specs.items() if isinstance(s, CircularBufferSpec)
    )
    assert (
        ring_group.kv_cache_spec.max_num_blocks_per_req(vllm_config, MAX_MODEL_LEN) == 1
    )
    assert not ring_group.kv_cache_spec.participates_in_prefix_caching
    assert ring_group.kv_cache_spec.block_size == RING_BLOCK

    # No group mixes rings with anything else, and no group mixes MLA with SWA.
    for g in groups:
        kinds = {type(s) for s in layer_specs_of(g)}
        assert len(kinds) == 1, kinds

    # The ring group never widens the block: the MLA group is the anchor.
    mla_group = next(
        g for g in groups if all(type(s) is MLAAttentionSpec for s in layer_specs_of(g))
    )
    assert (
        _get_kv_cache_bytes_per_block(groups) == mla_group.kv_cache_spec.page_size_bytes
    )
    assert ring_group.kv_cache_spec.page_size_bytes <= _get_kv_cache_bytes_per_block(
        groups
    )


def test_real_flash_group_plan():
    vllm_config = make_vllm_config()
    specs = real_flash_specs()
    groups = get_kv_cache_groups(vllm_config, specs)

    mla_groups = [
        g for g in groups if all(type(s) is MLAAttentionSpec for s in layer_specs_of(g))
    ]
    swa_groups = [
        g
        for g in groups
        if all(isinstance(s, SlidingWindowMLASpec) for s in layer_specs_of(g))
    ]
    ring_groups = ring_groups_of(groups)
    assert len(groups) == len(mla_groups) + len(swa_groups) + len(ring_groups)

    # 4 compressed-KV + 4 indexer caches in one MLA group (block 128).
    assert len(mla_groups) == 1
    assert len(mla_groups[0].layer_names) == 8
    assert mla_groups[0].kv_cache_spec.block_size == MLA_BLOCK
    bytes_per_block = 3 * 37440 + 74880 + 3 * 8640 + 17280
    assert mla_groups[0].kv_cache_spec.page_size_bytes == bytes_per_block
    assert _get_kv_cache_bytes_per_block(groups) == bytes_per_block

    # 43 SWA caches are a state bucket: split only as far as the anchor block
    # already fits (230400 // 19008 = 12 per group -> 4 groups), not widened.
    assert len(swa_groups) == 4
    assert sorted(len(g.layer_names) for g in swa_groups) == [10, 11, 11, 11]
    for g in swa_groups:
        assert g.kv_cache_spec.page_size_bytes <= bytes_per_block
        assert g.kv_cache_spec.block_size == SWA_BLOCK

    # 3 rings fit one block (230400 // 32768 = 7 >= 3) -> a single ring group.
    assert len(ring_groups) == 1
    assert len(ring_groups[0].layer_names) == 3

    # DSpark: the draft attention layer is the last registered layer and its
    # (SWA) group is flagged as the eagle group; nothing else is.
    last_layer = next(reversed(specs))
    eagle = [g for g in groups if g.is_eagle_group]
    assert len(eagle) == 1
    assert last_layer in eagle[0].layer_names
    assert eagle[0] in swa_groups


def test_no_eagle_flag_without_speculation():
    groups = get_kv_cache_groups(make_vllm_config(use_eagle=False), small_specs())
    assert not any(g.is_eagle_group for g in groups)


def test_kv_cache_config_and_scheduler_view():
    vllm_config = make_vllm_config()
    specs = real_flash_specs()
    groups = get_kv_cache_groups(vllm_config, specs)
    bytes_per_block = _get_kv_cache_bytes_per_block(groups)
    num_blocks = 64
    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config, groups, bytes_per_block * num_blocks
    )
    assert kv_cache_config.num_blocks == num_blocks
    tensor_layers = [
        name for t in kv_cache_config.kv_cache_tensors for name in t.layers
    ]
    assert sorted(tensor_layers) == sorted(specs)
    # Groups overlay from byte 0 and every block is strided by the anchor.
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size == bytes_per_block * num_blocks
        assert tensor.block_stride == bytes_per_block

    # The scheduler sees one representative spec per group: the ring group
    # becomes a bare CircularBufferSpec, which maps to CircularBufferManager
    # and reports one block per request.
    scheduler_config = generate_scheduler_kv_cache_config([kv_cache_config])
    ring_specs = [
        g.kv_cache_spec
        for g in scheduler_config.kv_cache_groups
        if isinstance(g.kv_cache_spec, CircularBufferSpec)
    ]
    assert len(ring_specs) == 1
    assert ring_specs[0].max_num_blocks_per_req(vllm_config, MAX_MODEL_LEN) == 1
    assert not ring_specs[0].participates_in_prefix_caching

    # Block sizes: LCM(128, 32, 8) = 128 for the scheduler; the ring opts out of
    # hashing, so hash_block_size = gcd(128, 32) = 32 rather than 8.
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        scheduler_config, vllm_config
    )
    assert scheduler_block_size == MLA_BLOCK
    assert hash_block_size == SWA_BLOCK


def test_tuple_packer_refuses_rings():
    with pytest.raises(AssertionError, match="_get_packed_kv_cache_groups"):
        group_and_unify_kv_cache_specs(small_specs())


def test_layer_outermost_layout_is_rejected():
    with pytest.raises(NotImplementedError, match="block-outermost"):
        get_kv_cache_groups(make_vllm_config(layout="LBHNC"), small_specs())


def test_v41_without_rings_keeps_the_tuple_packer():
    # A ratio-1-only V4.1 (or V4.0) mix has no ring and must still take the
    # pre-existing DeepseekV4 tuple-packing path unchanged.
    specs = v41_specs(
        compress_ratios=[0, 1, 1, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1, 3],
        num_draft_layers=0,
    )
    assert not any(isinstance(s, CircularBufferSpec) for s in specs.values())
    grouped = group_and_unify_kv_cache_specs(specs)
    assert grouped is not None
    groups = get_kv_cache_groups(make_vllm_config(), specs)
    assigned = [name for g in groups for name in g.layer_names]
    assert sorted(assigned) == sorted(specs)
