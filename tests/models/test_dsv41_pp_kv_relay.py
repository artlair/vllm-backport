# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 pp-relay: tests for relaying V4.1 shared caches across PP stages.

CPU part: the per-stage relay plan for the shipped DeepSeek-V4.1-Flash
topology under the deployment partitions, the send/recv key contract between
neighbouring stages, and the KV-cache grouping of a mirror rank (a rank that
registers a source's layer names without the source) against the source
rank, using the same planner the engine uses.

GPU part (one CUDA device): the mirror module itself, i.e. that it registers
the source's names with equal specs, redirects the right checkpoint names and
rebuilds both caches from a latent exactly like the source's kernels do.

``vllm.models.deepseek_v4_1`` imports the whole model (GPU kernels) from its
package ``__init__``, so the CPU tests load the dependency-free planning
module by path.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    _project_kv_cache_groups_to_worker,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)

_VLLM_ROOT = Path(vllm.__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_V41_CONFIG = _REPO_ROOT / "tools" / "dsv41" / "v41-config.json"


def _load_plan_module():
    path = _VLLM_ROOT / "models" / "deepseek_v4_1" / "pp_relay.py"
    spec = importlib.util.spec_from_file_location("dsv41_pp_relay_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


relay = _load_plan_module()

# The shipped DeepSeek-V4.1-Flash topology (tools/dsv41/v41-config.json).
FLASH_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
FLASH = relay.V41Topology(
    num_hidden_layers=40,
    compress_ratios=FLASH_RATIOS,
    kv_source_layer_ids=(2, 8, 14, 20),
    index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    index_topk=512,
    head_dim=512,
)


def keys(payloads) -> list[str]:
    return [p.key for p in payloads]


def plans_for(partition: list[int]):
    return relay.plan_pp_kv_relay(FLASH, relay.stage_bounds_from_partition(partition))


def test_topology_matches_the_shipped_config():
    text = json.loads(_V41_CONFIG.read_text())["text_config"]
    topology = relay.V41Topology.from_hf_config(SimpleNamespace(**text))
    # The config lists the 3 DSpark draft layers' ratios too (all 0).
    assert topology.compress_ratios == FLASH_RATIOS + (0, 0, 0)
    assert topology.num_hidden_layers == FLASH.num_hidden_layers
    assert topology.kv_source_layer_ids == FLASH.kv_source_layer_ids
    assert topology.index_source_layer_ids == FLASH.index_source_layer_ids
    assert topology.candidate_source_layer_id == 20
    assert topology.candidate_topk_blocks == 2048
    assert topology.index_topk == 512
    assert topology.head_dim == 512


def test_deployment_partition_8_7_9_8_8():
    plans = plans_for([8, 7, 9, 8, 8])
    assert [(p.start_layer, p.end_layer) for p in plans] == [
        (0, 8),
        (8, 15),
        (15, 24),
        (24, 32),
        (32, 40),
    ]
    # Groups 2 and 8 are stage-local.
    assert keys(plans[0].recv) == [] and keys(plans[0].send) == []
    # Group 14 is split: 14 on stage 1, 15..19 on stage 2. They are not index
    # sources, so they need 14's top-k rows as well as its latent.
    assert keys(plans[1].recv) == []
    assert keys(plans[1].send) == ["dsv41_relay_latent_14", "dsv41_relay_topk_14"]
    assert plans[1].produced == plans[1].send
    assert plans[2].recv == plans[1].send
    assert plans[2].consumed == plans[2].recv
    assert plans[2].mirrors == (relay.MirrorSpec(14, need_k_cache=False),)
    # Group 20 spans stages 2..4; 24/28 and 32/36 are non-kv index sources,
    # so both later stages need the K cache mirror and the candidates.
    assert keys(plans[2].send) == ["dsv41_relay_latent_20", "dsv41_relay_cand_20"]
    assert plans[2].produced == plans[2].send
    assert plans[3].recv == plans[2].send
    assert plans[3].consumed == plans[3].recv
    assert plans[3].mirrors == (relay.MirrorSpec(20, need_k_cache=True),)
    # Stage 3 forwards 20's payloads to stage 4 unchanged.
    assert plans[3].send == plans[3].recv
    assert plans[3].produced == ()
    assert plans[4].recv == plans[3].send
    assert plans[4].consumed == plans[4].recv
    assert plans[4].mirrors == (relay.MirrorSpec(20, need_k_cache=True),)
    assert plans[4].send == ()
    # Bytes per token on top of the hidden state (bf16 latent, int32 rows).
    assert plans[1].send_bytes_per_token(FLASH) == 512 * 2 + 512 * 4
    assert plans[2].send_bytes_per_token(FLASH) == 512 * 2 + 2048 * 4
    assert plans[3].send_bytes_per_token(FLASH) == 512 * 2 + 2048 * 4


def test_partition_11_9_9_9_7_does_not_sum_to_the_layer_count():
    # 45 layers: get_pp_indices rejects it and so does the planner.
    with pytest.raises(ValueError, match="cover 45 layers"):
        plans_for([11, 9, 9, 9, 7])


def test_stage_bounds_follow_vllm_pp_layer_partition(monkeypatch):
    # The runtime derives every rank's bounds the way make_layers does.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "8,7,9,8,8")
    assert relay.stage_bounds(40, 5) == relay.stage_bounds_from_partition(
        [8, 7, 9, 8, 8]
    )
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    assert relay.stage_bounds(40, 5) == relay.stage_bounds_from_partition(
        [8, 8, 8, 8, 8]
    )


def test_partition_11_9_9_9_2_needs_a_topk_only_relay():
    plans = plans_for([11, 9, 9, 9, 2])
    # Group 8: 8..10 on stage 0, 11..13 on stage 1.
    assert keys(plans[0].send) == ["dsv41_relay_latent_8", "dsv41_relay_topk_8"]
    assert plans[1].mirrors == (relay.MirrorSpec(8, need_k_cache=False),)
    # Group 14 is stage-local; 20 sits on stage 2, so stage 1 sends nothing.
    assert keys(plans[1].send) == []
    assert keys(plans[2].recv) == []
    # Stage 3 (29..37): 29..31 read index source 28's top-k (28 is the last
    # layer of stage 2), 32/36 borrow 20's K cache and mask with candidates.
    assert keys(plans[2].send) == [
        "dsv41_relay_latent_20",
        "dsv41_relay_topk_28",
        "dsv41_relay_cand_20",
    ]
    assert plans[2].produced == plans[2].send
    assert plans[3].consumed == plans[3].recv
    assert plans[3].mirrors == (relay.MirrorSpec(20, need_k_cache=True),)
    # Stage 4 (38, 39) consumes 20's compressed KV and 36's top-k rows but
    # has no index source: no K cache and no candidates.
    assert keys(plans[3].send) == ["dsv41_relay_latent_20", "dsv41_relay_topk_36"]
    assert keys(plans[3].produced) == ["dsv41_relay_topk_36"]
    assert plans[4].recv == plans[3].send
    assert plans[4].mirrors == (relay.MirrorSpec(20, need_k_cache=False),)
    assert plans[4].send == ()


def test_gap_stage_forwards_candidates_without_consuming_them():
    # Stage 2 holds only 29..31: consumers of 20's KV and 28's top-k, but no
    # index source, so it must forward the candidates untouched to stage 3.
    plans = plans_for([21, 8, 3, 8])
    assert keys(plans[2].recv) == [
        "dsv41_relay_latent_20",
        "dsv41_relay_topk_28",
        "dsv41_relay_cand_20",
    ]
    assert keys(plans[2].consumed) == ["dsv41_relay_latent_20", "dsv41_relay_topk_28"]
    assert keys(plans[2].send) == ["dsv41_relay_latent_20", "dsv41_relay_cand_20"]
    assert plans[2].produced == ()
    assert keys(plans[3].consumed) == ["dsv41_relay_latent_20", "dsv41_relay_cand_20"]


@pytest.mark.parametrize("partition", [[40], [2, 6, 6, 6, 20], [8, 6, 6, 20]])
def test_partitions_on_group_boundaries_relay_nothing(partition):
    for plan in plans_for(partition):
        assert not plan.is_active
        assert plan.mirrors == ()


@pytest.mark.parametrize(
    "partition",
    [[8, 7, 9, 8, 8], [11, 9, 9, 9, 2], [21, 8, 3, 8], [3, 3, 3, 3, 3, 3, 3, 19]],
)
def test_send_of_a_stage_is_the_recv_of_the_next(partition):
    plans = plans_for(partition)
    for prev, nxt in zip(plans, plans[1:]):
        assert prev.send == nxt.recv
        # A stage forwards only what it received, and produces only what
        # its own layers publish.
        for payload in prev.send:
            if payload in prev.produced:
                assert prev.start_layer <= payload.source < prev.end_layer
            else:
                assert payload in prev.recv
        for payload in nxt.consumed:
            assert payload in nxt.recv
        # Every mirror rebuilds a latent this stage consumes.
        for mirror in nxt.mirrors:
            assert relay.RelayPayload(relay.LATENT, mirror.kv_source) in nxt.consumed
    assert plans[0].recv == ()
    assert plans[-1].send == ()


def test_dspark_draft_layers_never_consume_shared_state():
    topology = relay.V41Topology.from_hf_config(
        SimpleNamespace(**json.loads(_V41_CONFIG.read_text())["text_config"])
    )
    for draft in (40, 41, 42):
        assert topology.compress_ratio(draft) == 0
        assert topology.kv_source_of(draft) is None
        assert topology.index_source_of(draft) is None
        assert not topology.is_index_source(draft)
        assert not topology.uses_candidates(draft)
    # And no payload ever lists them as a consumer, even when the last
    # backbone group would spill onto a further stage.
    plans = relay.plan_pp_kv_relay(
        topology, relay.stage_bounds_from_partition([20, 19, 1])
    )
    assert keys(plans[2].recv) == ["dsv41_relay_latent_20", "dsv41_relay_topk_36"]


# ---------------------------------------------------------------------------
# KV-cache grouping of a mirror rank versus the source rank. Spec builders
# mirror tests/v1/core/test_dsv41_kv_cache_groups.py (sm8x values).
# ---------------------------------------------------------------------------

MLA_BLOCK = 128
SWA_BLOCK = 32
WINDOW = 128
RING_BLOCK = 8
HEAD_DIM = 512
INDEXER_ROW_BYTES = 132
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
        head_size=2 * HEAD_DIM,
        head_size_v=0,
        dtype=torch.float32,
    )


def layer_specs(topology, layer_id: int) -> dict[str, KVCacheSpec]:
    """What DeepseekV4Attention registers for one layer (draft layers past
    the backbone are pure sliding window)."""
    prefix = f"model.layers.{layer_id}.attn"
    ratio = topology.compress_ratio(layer_id)
    specs: dict[str, KVCacheSpec] = {}
    if topology.is_kv_source(layer_id):
        specs[prefix] = mla_kv_spec(ratio)
        if topology.is_index_source(layer_id):
            specs[f"{prefix}.indexer.k_cache"] = indexer_spec(ratio)
    specs[f"{prefix}.swa_cache"] = swa_spec()
    if topology.is_kv_source(layer_id) and ratio > 1:
        specs[f"{prefix}.compressor.state_cache"] = ring_spec()
    return specs


def mirror_specs(topology, mirror) -> dict[str, KVCacheSpec]:
    """What DeepseekV41PPRelayMirror registers (same names, same specs)."""
    prefix = f"model.layers.{mirror.kv_source}.attn"
    ratio = topology.compress_ratio(mirror.kv_source)
    specs: dict[str, KVCacheSpec] = {prefix: mla_kv_spec(ratio)}
    if mirror.need_k_cache:
        specs[f"{prefix}.indexer.k_cache"] = indexer_spec(ratio)
    return specs


def make_vllm_config():
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "BLHNC"
    cache_config.num_gpu_blocks_override = None
    cache_config.enable_prefix_caching = True
    return SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=3,
            use_eagle=lambda: True,
            use_multi_module_mtp=lambda: False,
        ),
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN, original_max_model_len=MAX_MODEL_LEN
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None,
        max_in_flight_tokens=MLA_BLOCK,
    )


def group_index_of(groups, layer_name: str) -> int:
    (index,) = [i for i, g in enumerate(groups) if layer_name in g.layer_names]
    return index


@pytest.mark.parametrize("partition", [[8, 7, 9, 8, 8], [11, 9, 9, 9, 2]])
def test_mirror_rank_groups_like_the_source_rank(partition):
    num_draft_layers = 3
    plans = plans_for(partition)
    # Per-worker specs exactly as get_kv_cache_spec collects them: this
    # stage's layers (draft layers on the last stage) plus its mirrors.
    worker_specs: list[dict[str, KVCacheSpec]] = []
    for plan in plans:
        specs: dict[str, KVCacheSpec] = {}
        for mirror in plan.mirrors:
            specs.update(mirror_specs(FLASH, mirror))
        for layer_id in range(plan.start_layer, plan.end_layer):
            specs.update(layer_specs(FLASH, layer_id))
        if plan.pp_rank == len(plans) - 1:
            for i in range(num_draft_layers):
                specs.update(layer_specs(FLASH, FLASH.num_hidden_layers + i))
        worker_specs.append(specs)

    # get_kv_cache_configs' merge: a name seen on several workers must carry
    # an equal spec. Mirrors add no new names, so the union equals the
    # unsplit model's spec dict.
    merged: dict[str, KVCacheSpec] = {}
    for specs in worker_specs:
        for name, spec in specs.items():
            assert merged.setdefault(name, spec) == spec, name
    unsplit: dict[str, KVCacheSpec] = {}
    for layer_id in range(FLASH.num_hidden_layers + num_draft_layers):
        unsplit.update(layer_specs(FLASH, layer_id))
    assert merged == unsplit

    vllm_config = make_vllm_config()
    groups = get_kv_cache_groups(vllm_config, merged)
    projected = [_project_kv_cache_groups_to_worker(groups, s) for s in worker_specs]
    for plan, specs, worker_groups in zip(plans, worker_specs, projected):
        # "Some layers are not assigned to any group" guard from the engine.
        assert sum(len(g.layer_names) for g in worker_groups) == len(specs)
        for mirror in plan.mirrors:
            for name in mirror_specs(FLASH, mirror):
                # Same global group index (hence the same block table) as
                # on the rank that owns the source layer.
                source_rank = next(
                    p.pp_rank
                    for p in plans
                    if p.start_layer <= mirror.kv_source < p.end_layer
                )
                assert group_index_of(worker_groups, name) == group_index_of(
                    projected[source_rank], name
                )
                assert group_index_of(worker_groups, name) == group_index_of(
                    groups, name
                )
                # And the per-layer spec inside the projected group is the
                # source's.
                group = worker_groups[group_index_of(worker_groups, name)]
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                assert spec == unsplit[name]


# ---------------------------------------------------------------------------
# GPU: the mirror module (registration, checkpoint redirect, cache rebuild).
# ---------------------------------------------------------------------------

NUM_BLOCKS = 4
MAX_TOKENS = 64
ROPE_MAX_POSITION = 8192


@pytest.fixture(scope="module")
def dist_env():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    vllm_config = VllmConfig()
    vllm_config.cache_config.block_size = MLA_BLOCK
    vllm_config.scheduler_config.max_num_batched_tokens = MAX_TOKENS
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)
    yield vllm_config
    destroy_model_parallel()
    destroy_distributed_environment()


def _text_config() -> SimpleNamespace:
    text = json.loads(_V41_CONFIG.read_text())["text_config"]
    # transformers folds rope_scaling + rope_theta into rope_parameters when
    # it loads the config; the raw JSON still carries the old keys.
    text["rope_parameters"] = {**text["rope_scaling"], "rope_theta": text["rope_theta"]}
    # 1M positions would make the RoPE cache 256 MB; the test only needs a
    # few thousand.
    text["max_position_embeddings"] = ROPE_MAX_POSITION
    return SimpleNamespace(**text)


def _config_view(real_config) -> SimpleNamespace:
    """What the mirror reads off VllmConfig, with the V4.1 text config."""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_text_config()),
        cache_config=real_config.cache_config,
        attention_config=real_config.attention_config,
        scheduler_config=real_config.scheduler_config,
        compilation_config=real_config.compilation_config,
    )


@pytest.fixture
def mirror_env(dist_env):
    """A mirror of kv source 20 (ratio 1, needs the K cache) built the way
    DeepseekV4Model builds it: bf16 default dtype, on the CUDA device."""
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
    )
    from vllm.models.deepseek_v4_1.pp_relay_runtime import DeepseekV41PPRelayMirror

    config_view = _config_view(dist_env)
    static_forward_context = dist_env.compilation_config.static_forward_context
    static_forward_context.clear()
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(dist_env), torch.device("cuda"):
            mirror = DeepseekV41PPRelayMirror(
                config_view,
                DeepseekV41AmpereMLAAttention,
                kv_source=20,
                attn_prefix="model.layers.20.attn",
                need_k_cache=True,
            )
    finally:
        torch.set_default_dtype(default_dtype)
    yield mirror, config_view, dist_env, static_forward_context
    static_forward_context.clear()


def test_mirror_registers_the_source_names_with_the_source_specs(mirror_env):
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
    )
    from vllm.models.deepseek_v4_1.attention import (
        DeepseekV4IndexerCache,
        compressed_kv_cache_spec,
    )
    from vllm.models.deepseek_v4_1.pp_relay_runtime import (
        DeepseekV41RelayCompressedCache,
    )

    mirror, config_view, _, static_forward_context = mirror_env
    assert set(static_forward_context) == {
        "model.layers.20.attn",
        "model.layers.20.attn.indexer.k_cache",
    }
    compressed = static_forward_context["model.layers.20.attn"]
    assert isinstance(compressed, DeepseekV41RelayCompressedCache)
    assert compressed is mirror.compressed_cache
    assert static_forward_context["model.layers.20.attn.indexer.k_cache"] is (
        mirror.k_cache
    )
    assert isinstance(mirror.k_cache, DeepseekV4IndexerCache)
    # fp8_ds_mla resolution wrote itself back onto the cache config, as the
    # source's DeepseekV4Attention would on its own rank.
    assert config_view.cache_config.cache_dtype == "fp8_ds_mla"
    spec = compressed.get_kv_cache_spec(config_view)
    assert spec == compressed_kv_cache_spec(
        config_view, HEAD_DIM, 1, "fp8_ds_mla", torch.uint8
    )
    assert spec == mla_kv_spec(1)
    assert mirror.k_cache.get_kv_cache_spec(config_view) == indexer_spec(1)
    assert compressed.get_attn_backend() is DeepseekV41AmpereMLAAttention.backend_cls
    assert mirror.compress_ratio == 1
    assert mirror.wk.weight.shape == (128, HEAD_DIM)
    assert mirror.wk.weight.dtype == torch.bfloat16
    assert mirror.k_norm.weight.shape == (128,)
    # Checkpoint names of the source indexer's wk / k_norm -> mirror params.
    assert dict(mirror.checkpoint_params()) == {
        "layers.20.attn.indexer.wk.weight": "wk.weight",
        "layers.20.attn.indexer.k_norm.weight": "k_norm.weight",
    }


def test_mirror_without_k_cache_carries_no_weights(mirror_env):
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
    )
    from vllm.models.deepseek_v4_1.pp_relay_runtime import DeepseekV41PPRelayMirror

    _, config_view, real_config, static_forward_context = mirror_env
    with set_current_vllm_config(real_config), torch.device("cuda"):
        mirror = DeepseekV41PPRelayMirror(
            config_view,
            DeepseekV41AmpereMLAAttention,
            kv_source=14,
            attn_prefix="model.layers.14.attn",
            need_k_cache=False,
        )
    assert mirror.compress_ratio == 2
    assert mirror.k_cache is None
    assert dict(mirror.checkpoint_params()) == {}
    assert list(mirror.parameters()) == []
    assert "model.layers.14.attn" in static_forward_context
    assert "model.layers.14.attn.indexer.k_cache" not in static_forward_context
    assert mirror.compressed_cache.get_kv_cache_spec(config_view) == mla_kv_spec(2)


def test_mirror_refuses_plain_fp8_rows(mirror_env):
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.models.deepseek_v4_1.pp_relay_runtime import DeepseekV41PPRelayMirror

    _, config_view, real_config, static_forward_context = mirror_env
    assert not DeepseekV4FlashInferMLAAttention.use_fp8_ds_mla_layout
    with (
        set_current_vllm_config(real_config),
        pytest.raises(NotImplementedError, match="fp8_ds_mla or bf16"),
    ):
        DeepseekV41PPRelayMirror(
            config_view,
            DeepseekV4FlashInferMLAAttention,
            kv_source=8,
            attn_prefix="model.layers.8.attn",
            need_k_cache=False,
        )
    assert "model.layers.8.attn" not in static_forward_context


def test_mirror_rebuilds_both_caches_like_the_source(mirror_env):
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v4_1.common.ops import indexer_k_norm_rope_store
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    mirror, _, real_config, _ = mirror_env
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        mirror.wk.weight.normal_(0, 0.05, generator=gen)
        mirror.k_norm.weight.uniform_(0.5, 1.5, generator=gen)

    # Bound caches: [B, H=1, N, C] like the runner hands them out.
    kv_cache = torch.zeros(
        NUM_BLOCKS, 1, MLA_BLOCK, DS_MLA_ROW_BYTES, dtype=torch.uint8, device=device
    )
    k_cache = torch.zeros(
        NUM_BLOCKS, 1, MLA_BLOCK, INDEXER_ROW_BYTES, dtype=torch.uint8, device=device
    )
    mirror.compressed_cache.bind_kv_cache(kv_cache)
    mirror.k_cache.bind_kv_cache(k_cache)

    # A padded batch: the runner's positions / latent are [T] with T the
    # padded token count; the builders mark padding rows with slot -1.
    num_tokens = 24
    positions = torch.arange(100, 100 + MAX_TOKENS, dtype=torch.int64, device=device)
    kv_slots = torch.full((MAX_TOKENS,), -1, dtype=torch.int64, device=device)
    kv_slots[:num_tokens] = (
        torch.arange(3, 3 + num_tokens, device=device) * 7 % (NUM_BLOCKS * MLA_BLOCK)
    )
    k_slots = torch.full((MAX_TOKENS,), -1, dtype=torch.int64, device=device)
    k_slots[:num_tokens] = (
        torch.arange(5, 5 + num_tokens, device=device) * 11 % (NUM_BLOCKS * MLA_BLOCK)
    )
    latent = torch.randn(
        MAX_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen
    )
    attn_metadata = {
        "model.layers.20.attn": SimpleNamespace(slot_mapping=kv_slots),
        "model.layers.20.attn.indexer.k_cache": SimpleNamespace(slot_mapping=k_slots),
    }

    with set_forward_context(attn_metadata, real_config):
        mirror.write(latent, positions)

    # Reference: the source layer's own two insert paths on fresh caches.
    ref_kv = torch.zeros_like(kv_cache).squeeze(1)
    ref_k = torch.zeros_like(k_cache).squeeze(1)
    rope_quant_insert(
        latent, positions, mirror.rotary_emb.cos_sin_cache, ref_kv, kv_slots, 1
    )
    k_pre, _ = mirror.wk(latent)
    indexer_k_norm_rope_store(
        k_pre,
        positions,
        mirror.rotary_emb.cos_sin_cache,
        mirror.k_norm.weight,
        mirror.k_norm.variance_epsilon,
        ref_k,
        k_slots,
        1,
        False,
    )
    torch.cuda.synchronize()
    assert kv_cache.any() and k_cache.any()
    assert torch.equal(kv_cache.squeeze(1), ref_kv)
    assert torch.equal(k_cache.squeeze(1), ref_k)

    # Profile run (no metadata dict): a no-op rather than a crash.
    kv_cache.zero_()
    k_cache.zero_()
    with set_forward_context(None, real_config):
        mirror.write(latent, positions)
    torch.cuda.synchronize()
    assert not kv_cache.any() and not k_cache.any()
