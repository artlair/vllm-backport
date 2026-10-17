# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP relay of DeepSeek V4.1 cross-layer sharing sources."""

import itertools
from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4_1.common.pp_relay import (
    relay_source_ids,
    shadow_source_ids,
    source_reads,
)

# Sharing topology of DeepSeek-V4.1-Flash (config.json text_config).
V41_FLASH = SimpleNamespace(
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
)


def _stages(partition: list[int]) -> list[tuple[int, int]]:
    ends = list(itertools.accumulate(partition))
    return list(zip([0] + ends[:-1], ends))


@pytest.mark.cpu_test
def test_source_reads_follow_latest_source_below():
    assert source_reads(V41_FLASH, 1) == set()
    assert source_reads(V41_FLASH, 2) == set()
    assert source_reads(V41_FLASH, 7) == {2}
    assert source_reads(V41_FLASH, 20) == set()
    assert source_reads(V41_FLASH, 23) == {20}
    # Non-kv index sources read the kv source's index-K cache and the
    # candidate blocks; their consumers read their top-k.
    assert source_reads(V41_FLASH, 24) == {20}
    assert source_reads(V41_FLASH, 31) == {20, 28}


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        (2, ()),
        (5, (2,)),
        (8, ()),
        (14, ()),
        (17, (14,)),
        (20, ()),
        (21, (20,)),
        (24, (20,)),
        (26, (20, 24)),
        (30, (20, 28)),
        (37, (20, 36)),
    ],
)
def test_relay_source_ids(boundary, expected):
    assert relay_source_ids(V41_FLASH, boundary) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("partition", "relay_in", "shadows"),
    [
        (
            [2, 6, 6, 6, 10, 10],
            [(), (), (), (), (), (20, 28)],
            [(), (), (), (), (), (20, 28)],
        ),
        (
            # Stage 4 forwards layer 20's input to stage 5 untouched.
            [2, 6, 6, 10, 8, 8],
            [(), (), (), (), (20,), (20,)],
            [(), (), (), (), (20,), (20,)],
        ),
    ],
)
def test_lane_partitions(partition, relay_in, shadows):
    stages = _stages(partition)
    assert [relay_source_ids(V41_FLASH, s) if s else () for s, _ in stages] == (
        relay_in
    )
    assert [shadow_source_ids(V41_FLASH, s, e) for s, e in stages] == shadows


@pytest.mark.cpu_test
def test_every_stage_can_serve_its_layers_and_relay():
    """For every boundary pair, a stage gets each source it shadows from the
    previous stage, and produces or passes through each source it relays."""
    num_layers = V41_FLASH.num_hidden_layers
    for start, end in itertools.combinations(range(num_layers + 1), 2):
        relay_in = set(relay_source_ids(V41_FLASH, start)) if start else set()
        local = set(range(start, end))
        shadows = set(shadow_source_ids(V41_FLASH, start, end))
        assert shadows <= relay_in
        for layer_id in local:
            assert source_reads(V41_FLASH, layer_id) <= local | shadows
        for shadow in shadows:
            assert source_reads(V41_FLASH, shadow) <= shadows
        if end < num_layers:
            assert set(relay_source_ids(V41_FLASH, end)) <= local | relay_in


# ---------------------------------------------------------------------------
# In-process PP emulation on one GPU
# ---------------------------------------------------------------------------
#
# Real PP needs one GPU per stage. Instead, one PP=1 engine runs a harness
# model holding the unsplit reference and, next to it, every stage of several
# PP splits (built and run as their PP rank, under their own layer-name
# prefixes so all caches coexist). Each step feeds the same batch through the
# reference and through each split stage by stage, passing the intermediate
# tensors through static per-stage buffers from make_empty_intermediate_tensors
# like the PP runner does, and records how far each split's final hidden
# states drift from the reference.

TINY_TEXT_CONFIG = {
    "vocab_size": 129280,
    "hidden_size": 1024,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "max_position_embeddings": 65536,
    "rope_theta": 10000,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [0, 0, 2, 2, 2] + [1] * 7,
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 5],
    "index_source_layer_ids": [2, 5, 7, 9],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 5,
    "candidate_topk_blocks": 128,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "num_nextn_predict_layers": 0,
}

# name -> (layers per stage, zero the relayed source inputs). [3, 9] shadows
# the ratio-2 kv source 2 (with its compressor state cache); [6, 2, 4]
# shadows the ratio-1 kv/candidate source 5 on stage 1, which forwards layer
# 5's input to stage 2 next to layer 7's. The zeroed split is the negative
# control showing the comparison sees the relayed state.
PP_SPLITS = {
    "copy": ([12], False),
    "split_a": ([3, 9], False),
    "split_b": ([6, 2, 4], False),
    "split_b_zeroed": ([6, 2, 4], True),
}


def _canonical_param_name(name: str) -> str:
    """Name a model parameter by the checkpoint weight it is loaded from."""
    if name.startswith("pp_relay_sources."):
        return "layers." + name.removeprefix("pp_relay_sources.")
    return name


def _make_harness_cls():
    import contextlib
    import zlib

    import torch
    from torch import nn

    import vllm.distributed.parallel_state as parallel_state
    import vllm.distributed.utils as dist_utils
    import vllm.models.deepseek_v4_1.nvidia.model as v41_model
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )

    @contextlib.contextmanager
    def as_pp_stage(partition: list[int], rank: int):
        group = SimpleNamespace(
            rank_in_group=rank,
            world_size=len(partition),
            is_first_rank=rank == 0,
            is_last_rank=rank == len(partition) - 1,
        )
        bounds = _stages(partition)
        saved = (
            v41_model.get_pp_group,
            v41_model.get_pp_indices,
            parallel_state.get_pp_group,
            dist_utils.get_pp_indices,
        )
        v41_model.get_pp_group = lambda: group
        parallel_state.get_pp_group = lambda: group
        v41_model.get_pp_indices = dist_utils.get_pp_indices = (
            lambda num_layers, r, size: bounds[r]
        )
        try:
            yield
        finally:
            (
                v41_model.get_pp_group,
                v41_model.get_pp_indices,
                parallel_state.get_pp_group,
                dist_utils.get_pp_indices,
            ) = saved

    class DeepseekV41PPSplitHarness(v41_model.DeepseekV41LLMForCausalLM):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.splits = nn.ModuleDict()
            for split, (partition, _) in PP_SPLITS.items():
                stages = nn.ModuleList()
                for rank in range(len(partition)):
                    with as_pp_stage(partition, rank):
                        stages.append(
                            v41_model.DeepseekV4Model(
                                vllm_config=vllm_config, prefix=f"{split}_s{rank}"
                            )
                        )
                self.splits[split] = stages
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            dtype = vllm_config.model_config.dtype
            self.stage_inputs = {
                split: [None]
                + [
                    stage.make_empty_intermediate_tensors(max_tokens, dtype, "cuda")
                    for stage in list(stages)[1:]
                ]
                for split, stages in self.splits.items()
            }
            self.max_diff = torch.zeros(len(PP_SPLITS), device="cuda")
            self.max_ref = torch.zeros((), device="cuda")
            self.num_steps = torch.zeros((), dtype=torch.int64, device="cuda")

        def process_weights_after_loading(self) -> None:
            # Dummy weights depend only on shape, so all layers would match;
            # reseed each parameter from its checkpoint name so a shadow with
            # another layer's weights, or reading another layer, cannot pass.
            models = [
                self.model,
                *(s for stages in self.splits.values() for s in stages),
            ]
            for model in models:
                for name, param in model.named_parameters():
                    initialize_single_dummy_weight(
                        param,
                        low=-0.03,
                        high=0.03,
                        seed=zlib.crc32(_canonical_param_name(name).encode()),
                    )
            super().process_weights_after_loading()
            for split, (partition, _) in PP_SPLITS.items():
                with as_pp_stage(partition, 0):
                    self.splits[split][0].finalize_mhc_broadcast_weights()

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
            lookback_token_ids=None,
        ):
            ref = super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids=lookback_token_ids,
            )
            attn_metadata = get_forward_context().attn_metadata
            real_step = isinstance(attn_metadata, dict)
            num_tokens = positions.shape[0]
            if real_step:
                # Padding rows read unwritten cache slots; compare real tokens.
                swa = attn_metadata[self.model.layers[0].attn.swa_cache_layer.prefix]
                n = swa.num_decode_tokens + swa.num_prefill_tokens
            for i, (split, (partition, zeroed)) in enumerate(PP_SPLITS.items()):
                hidden = None
                for rank, stage in enumerate(self.splits[split]):
                    stage_input = self.stage_inputs[split][rank]
                    if stage_input is not None:
                        sent = hidden.tensors
                        expected = stage_input.tensors
                        assert sent.keys() == expected.keys(), (split, rank)
                        for key, buf in expected.items():
                            assert sent[key].shape[1:] == buf.shape[1:], key
                            assert sent[key].dtype == buf.dtype, key
                            if zeroed and key.startswith("relay_attn_input."):
                                buf[:num_tokens].zero_()
                            else:
                                buf[:num_tokens].copy_(sent[key][:num_tokens])
                        stage_input = stage_input[:num_tokens]
                    with as_pp_stage(partition, rank):
                        hidden = stage(
                            input_ids,
                            positions,
                            stage_input,
                            inputs_embeds,
                            lookback_token_ids=lookback_token_ids,
                        )
                if real_step:
                    diff = (hidden[:n] - ref[:n]).float().abs().max()
                    self.max_diff[i] = torch.maximum(self.max_diff[i], diff)
            if real_step:
                ref_max = ref[:n].abs().max().float()
                self.max_ref.copy_(torch.maximum(self.max_ref, ref_max))
                self.num_steps += 1
            return ref

    return DeepseekV41PPSplitHarness


@pytest.fixture
def tiny_v41_dir(tmp_path):
    import json

    config = {
        "architectures": ["DeepseekV41PPSplitHarness"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "text_config": {"model_type": "deepseek_v41_text", **TINY_TEXT_CONFIG},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.mark.parametrize("cudagraph", [False, True])
def test_pp_split_matches_unsplit(tiny_v41_dir, monkeypatch, cudagraph):
    """Each split's last stage reproduces the unsplit model's final hidden
    states through chunked prefill, decode (optionally replayed from full
    CUDA graphs) and prefix-cache hits, while zeroing the relayed source
    inputs changes them."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    ModelRegistry.register_model("DeepseekV41PPSplitHarness", _make_harness_cls())

    llm = LLM(
        model=str(tiny_v41_dir),
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        enforce_eager=not cudagraph,
        # Decode batches of 1, 2 and 4 requests replay unpadded full graphs.
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        kv_cache_memory_bytes=1 << 30,
        # The test GPU may be shared; only the startup free-memory check
        # reads this once the KV cache size is fixed.
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    generator = torch.Generator().manual_seed(0)
    prompts = [
        torch.randint(10, 120000, (n,), generator=generator).tolist()
        for n in (2600, 1500, 200, 2100)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    # Shared prefixes hit the prefix cache, so the shadow caches must hold
    # the same blocks as the sources'.
    llm.generate(
        [
            TokensPrompt(prompt_token_ids=prompts[0][:1800] + prompts[2]),
            TokensPrompt(prompt_token_ids=prompts[3] + prompts[2]),
        ],
        params,
    )

    def collect(model):
        return (
            int(model.num_steps),
            float(model.max_ref),
            dict(zip(PP_SPLITS, model.max_diff.tolist())),
        )

    num_steps, max_ref, max_diff = llm.apply_model(collect)[0]
    print(f"steps={num_steps} max|ref|={max_ref} max|split-ref|={max_diff}")
    assert num_steps > 10
    assert max_ref > 0
    assert max_diff["copy"] == 0.0
    assert max_diff["split_a"] == 0.0
    assert max_diff["split_b"] == 0.0
    assert max_diff["split_b_zeroed"] > 1e-3 * max_ref

    def load_shadow_weights(model):
        # Last stage of split_b shadows source 5 (from stage 0) and source 7.
        stage = model.splits["split_b"][2]
        shadow = stage.pp_relay_sources["5"].attn
        weights_proj = torch.randn_like(shadow.indexer.weights_proj.weight)
        wq_a = torch.randn_like(shadow.fused_wqa_wkv.weight[: shadow.q_lora_rank])
        loaded = stage.load_weights(
            [
                ("layers.5.attn.indexer.weights_proj.weight", weights_proj),
                ("layers.5.attn.wq_a.weight", wq_a),
                # Layer 5's non-attention weights stay on stage 0.
                ("layers.5.attn_norm.weight", torch.ones(1024)),
            ]
        )
        return (
            loaded,
            torch.equal(shadow.indexer.weights_proj.weight, weights_proj),
            torch.equal(shadow.fused_wqa_wkv.weight[: shadow.q_lora_rank], wq_a),
        )

    loaded, weights_proj_ok, wq_a_ok = llm.apply_model(load_shadow_weights)[0]
    assert loaded == {
        "pp_relay_sources.5.attn.indexer.weights_proj.weight",
        "pp_relay_sources.5.attn.fused_wqa_wkv.weight",
    }
    assert weights_proj_ok and wq_a_ok
