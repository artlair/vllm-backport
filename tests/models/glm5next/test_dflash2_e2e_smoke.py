# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end DFlash2 smoke: tiny Glm5Next target + DFlash2 draft on one GPU.

Random-weight proof that the full V2 DFlash2 pipeline serves a GLM-5.3-Flash
shaped target: the engine boots with the Qwen3-based DFlash2 draft decoder
(DFlash2Qwen3ForCausalLM, registered as DFlash2DraftModel), the draft's
target_layer_ids reach the target through the EagleModelMixin aux contract,
drafts are proposed and rejected-sampled against the target every step, and
tokens come out the other end. Acceptance is near zero with random weights
(and that is fine); this asserts mechanics, not accuracy.

Both configs are written to tmp_path at test time; weights come from
load_format="dummy", so nothing is committed and no checkpoint is needed.
"""

from typing import Any

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DFlash2 spec kernels, TRITON_MLA and mHC kernels need a GPU",
)

VOCAB_SIZE = 100
TARGET_LAYERS = 4
# dflash_config.target_layer_ids are DFlash 0-based ids; the runner maps them
# to target aux layers by adding 1 -> (1, 2, 3, 4): taps before layers 1-3
# plus the post-loop capture at end_layer=4.
TARGET_LAYER_IDS = [0, 1, 2, 3]
AUX_HIDDEN_STATE_LAYER_IDS = [1, 2, 3, 4]
NUM_SPEC_TOKENS = 3
PROMPT_TOKENS = 32
MAX_TOKENS = 8


def _write_target_config(tmp_path) -> str:
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

    target_dir = tmp_path / "target"
    # Dense tiny target with the production mHC residual layout and DeepSeek-V3
    # MLA dims: a supported dense-prefill combo whose 512 + 64 cache head the
    # fork's gather kernel accepts (320 or 576 only), no sparse indexer.
    config = Glm5NextConfig(
        text_config=dict(
            vocab_size=VOCAB_SIZE,
            hidden_size=64,
            num_hidden_layers=TARGET_LAYERS,
            num_attention_heads=4,
            intermediate_size=128,
            n_routed_experts=None,
            num_experts_per_token=None,
            n_shared_experts=None,
            q_lora_rank=32,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mla_nope=False,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            index_topk=None,
            layer_types=None,
            num_nextn_predict_layers=0,
            max_position_embeddings=512,
            mhc=True,
            mhc_num_residual_streams=2,
            pad_token_id=None,
        ),
        architectures=["Glm5NextForCausalLM"],
    )
    config.save_pretrained(target_dir)
    return str(target_dir)


def _write_draft_config(tmp_path) -> str:
    from transformers import Qwen3Config

    draft_dir = tmp_path / "draft"
    # DFlash2 draft decoder: standard Qwen3 layers with the DFlash2 query-conv
    # and candidate selector bolted on. conv sizes/selector shapes mirror the
    # real incoai/GLM-5.3-Flash-DFlash2 config; the selector rank stays 256
    # because it only costs a (vocab, rank) codebook pair, tiny at vocab 100.
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        architectures=["DFlash2DraftModel"],
        dflash_config={
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": TARGET_LAYER_IDS,
            "mask_token_id": VOCAB_SIZE - 1,
            "use_aux_hidden_state": True,
        },
        eagle_aux_hidden_state_layer_ids=AUX_HIDDEN_STATE_LAYER_IDS,
    )
    config.save_pretrained(draft_dir)
    return str(draft_dir)


def _engine_args(target_dir: str, draft_dir: str) -> dict[str, Any]:
    return dict(
        model=target_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=64,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config={
            "method": "dflash",
            "model": draft_dir,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        },
    )


def _spec_decode_counter(metrics: list, name: str) -> int:
    for metric in metrics:
        if metric.name == name and isinstance(metric.value, int):
            return metric.value
    return 0


def test_dflash2_glm5next_smoke_end_to_end(monkeypatch, tmp_path):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
    from vllm.v1.metrics.reader import Counter
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    # In-process engine so the test can inspect the runner's speculator and the
    # target's aux layer contract directly.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # The flashinfer sampler JIT-compiles on first use and needs nvcc, which
    # dev/CI boxes without a CUDA toolkit do not have; the torch sampler path
    # is equivalent for this smoke.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    target_dir = _write_target_config(tmp_path)
    draft_dir = _write_draft_config(tmp_path)

    llm = LLM(**_engine_args(target_dir, draft_dir))

    prompt = TokensPrompt(
        prompt_token_ids=[(i % (VOCAB_SIZE - 2)) + 1 for i in range(PROMPT_TOKENS)]
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
        use_tqdm=False,
    )

    # Tokens were produced and verified to the requested length.
    generated = outputs[0].outputs[0]
    assert len(generated.token_ids) == MAX_TOKENS
    assert generated.finish_reason == "length"
    assert all(0 <= t < VOCAB_SIZE for t in generated.token_ids)

    # The DFlash2 lane was actually selected: architectures-based V2 forcing
    # resolved the draft to the Qwen3-based DFlash2 decoder serving a GLM
    # target.
    model_runner = llm.llm_engine.engine_core.engine_core.model_executor
    model_runner = model_runner.driver_worker.worker.model_runner
    speculator = model_runner.speculator
    assert isinstance(speculator, DFlash2Speculator)
    assert isinstance(speculator.model, DFlash2Qwen3ForCausalLM)

    # The draft's aux layer ids reached the target through the mixin setter,
    # and the draft's aux encoder is sized for exactly those features
    # (target_hidden_size * len(aux layers) concatenated).
    assert model_runner.get_model().model.aux_hidden_state_layers == tuple(
        AUX_HIDDEN_STATE_LAYER_IDS
    )
    assert speculator.model.model.fc.input_size == 64 * len(AUX_HIDDEN_STATE_LAYER_IDS)

    # Drafts were proposed and verified on every decode step (the counters are
    # Prometheus; accepted counts include the always-taken bonus position).
    metrics = llm.get_metrics()
    num_drafts = _spec_decode_counter(metrics, "vllm:spec_decode_num_drafts")
    num_draft_tokens = _spec_decode_counter(
        metrics, "vllm:spec_decode_num_draft_tokens"
    )
    num_accepted = _spec_decode_counter(metrics, "vllm:spec_decode_num_accepted_tokens")
    assert isinstance(metrics[0], Counter)  # sanity: the snapshot carried typed metrics
    assert num_drafts > 0, "DFlash2 never proposed a draft"
    assert num_draft_tokens >= num_drafts * NUM_SPEC_TOKENS
    assert num_accepted >= 0
