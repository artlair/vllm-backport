# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-side aux hidden state contract for DFlash2/EAGLE3 drafting.

Upstream #56983: Glm5NextModel gains EagleModelMixin, so the DFlash2/EAGLE3
drafters can be fed per-layer target hidden states. Under mHC the capture
must complete the deferred hc_post and contract the widened residual stream
back to hidden_size; the shape asserts below would catch a raw widened or
un-summed stream leaking into the draft.

The draft-side wiring (speculator, V2 forcing, extract_hidden_states) is
covered by tests/v1/spec_decode/test_dflash2.py and
test_extract_hidden_states.py; this file owns the target contract, which
needs a real model build (MLA backend construction and the mHC kernels are
CUDA-only here), so it lives with the glm5next model tests.
"""

import contextlib
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    supports_eagle3,
)
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="tiny GLM5Next build needs an MLA attention backend and mHC kernels",
)

HIDDEN_SIZE = 64
NUM_LAYERS = 4
END_LAYER = NUM_LAYERS  # exclusive make_layers bound == num_hidden_layers at PP=1
TOKENS = 5


def _tiny_config():
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

    return Glm5NextConfig(
        text_config=dict(
            vocab_size=100,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=4,
            intermediate_size=128,
            # Dense MLP: the aux contract does not involve MoE routing.
            n_routed_experts=None,
            num_experts_per_token=None,
            n_shared_experts=None,
            # Smallest MLA dims the prefill backend selector accepts.
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mla_nope=True,
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


class _StubAttention(nn.Module):
    """Deterministic stand-in for the MLA layer.

    Real attention would need KV-cache metadata scaffolding that does not
    participate in the aux contract; the capture logic under test lives in
    Glm5NextModel.forward around the layers (the kimi_k3 eagle3 tests stub
    the same way).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(hidden_states)


@pytest.fixture
def tiny_model(tmp_path):
    """A tiny randomly-weighted Glm5NextForCausalLM on one GPU.

    One forward per aux-layer configuration reuses the build; the setter
    (not a rebuild) switches what each forward captures.
    """
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    config = _tiny_config()
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        runner="generate",
        max_model_len=64,
        dtype="bfloat16",
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(cache_dtype="auto"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=128,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cuda"),
    )

    fd, dist_init_file = tempfile.mkstemp()
    os.close(fd)
    prev_default_dtype = torch.get_default_dtype()
    # MLAAttention resolves its backend from the global default dtype, as in
    # serving where the engine sets it from the model config before build.
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{dist_init_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)

            with torch.device("cuda"):
                model = Glm5NextForCausalLM(vllm_config=vllm_config, prefix="")
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name.endswith((".hc_attn_scale", ".hc_ffn_scale")):
                            param.fill_(1.0)
                        elif param.dtype.is_floating_point:
                            param.normal_(0, 0.02)
                for layer in model.model.layers:
                    layer.self_attn = _StubAttention(HIDDEN_SIZE)
        yield model
    finally:
        torch.set_default_dtype(prev_default_dtype)
        cleanup_dist_env_and_memory()
        with contextlib.suppress(OSError):
            os.unlink(dist_init_file)


def _forward(model, input_ids, positions):
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
        )


def test_glm5next_advertises_eagle3_support():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)
    assert EagleModelMixin in Glm5NextModel.__mro__


def test_aux_layer_setter_reaches_the_decoder():
    """set_aux_hidden_state_layers must unwrap both wrappers to the decoder.

    The causal-lm wrapper holds the decoder at ``.model``; the multimodal
    wrapper at ``.language_model.model``. The draft-side loader calls the
    setter on the top-level target model, so a missed unwrap would silently
    configure nothing.
    """
    causal = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(causal)
    causal.model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(causal.model)
    causal.set_aux_hidden_state_layers((1, 2))
    assert causal.model.aux_hidden_state_layers == (1, 2)

    mm = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(mm)
    # get_language_model() resolves the wrapper's decoder; give it what the
    # real _mark_language_model registration provides (kimi_k3 convention).
    mm.language_model = SimpleNamespace(
        model=causal.model,
        embed_input_ids=lambda _: None,
    )
    object.__setattr__(mm, "_language_model_names", ["language_model"])
    mm.set_aux_hidden_state_layers((2, 3))
    assert mm.language_model.model.aux_hidden_state_layers == (2, 3)


def test_forward_captures_aux_hidden_states(tiny_model):
    model = tiny_model
    input_ids = torch.randint(0, 100, (TOKENS,), device="cuda")
    positions = torch.arange(TOKENS, device="cuda")

    # (1, 2, END_LAYER): two in-loop taps plus the post-loop final aux.
    aux_layer_sets = [(1, 2, END_LAYER), (1, 2), (0,), ()]
    for aux_layers in aux_layer_sets:
        model.set_aux_hidden_state_layers(aux_layers)
        out = _forward(model, input_ids, positions)

        if not aux_layers:
            assert isinstance(out, torch.Tensor)
            assert out.shape == (TOKENS, HIDDEN_SIZE)
            assert torch.isfinite(out).all()
            continue

        assert isinstance(out, tuple) and len(out) == 2
        hidden_states, aux_hidden_states = out
        assert hidden_states.shape == (TOKENS, HIDDEN_SIZE)
        assert len(aux_hidden_states) == len(aux_layers)
        for aux in aux_hidden_states:
            # mHC-contracted: the widened stream would be n * hidden_size.
            assert aux.shape == (TOKENS, HIDDEN_SIZE)
            assert aux.dtype == hidden_states.dtype
            assert torch.isfinite(aux).all()

    # Distinct tap points feed distinct states to the drafter.
    model.set_aux_hidden_state_layers((1, 2))
    _, aux = _forward(model, input_ids, positions)
    assert not torch.equal(aux[0], aux[1])
