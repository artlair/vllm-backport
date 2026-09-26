# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer on a mixed MTP-verify + prefill step.

When the spec and non-spec tokens are two contiguous runs, ``_forward``
slices them in place and lets both kernels write their run of the output
instead of gathering and scattering. That must be bit-identical to the
gather/scatter path: same output and same conv / recurrent state updates.
"""

import dataclasses
import types

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.models.glm5next.nvidia import kda as kda_mod
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernels"
)

H, D, NUM_SPEC = 16, 128, 3
PROJ = H * D
PREFIX = "model.layers.0.self_attn"


def build_metadata(layout: list[int | str], device: torch.device):
    """``layout``: ``"s"`` for a drafted request, an int for a non-spec
    request with that many tokens (and prior state)."""
    query_lens = [NUM_SPEC + 1 if r == "s" else r for r in layout]
    drafts = [NUM_SPEC if r == "s" else -1 for r in layout]
    batch = BatchSpec(seq_lens=[q + 100 for q in query_lens], query_lens=query_lens)
    common = create_common_attn_metadata(batch, 16, device)
    num_reqs = len(layout)
    slots = torch.arange(1, num_reqs * (NUM_SPEC + 1) + 1, dtype=torch.int32)
    common = dataclasses.replace(
        common, block_table_tensor=slots.view(num_reqs, -1).to(device)
    )
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", block_size=16)
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=16, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    meta = builder.build(
        0,
        common,
        num_accepted_tokens=torch.randint(
            1, NUM_SPEC + 2, (num_reqs,), dtype=torch.int32, device=device
        ),
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
    )
    return meta, sum(query_lens), slots.numel() + 1


def run_forward(layer, meta, projected, g1, conv_state, recurrent_state):
    num_tokens = projected.shape[0]
    layer.kv_cache = (conv_state, recurrent_state)
    out = torch.zeros(1, num_tokens, H, D, dtype=torch.bfloat16, device=g1.device)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kda_mod, "get_forward_context", lambda: ctx)
        kda_mod.Glm5NextLinearAttention._forward(
            layer,
            qkv_proj_states=projected[:, : 3 * PROJ],
            g1=g1,
            beta=projected[:, 3 * PROJ : 3 * PROJ + H].unsqueeze(0),
            core_attn_out=out,
        )
    return out


@pytest.mark.parametrize(
    "layout",
    [["s", "s", 300], [300, "s", "s"], ["s", "s", 1, 70]],
    ids=["spec-first", "prefill-first", "spec-then-two-prefills"],
)
@torch.inference_mode()
def test_contiguous_mixed_step_matches_gather_path(layout):
    torch.manual_seed(0)
    device = torch.device("cuda")
    meta, num_tokens, num_slots = build_metadata(layout, device)
    assert meta.spec_token_start is not None
    gather_meta = dataclasses.replace(
        meta, spec_token_start=None, non_spec_token_start=None
    )

    conv_shape, rec_shape = MambaStateShapeCalculator.kda_state_shape(
        4, 4 * H, D, conv_kernel_size=4, num_spec=NUM_SPEC
    )
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        _conv_state_dim_first=is_conv_state_dim_first(),
        _merged_conv_weight=0.3 * torch.randn(3 * PROJ, 4, device=device),
        q_conv1d=types.SimpleNamespace(bias=None),
        local_projection_size=PROJ,
        local_num_heads=H,
        head_dim=D,
        A_log=0.5 * torch.randn(1, 1, H, 1, device=device),
        dt_bias=0.1 * torch.randn(PROJ, device=device),
    )
    conv_state = (0.3 * torch.randn(num_slots, *conv_shape, device=device)).to(
        torch.bfloat16
    )
    recurrent_state = 0.05 * torch.randn(num_slots, *rec_shape, device=device)
    projected = torch.randn(
        num_tokens, 3 * PROJ + H + 2 * D, dtype=torch.bfloat16, device=device
    )
    g1 = torch.randn(1, num_tokens, H, D, dtype=torch.bfloat16, device=device)

    results = []
    for m in (gather_meta, meta):
        states = (conv_state.clone(), recurrent_state.clone())
        out = run_forward(layer, m, projected.clone(), g1, *states)
        results.append((out, *states))

    for ref, actual in zip(*results):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
