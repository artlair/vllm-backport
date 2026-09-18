# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA top-k.")
@pytest.mark.parametrize("is_padding", [True, False])
def test_make_dummy_padding_controls_moe_routing(monkeypatch, is_padding: bool):
    """Dummy tokens marked as padding are dropped by the MoE router (top-k id
    -1), which is wanted for idle DP ranks. Profile runs must not mark them, or
    no token reaches the experts and MoE memory is never profiled."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    num_tokens = 16
    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(4, num_tokens, buffers, is_padding=is_padding)
    hidden_states = torch.randn(num_tokens, 4, device=DEVICE)
    router_logits = torch.randn(num_tokens, 8, device=DEVICE)

    with set_forward_context(None, VllmConfig(), is_padding=batch.is_padding):
        _, topk_ids, _ = fused_topk(hidden_states, router_logits, 2, False)

    assert bool((topk_ids == -1).all()) is is_padding
    assert bool((topk_ids >= 0).all()) is not is_padding
