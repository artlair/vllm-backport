# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_fp64", [False, True])
def test_dspark_revisited_position_uses_step_seed(use_fp64):
    """A cached draft at a revisited position must use its current step."""
    device = torch.device("cuda")
    batch = 4096
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.acceptance_estimator = None
    spec._d2t_scatter_index = None
    spec.draft_watermarker = None
    spec.use_fp64_gumbel = use_fp64
    spec.temperature = torch.ones(batch, device=device)
    spec.seeds = torch.arange(batch, dtype=torch.int64, device=device)
    spec._step_cols = torch.arange(2, dtype=torch.int32, device=device)
    spec.draft_logits = torch.empty(batch, 2, 2, device=device)
    logits = torch.tensor([0.7, 0.3], device=device).log().expand(batch, -1)
    mapping = torch.arange(batch, device=device)
    positions = torch.full((batch,), 22, dtype=torch.int64, device=device)

    first = spec._sample_logits(logits, mapping, positions, 1)
    repeated = spec._sample_logits(logits, mapping, positions, 1)
    next_round = spec._sample_logits(logits, mapping, positions, 0)
    torch.testing.assert_close(first, repeated)
    expected = gumbel_sample(
        logits,
        mapping,
        spec.temperature,
        spec.seeds,
        positions - 1,
        apply_temperature=True,
        logits_cache=spec.draft_logits,
        logits_cache_col=spec._step_cols[0],
        use_fp64=use_fp64,
    )
    torch.testing.assert_close(next_round, expected)
    mismatch_rate = (first != next_round).float().mean().item()
    assert 0.35 < mismatch_rate < 0.5
