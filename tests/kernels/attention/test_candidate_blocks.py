# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-block selection: reference check plus opt-in radix backend."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Runs under the Triton CPU interpreter so the selector math is checkable
# without a GPU. Inline PTX (warp-matched histogram aggregation) is not
# interpretable, so the probe takes the reference histogram branch.
_CPU_PROBE = r"""
import torch

from vllm.model_executor.kernels.attention.dsa import candidate_radix as radix

radix._WARP_AGG = False


def reference(logits, starts, ends, k, block_size, row_repeat):
    rows, width = logits.shape
    nblocks = -(-width // block_size)
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for row in range(rows):
        start = 0 if starts is None else int(starts[row // row_repeat])
        end = int(ends[row // row_repeat])
        col_end = min(end, width)
        if col_end <= start:
            continue
        blocks = -(-(col_end - start) // block_size)
        newest = (end - start - 1) // block_size
        if newest >= 0:
            blocks = max(blocks, newest + 1)
        blocks = min(blocks, nblocks)
        cols = torch.arange(start, start + blocks * block_size)
        padded = torch.full(cols.shape, -float("inf"), dtype=logits.dtype)
        inside = cols < col_end
        padded[inside] = logits[row, cols[inside]]
        scores = padded.view(blocks, block_size).float().amax(-1)
        if newest < blocks:
            scores[newest] = float("inf")
        # Lowest block ID wins ties, which is what the selector promises.
        order = torch.argsort(scores, descending=True, stable=True)[: min(k, blocks)]
        chosen = order[scores[order] > -float("inf")]
        out[row, : chosen.numel()] = chosen.int()
    return out


def check(name, logits, starts, ends, k, block_size, row_repeat=1, out_stride=1):
    rows = logits.shape[0]
    backing = torch.full((rows, k * out_stride + 1), -9, dtype=torch.int32)
    out = backing[:, : k * out_stride : out_stride]
    radix.select_candidate_blocks_radix(
        logits, starts, ends, k, block_size, out, row_repeat
    )
    want = reference(logits, starts, ends, k, block_size, row_repeat)
    assert torch.equal(out.sort(1).values, want.sort(1).values), name
    if out_stride > 1:
        assert (backing[:, 1 : k * out_stride : out_stride] == -9).all(), name
        assert (backing[:, -1] == -9).all(), name


torch.manual_seed(17)
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    width, block_size, k = 97, 3, 7
    logits = torch.randn(6, width, dtype=dtype)
    logits[0, :16] = 0
    logits[2, 5] = float("nan")
    logits[3, :] = -float("inf")
    logits[4, ::7] = float("inf")
    starts = torch.tensor([0, 5, 3, 0, 7, 1], dtype=torch.int32)
    ends = torch.tensor([0, width, width - 1, 1, 11, width], dtype=torch.int32)
    check(f"prefill {dtype}", logits, starts, ends, k, block_size)
    ends = torch.tensor([0, 1, width], dtype=torch.int32)
    check(f"decode {dtype}", logits, None, ends, k, block_size, 2)
logits = torch.randn(4, 73)
ends = torch.tensor([73, 73, 80, 0], dtype=torch.int32)
check("k above block count", logits, None, ends, 16, 8)
check("strided output", logits, None, ends, 5, 8, out_stride=3)
# Long rows take the coarse-histogram, refinement and scatter path.
width, block_size, k = 9 * 8192, 8, 512
logits = torch.randn(2, width)
logits[1, ::13] = float("nan")
ends = torch.tensor([width, width - 5], dtype=torch.int32)
check("long rows", logits, None, ends, k, block_size)
check("long row pinned past width", logits, torch.zeros(2, dtype=torch.int64),
      torch.tensor([width + 40, 0], dtype=torch.int32), k, block_size)
print("candidate radix reference cases passed")
"""


def test_candidate_radix_matches_reference_on_cpu_interpreter():
    """The selector math is checkable on CPU through Triton's interpreter."""
    env = dict(os.environ, TRITON_INTERPRET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_REPO_ROOT.as_posix(), env.get("PYTHONPATH", "")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", _CPU_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_cuda(),
    reason="CUDA required",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "width,block_size,k", [(97, 3, 7), (8193, 8, 33), (278528, 8, 2048)]
)
@pytest.mark.parametrize("decode", [False, True])
def test_candidate_radix_preserves_membership_and_mask(
    monkeypatch, dtype, width, block_size, k, decode
):
    """Changing bounds, NaNs and boundary ties must preserve the consumer mask."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        select_candidate_blocks,
    )

    torch.manual_seed(73)
    rows = 6
    repeat = 2 if decode else 1
    logits = torch.randn(rows, width * 2, device="cuda", dtype=dtype)[:, ::2]
    logits[0].fill_(0)
    logits[1, ::17] = torch.nan
    logits[2, ::11] = torch.inf
    logits[3].fill_(-torch.inf)
    logits[4].round_()
    if dtype == torch.float32:
        logits[5].copy_(torch.logspace(-30, 30, width, device="cuda"))
    starts = (
        None
        if decode
        else torch.tensor(
            [0, 3, width - 1], device="cuda", dtype=torch.int64
        ).repeat_interleave(2)
    )
    ends = torch.full((3 if decode else 6,), width, device="cuda", dtype=torch.int32)
    backing = torch.full((rows, k * 2), -9, device="cuda", dtype=torch.int32)
    actual = backing[:, ::2]
    expected = torch.empty_like(actual)
    monkeypatch.setattr(envs, "VLLM_DSV4_CANDIDATE_SELECTION_BACKEND", "triton_radix")
    select_candidate_blocks(
        logits, starts, ends, k, block_size, actual, row_repeat=repeat
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_candidate_blocks(
            logits, starts, ends, k, block_size, actual, row_repeat=repeat
        )
    for bounds in ([0, 3, width - 1], [1, 5, width + 3], [width, width, width]):
        bound_tensor = torch.tensor(bounds, dtype=ends.dtype, device="cuda")
        ends.copy_(bound_tensor if decode else bound_tensor.repeat_interleave(2))
        graph.replay()
        monkeypatch.setattr(envs, "VLLM_DSV4_CANDIDATE_SELECTION_BACKEND", "torch")
        select_candidate_blocks(
            logits, starts, ends, k, block_size, expected, row_repeat=repeat
        )
        # Candidate order is discarded by the membership-mask consumer.
        torch.testing.assert_close(
            actual.sort(1).values, expected.sort(1).values, rtol=0, atol=0
        )
        assert (backing[:, 1::2] == -9).all()
        masked = logits.clone()
        reference = logits.clone()
        apply_candidate_mask(
            masked, starts, ends, actual, block_size, row_repeat=repeat
        )
        apply_candidate_mask(
            reference, starts, ends, expected, block_size, row_repeat=repeat
        )
        torch.testing.assert_close(masked, reference, rtol=0, atol=0, equal_nan=True)
