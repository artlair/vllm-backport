# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the k < n < 2k band of persistent_topk.

An external report against this branch (vllm-project/vllm#50576, comment
5158563519) claimed the radix persistent_topk returns wrong indices when the
candidate count falls strictly between k and 2k, observed as degenerate
output at prompt lengths 2049-4096 (index_topk=512, compress_ratio=4).

The claim was audited on SM80 (2026-08-03) and did NOT reproduce, at either
level: 2x 286/286 correct end-to-end requests across every kernel-path tier,
and 183/183 direct kernel checks (this test's cases). The reporter ran a
0.25.2-era fork with a vendored third-party kernel, not this tree's
csrc/libtorch_stable/topk.cu. This test pins the in-band behaviour so any
future topk.cu change that regresses it fails loudly rather than silently.

Poison values above ``lengths`` mirror production, which calls the indexer
logits kernels with clean_logits=False: memory beyond seq_len is garbage,
and the kernel must respect ``lengths`` internally.
"""

import pytest
import torch

from vllm.platforms import current_platform

K = 512
MAX_SEQ_LEN = 4352
WORKSPACE_BYTES = 1024 * 1024
POISON = 1.0e9

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only op"
)


@requires_cuda
@pytest.mark.parametrize(
    "n",
    [
        496,  # n < k: top-k selects everything
        523,  # in the reported corruption band (k < n < 2k) ...
        749,
        971,
        1021,
        4096,  # n > 2k
    ],
)
@pytest.mark.parametrize("num_rows", [1, 8, 32])
def test_persistent_topk_in_band(n: int, num_rows: int):
    torch.manual_seed(0)
    device = "cuda"
    logits = torch.full(
        (num_rows, MAX_SEQ_LEN), POISON, dtype=torch.float32, device=device
    )
    logits[:, :n] = torch.randn(num_rows, n, device=device)
    lengths = torch.full((num_rows,), n, dtype=torch.int32, device=device)
    out = torch.full((num_rows, K), -1, dtype=torch.int32, device=device)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    torch.ops._C.persistent_topk(logits, lengths, out, ws, K, MAX_SEQ_LEN)
    # .tolist() below synchronizes; no explicit torch.cuda call needed.

    kk = min(n, K)
    for r in range(num_rows):
        ref = set(torch.topk(logits[r, :n], kk).indices.tolist())
        got_raw = [i for i in out[r].tolist() if i >= 0]
        # No index may point past the row's candidate count (poison region).
        assert all(i < n for i in got_raw), f"row {r}: index >= n leaked"
        got = set(got_raw)
        assert len(got) == len(got_raw), f"row {r}: duplicate indices"
        assert got == ref, (
            f"row {r} (n={n}): {len(ref - got)} missing / "
            f"{len(got - ref)} extra vs torch.topk"
        )


# Rows at or below 32 dispatch to FilteredTopK only while `max_seq_len` stays
# within RADIX_THRESHOLD; past it topk.cu keeps the multi-CTA radix path, whose
# cooperative barrier and RadixRowState workspace nothing else exercises at
# these row counts. One case per side of that boundary, so a dispatch change
# cannot silently leave one path untested.
@requires_cuda
@pytest.mark.parametrize("n", [32768, 40960])
@pytest.mark.parametrize("num_rows", [1, 4])
def test_persistent_topk_across_radix_threshold(n: int, num_rows: int):
    torch.manual_seed(0)
    device = "cuda"
    width = 40960  # > RADIX_THRESHOLD (32768), 128-aligned like production
    logits = torch.full((num_rows, width), POISON, dtype=torch.float32, device=device)
    logits[:, :n] = torch.randn(num_rows, n, device=device)
    lengths = torch.full((num_rows,), n, dtype=torch.int32, device=device)
    out = torch.full((num_rows, K), -1, dtype=torch.int32, device=device)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    torch.ops._C.persistent_topk(logits, lengths, out, ws, K, width)

    for r in range(num_rows):
        ref = set(torch.topk(logits[r, :n], K).indices.tolist())
        got_raw = [i for i in out[r].tolist() if i >= 0]
        assert all(i < n for i in got_raw), f"row {r}: index >= n leaked"
        got = set(got_raw)
        assert len(got) == len(got_raw), f"row {r}: duplicate indices"
        assert got == ref, (
            f"row {r} (n={n}): {len(ref - got)} missing / "
            f"{len(got - ref)} extra vs torch.topk"
        )


# ---------------------------------------------------------------------------
# Live length vs padded row width.
#
# Production captures the indexer logits at max_model_len columns because the
# CUDA graph freezes the row width, while the live context is much shorter.
# The device-side regrouping in topk.cu must select the same indices, in the
# same order, for the same live length at any padded width, and must keep the
# single-CTA / T=0 paths intact.
# ---------------------------------------------------------------------------


def _run_persistent_topk(logits, lengths, k, max_seq_len):
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(logits, lengths, out, ws, k, max_seq_len)
    return out


def _assert_matches_torch(logits, lengths, out, k):
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        kk = min(n, k)
        ref = set(torch.topk(logits[r, :n], kk).indices.tolist()) if kk else set()
        got_raw = [i for i in out[r].tolist() if i >= 0]
        assert all(i < n for i in got_raw), f"row {r}: index >= live length"
        got = set(got_raw)
        assert len(got) == len(got_raw), f"row {r}: duplicate indices"
        assert got == ref, (
            f"row {r} (n={n}): {len(ref - got)} missing / {len(got - ref)} extra"
        )


@requires_cuda
@pytest.mark.parametrize("num_rows", [16, 48])
@torch.inference_mode()
def test_persistent_topk_nan_ordering_matches_torch(num_rows: int) -> None:
    live_len = 65536
    topk = 2048
    gen = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((num_rows, live_len), generator=gen, device="cuda")
    positions = (
        torch.rand((num_rows, live_len), generator=gen, device="cuda")
        .topk(topk // 4 + 3 * (topk // 8), dim=1)
        .indices
    )
    n_nan, n_big = topk // 4, topk // 8
    logits.scatter_(1, positions[:, :n_nan], float("nan"))
    logits.scatter_(1, positions[:, n_nan : n_nan + n_big], float("inf"))
    logits.scatter_(
        1,
        positions[:, n_nan + n_big : n_nan + 2 * n_big],
        torch.finfo(torch.float32).max,
    )
    logits.scatter_(1, positions[:, n_nan + 2 * n_big :], 1e30)
    lengths = torch.full((num_rows,), live_len, dtype=torch.int32, device="cuda")

    out = _run_persistent_topk(logits, lengths, topk, live_len)
    assert torch.all((out >= 0) & (out < live_len))
    sorted_indices = out.sort(dim=1).values
    assert torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1])
    actual = logits.gather(1, out.long()).sort(dim=1).values
    expected = logits.topk(topk, dim=1).values.sort(dim=1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@requires_cuda
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("num_rows", [5, 10, 20])
@pytest.mark.parametrize("live_len", [40000, 86016, 120000])
def test_persistent_topk_stride_independent(live_len, num_rows, topk):
    torch.manual_seed(0)
    device = "cuda"
    live = torch.randn(num_rows, live_len, device=device)
    lengths = torch.full((num_rows,), live_len, dtype=torch.int32, device=device)

    outs = {}
    for width in (524288, 131072):
        logits = torch.full((num_rows, width), POISON, device=device)
        logits[:, :live_len] = live
        out = _run_persistent_topk(logits, lengths, topk, width)
        _assert_matches_torch(logits, lengths, out, topk)
        outs[width] = out.clone()
        del logits

    assert torch.equal(outs[524288], outs[131072]), (
        "top-k indices must not depend on the padded row width"
    )


@requires_cuda
@pytest.mark.parametrize("topk", [512, 2048])
def test_persistent_topk_mixed_row_lengths(topk):
    torch.manual_seed(0)
    device = "cuda"
    live_lens = [86016, 1000, 40000, 20, 0]
    lengths = torch.tensor(live_lens, dtype=torch.int32, device=device)
    rows = len(live_lens)
    data = [torch.randn(n, device=device) for n in live_lens]

    outs = {}
    for width in (524288, 131072):
        logits = torch.full((rows, width), POISON, device=device)
        for r, n in enumerate(live_lens):
            if n:
                logits[r, :n] = data[r]
        out = _run_persistent_topk(logits, lengths, topk, width)
        _assert_matches_torch(logits, lengths, out, topk)
        outs[width] = out.clone()
        del logits

    assert torch.equal(outs[524288], outs[131072]), (
        "mixed row lengths must not depend on the padded row width"
    )


@requires_cuda
def test_persistent_topk_stride_independent_with_ties():
    """Tie-heavy data pins the output order, not only the selected set."""
    torch.manual_seed(0)
    device = "cuda"
    num_rows, live_len, topk = 5, 86016, 512
    data = torch.randint(-4, 5, (num_rows, live_len), device=device).float()
    lengths = torch.full((num_rows,), live_len, dtype=torch.int32, device=device)

    outs = {}
    for width in (524288, 131072):
        logits = torch.full((num_rows, width), POISON, device=device)
        logits[:, :live_len] = data
        out = _run_persistent_topk(logits, lengths, topk, width)
        for r in range(num_rows):
            got_raw = [i for i in out[r].tolist() if i >= 0]
            assert len(got_raw) == topk, f"row {r}: short output"
            assert all(i < live_len for i in got_raw), f"row {r}: padded index"
            assert len(set(got_raw)) == topk, f"row {r}: duplicate indices"
            selected = data[r][out[r]]
            assert selected.min() >= torch.topk(data[r], topk).values.min()
        outs[width] = out.clone()
        del logits

    assert torch.equal(outs[524288], outs[131072]), (
        "tie ordering must not depend on the padded row width"
    )
