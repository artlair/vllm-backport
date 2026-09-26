# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_builder_req_id_per_token_matches_host_repeat():
    """The builder's device req_id_per_token equals the host repeat of the
    query lengths it replaced, with 0 for padding tokens past the last query
    and zero-length (padding) requests owning no token."""
    from types import SimpleNamespace

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        XPUMLASparseMetadataBuilder,
    )

    builder = object.__new__(XPUMLASparseMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.topk_tokens = 2048
    builder.req_id_per_token_buffer = torch.empty(
        8192, dtype=torch.int32, device="cuda"
    )
    gen = torch.Generator().manual_seed(0)
    for it in range(300):
        num_reqs = int(torch.randint(1, 33, (1,), generator=gen))
        max_q = (300, 5, 2)[it % 3]
        query_lens = torch.randint(0, max_q, (num_reqs,), generator=gen)
        query_lens[0] += 1
        pad_reqs = int(torch.randint(0, 4, (1,), generator=gen))
        query_lens = torch.cat([query_lens, torch.zeros(pad_reqs, dtype=torch.long)])
        qsl = torch.zeros(query_lens.numel() + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(query_lens, 0)
        num_tokens = int(qsl[-1]) + int(torch.randint(0, 20, (1,), generator=gen))
        builder.req_id_per_token_buffer.fill_(-7)
        cam = CommonAttentionMetadata(
            query_start_loc=qsl.cuda(),
            query_start_loc_cpu=qsl,
            seq_lens=torch.ones(qsl.numel() - 1, dtype=torch.int32, device="cuda"),
            num_reqs=qsl.numel() - 1,
            num_actual_tokens=num_tokens,
            max_query_len=int(query_lens.max()),
            max_seq_len=1,
            block_table_tensor=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
        )
        got = builder.build(0, cam).req_id_per_token
        expected = torch.zeros(num_tokens, dtype=torch.int32)
        mapped = torch.repeat_interleave(
            torch.arange(query_lens.numel(), dtype=torch.int32), query_lens
        )
        expected[: mapped.numel()] = mapped[:num_tokens]
        assert torch.equal(got.cpu(), expected)
