# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check streamed MXFP4 repacking against the allocating Marlin MoE path."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe  # noqa: F401 (import-order guard)
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repacked_nk,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    _repack_marlin_experts,
    _repack_mxfp4_marlin_scales,
    is_fp4_marlin_supported,
    marlin_permute_scales,
    mxfp4_marlin_process_scales,
    prepare_moe_fp4_layer_for_marlin,
    prepare_moe_mxfp4_layer_for_marlin,
)

E, K, N = 3, 256, 128


def _noncontiguous(x: torch.Tensor) -> torch.Tensor:
    padded = torch.empty(
        (*x.shape[:-1], x.shape[-1] + 16), dtype=x.dtype, device=x.device
    )
    view = padded[..., : x.shape[-1]]
    view.copy_(x)
    assert not view.is_contiguous()
    return view


def _reference_marlin_weight(packed: torch.Tensor, is_a_8bit: bool) -> torch.Tensor:
    """Pack nibbles in the thread/warp order of gptq_marlin_repack_kernel."""
    size_n, packed_k = packed.shape
    size_k = packed_k * 2
    values = (packed.to(torch.int64).unsqueeze(-1) >> torch.tensor([0, 4])) & 15
    values = values.flatten(-2).T
    tile_k, tile_n = (32, 32) if is_a_8bit else (16, 64)
    tiles = values.reshape(size_k // tile_k, tile_k, size_n // tile_n, tile_n)
    tiles = tiles.permute(0, 2, 1, 3)

    word = torch.arange(128)
    thread, warp = word // 4, word % 4
    if is_a_8bit:
        rows = (thread % 4 * 4)[:, None] + torch.tensor([0, 16, 1, 17, 2, 18, 3, 19])
        cols = (warp // 2 * 16 + thread // 4 + warp % 2 * 8)[:, None]
    else:
        rows = (thread % 4 * 2)[:, None] + torch.tensor([0, 8, 0, 8, 1, 9, 1, 9])
        cols = (warp * 16 + thread // 4)[:, None] + torch.tensor(
            [0, 0, 8, 8, 0, 0, 8, 8]
        )
    interleaved = tiles[:, :, rows, cols]
    words = (interleaved << (torch.arange(8) * 4)).sum(dim=-1).to(torch.int32)
    return words.reshape(size_k // 16, size_n * 2)


@pytest.mark.parametrize("size_n, size_k", [(2 * N, K), (K, N)])
@pytest.mark.parametrize("is_a_8bit", [False, True])
def test_streamed_weight_writes_stay_within_expert_on_cpu(
    monkeypatch, size_n, size_k, is_a_8bit
):
    """Every write consumes a detached expert and leaves adjacent bytes intact."""
    torch.manual_seed(0)
    backing = torch.randint(0, 256, (E + 2, size_n, size_k // 2), dtype=torch.uint8)
    weight = backing[1:-1]
    original = backing.clone()
    observed = backing.clone()
    expected = [_reference_marlin_weight(expert, is_a_8bit) for expert in weight]
    out = weight.view(torch.int32).view(E, size_k // 16, size_n * 2)
    expert_bytes = size_n * size_k // 2
    for i in range(E):
        assert out[i].data_ptr() == weight.data_ptr() + i * expert_bytes
        assert out[i].numel() * out.element_size() == expert_bytes
        assert marlin_repacked_nk(out[i], num_bits=4) == (size_n, size_k)
    calls = 0

    def repack_expert(b_q_weight, size_k, size_n, num_bits, is_a_8bit):
        nonlocal calls
        # Check the previous write and all not-yet-consumed experts together.
        assert torch.equal(backing, observed)
        assert b_q_weight.is_contiguous() and num_bits == 4
        assert (
            b_q_weight.untyped_storage().data_ptr()
            != weight.untyped_storage().data_ptr()
        )
        source = original[calls + 1]
        assert torch.equal(b_q_weight, source.view(torch.int32).T.contiguous())
        result = _reference_marlin_weight(
            b_q_weight.T.contiguous().view(torch.uint8), is_a_8bit
        )
        observed[calls + 1].copy_(expected[calls].view(torch.uint8).view_as(source))
        calls += 1
        return result

    monkeypatch.setattr(marlin_utils_fp4.ops, "gptq_marlin_repack", repack_expert)
    actual = _repack_marlin_experts(weight, size_n, size_k, is_a_8bit, out=out)

    assert calls == E and actual is out
    assert torch.equal(actual, torch.stack(expected))
    assert torch.equal(backing, observed)


def _whole_layer_repack(tensors, param_dtype, input_dtype):
    """Use the unchanged whole-layer path as the packed-byte reference."""
    layer = torch.nn.Module()
    names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    for name, tensor in zip(names, tensors):
        layer.register_parameter(
            name, torch.nn.Parameter(tensor.clone(), requires_grad=False)
        )
    layer.moe_config = SimpleNamespace(
        num_local_experts=E, hidden_dim=K, intermediate_size_per_partition=N
    )
    layer.params_dtype = param_dtype
    prepare_moe_fp4_layer_for_marlin(layer, input_dtype=input_dtype)
    return [getattr(layer, name).data for name in names]


@pytest.mark.parametrize("size_n, size_k", [(2 * N, K), (K, N)])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("input_dtype", [None, torch.float8_e4m3fn])
def test_streamed_scales_match_whole_layer_on_cpu(
    size_n, size_k, contiguous, param_dtype, input_dtype
):
    """Per-expert conversion preserves both activation-specific e8m0 layouts."""
    torch.manual_seed(0)
    scales = torch.randint(119, 132, (E, size_n, size_k // 32), dtype=torch.uint8)
    converted = scales.view(torch.float8_e8m0fnu).to(param_dtype)
    expected = torch.stack(
        [
            mxfp4_marlin_process_scales(
                marlin_permute_scales(
                    scale.T, size_k, size_n, 32, is_a_8bit=input_dtype is not None
                ),
                input_dtype=input_dtype,
            )
            for scale in converted
        ]
    )
    source = scales if contiguous else _noncontiguous(scales)
    actual = _repack_mxfp4_marlin_scales(
        source, size_n, size_k, param_dtype, input_dtype
    )

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert (
        actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    ) == contiguous
    if not contiguous:
        assert torch.equal(source, scales)


@pytest.mark.skipif(
    not is_fp4_marlin_supported(), reason="FP4 Marlin requires a supported CUDA device"
)
@pytest.mark.parametrize("contiguous_weights", [True, False])
@pytest.mark.parametrize("contiguous_scales", [True, False])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("input_dtype", [None, torch.float8_e4m3fn])
def test_streamed_repack_matches_whole_layer(
    monkeypatch, contiguous_weights, contiguous_scales, param_dtype, input_dtype
):
    """Reusing source storage must preserve weight and permuted scale bytes."""
    torch.manual_seed(0)
    tensors = [
        torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device="cuda"),
        torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device="cuda"),
        torch.randint(119, 132, (E, 2 * N, K // 32), dtype=torch.uint8, device="cuda"),
        torch.randint(119, 132, (E, K, N // 32), dtype=torch.uint8, device="cuda"),
    ]
    want = _whole_layer_repack(tensors, param_dtype, input_dtype)
    contiguous = [contiguous_weights] * 2 + [contiguous_scales] * 2
    sources = [
        tensor if reuse else _noncontiguous(tensor)
        for tensor, reuse in zip(tensors, contiguous)
    ]
    unchanged = [
        tensor.clone() if not reuse else None
        for tensor, reuse in zip(sources, contiguous)
    ]

    # FP8 execution requires SM89/SM12x, but repacking works on older devices.
    monkeypatch.setattr(marlin_utils_fp4, "get_marlin_input_dtype", lambda: input_dtype)
    layer = SimpleNamespace(params_dtype=param_dtype)
    got = prepare_moe_mxfp4_layer_for_marlin(layer, *sources, None, None)

    assert got[4:] == (None, None)
    for actual, expected, source, reuse, original in zip(
        got[:4], want, sources, contiguous, unchanged
    ):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        aliases = (
            actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
        assert aliases == reuse
        if original is not None:
            assert torch.equal(source, original)
