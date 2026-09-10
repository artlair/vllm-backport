# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 repack: the MXFP4 Marlin MoE repack streams one expert at a time
back into the source storage instead of holding a second copy of the layer.

Checks that the streamed path is bit-identical to the whole-layer path
(prepare_moe_fp4_layer_for_marlin) and that a fused-MoE forward through the
streamed layer matches a bf16 dequantised reference.

Run `pytest tests/kernels/moe/test_mxfp4_marlin_repack.py`.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe  # noqa: F401  (import-order guard)
from tests.kernels.utils import torch_moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
    prepare_moe_fp4_layer_for_marlin,
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types

E, K, N = 8, 512, 256  # experts, hidden, intermediate per rank
MXFP4_BLOCK = 32

requires_fp4_marlin = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fp4_marlin_supported(),
    reason="FP4 Marlin is not supported on this device",
)


def _rand_mxfp4(e: int, rows: int, cols: int, device: str):
    """Random packed MXFP4 nibbles (E, rows, cols / 2) and e8m0 scales
    (E, rows, cols / 32) in the loader's uint8 form."""
    packed = torch.randint(
        0, 256, (e, rows, cols // 2), dtype=torch.uint8, device=device
    )
    # 2^-6 .. 2^-2 so the dequantised weights and the MoE output are O(1)
    scales = torch.randint(
        121, 125, (e, rows, cols // MXFP4_BLOCK), dtype=torch.uint8, device=device
    )
    return packed, scales


def _dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype):
    """Mirror of rand_marlin_weight_mxfp4_like's decode: low nibble first."""

    def nibble(x: torch.Tensor) -> torch.Tensor:
        x = (x & 0b10000000) | ((x & 0b01110000) >> 2)
        return x.view(torch.float8_e4m3fn).to(dtype) * (2**6)

    w = torch.stack([nibble(packed << 4), nibble(packed)], dim=-1).flatten(-2)
    s = scales.view(torch.float8_e8m0fnu).to(dtype)
    return w * s.repeat_interleave(MXFP4_BLOCK, dim=-1)


def _noncontiguous(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty((*x.shape[:-1], x.shape[-1] + 16), dtype=x.dtype, device=x.device)
    y = y[..., : x.shape[-1]]
    y.copy_(x)
    assert not y.is_contiguous()
    return y


def _whole_layer_repack(w13, w2, w13_scale, w2_scale, input_dtype):
    """The pre-existing whole-layer path: a per-expert list plus torch.cat,
    run on a throwaway module so the inputs are untouched."""
    layer = torch.nn.Module()
    for name, t in (
        ("w13_weight", w13),
        ("w2_weight", w2),
        ("w13_weight_scale", w13_scale),
        ("w2_weight_scale", w2_scale),
    ):
        layer.register_parameter(
            name, torch.nn.Parameter(t.clone(), requires_grad=False)
        )
    layer.moe_config = SimpleNamespace(
        num_local_experts=E, hidden_dim=K, intermediate_size_per_partition=N
    )
    layer.params_dtype = torch.bfloat16
    prepare_moe_fp4_layer_for_marlin(layer, input_dtype=input_dtype)
    return (
        layer.w13_weight.data,
        layer.w2_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight_scale.data,
    )


@requires_fp4_marlin
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("input_dtype", [None, torch.float8_e4m3fn])
def test_streamed_repack_matches_whole_layer(monkeypatch, contiguous, input_dtype):
    torch.manual_seed(0)
    w13, w13_scale = _rand_mxfp4(E, 2 * N, K, "cuda")
    w2, w2_scale = _rand_mxfp4(E, K, N, "cuda")
    want = _whole_layer_repack(w13, w2, w13_scale, w2_scale, input_dtype)

    # The pure function reads the Marlin activation dtype from the env; the
    # fp8 flavour is refused on non-SM89 devices, so pin it directly.
    monkeypatch.setattr(
        marlin_utils_fp4, "get_marlin_input_dtype", lambda prefix=None: input_dtype
    )
    srcs = [w13, w2, w13_scale, w2_scale]
    if not contiguous:
        srcs = [_noncontiguous(x) for x in srcs]
    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    got = prepare_moe_mxfp4_layer_for_marlin(layer, *srcs, None, None)

    assert got[4] is None and got[5] is None
    for g, w, src in zip(got[:4], want, srcs):
        assert g.dtype == w.dtype and g.shape == w.shape
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))
        # Contiguous inputs are repacked in place (the memory win); the
        # non-contiguous fallback allocates a fresh output.
        aliases = g.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
        assert aliases == contiguous


@requires_fp4_marlin
@pytest.mark.parametrize("m", [1, 64])
def test_streamed_layer_forward_matches_bf16_reference(m):
    torch.manual_seed(0)
    topk = 2
    dtype = torch.bfloat16
    w13, w13_scale = _rand_mxfp4(E, 2 * N, K, "cuda")
    w2, w2_scale = _rand_mxfp4(E, K, N, "cuda")
    w13_ref = _dequant_mxfp4(w13, w13_scale, dtype)
    w2_ref = _dequant_mxfp4(w2, w2_scale, dtype)

    layer = SimpleNamespace(params_dtype=dtype)
    qw13, qw2, s13, s2, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer, w13, w2, w13_scale, w2_scale, None, None
    )

    a = torch.randn((m, K), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, E), device="cuda", dtype=dtype)
    with set_current_vllm_config(VllmConfig()):
        ref = torch_moe(a, w13_ref, w2_ref, score, topk)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
        out = fused_marlin_moe(
            a,
            qw13,
            qw2,
            None,
            None,
            s13,
            s2,
            topk_weights,
            topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=E,
            input_dtype=dtype,
        )

    assert ref.abs().mean() > 1e-2  # the tolerance below is not vacuous
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=0)
