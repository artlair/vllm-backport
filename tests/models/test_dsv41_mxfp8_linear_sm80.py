# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 native MXFP8 ([32, 32] ue8m0 block) linear route on sm8x.

Quantizes a random bf16 weight to e4m3 with one power-of-two e8m0 scale per
32x32 block (the DeepSeek-V4.1-Flash layout for attention and shared-expert
linears), pushes weight and scale through the layer's weight loaders the way
``load_weights`` does, runs ``process_weights_after_loading`` and checks the
layer output against a bf16 reference matmul on the dequantized weight.
"""

import pytest
import torch

BLOCK = 32
DTYPE = torch.bfloat16
N, K = 256, 512
M = 8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


@pytest.fixture(scope="module")
def dist_env():
    # Import the kernel registry first: pulling marlin_utils in before it
    # trips a fused_moe <-> marlin_utils import cycle in a bare interpreter.
    import vllm.model_executor.kernels.linear  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        is_fp8_marlin_supported,
    )

    if not is_fp8_marlin_supported():
        pytest.skip("FP8 Marlin is not supported on this GPU")

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _make_weight(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Random bf16 [n, k] weight whose 32x32 blocks span several octaves, so a
    misaligned scale expansion produces a large error instead of hiding in
    the quantization noise."""
    w = torch.randn(n, k, generator=gen, dtype=torch.float32)
    octave = torch.randint(-3, 2, (n // BLOCK, k // BLOCK), generator=gen)
    mag = torch.exp2(octave.float())
    w = w * mag.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return w.to(DTYPE)


def _quantize_block32(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [n, k] to e4m3 with one e8m0 scale per 32x32 block.

    Returns (q [n, k] float8_e4m3fn, scale [n/32, k/32] float8_e8m0fnu as the
    checkpoint stores it, dequant [n, k] float32).
    """
    n, k = weight.shape
    assert n % BLOCK == 0 and k % BLOCK == 0
    w = weight.float()
    blocks = w.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    # Smallest power of two with amax / scale <= 448 (e4m3 max).
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-126, 127)
    scale = torch.exp2(exp)
    scale_full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    q = (w / scale_full).to(torch.float8_e4m3fn)
    dequant = q.float() * scale_full
    e8m0 = (exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q, e8m0, dequant


def _quant_config():
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

    # DeepSeek-V4.1-Flash: fp8 e4m3 linears, ue8m0 scales, [32, 32] blocks;
    # expert_dtype defaults to "fp4" (no hf_config here), which selects the
    # e8m0 scale route exactly as the real checkpoint does.
    return DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[BLOCK, BLOCK],
    )


def _check_output(layer: torch.nn.Module, dequant: torch.Tensor, gen) -> None:
    x = (torch.randn(M, K, generator=gen, dtype=torch.float32) / K**0.5).to(DTYPE)
    x = x.cuda()
    with torch.no_grad():
        out = layer(x)
    ref = (x.float() @ dequant.cuda().T).to(DTYPE)
    assert out.shape == ref.shape == (M, dequant.shape[0])
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 2e-2, f"relative error {rel:.4f}"
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def _assert_marlin_on_sm8x(layer: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )

    major, _ = torch.cuda.get_device_capability()
    if major == 8:
        assert isinstance(layer.quant_method.kernel, MarlinMxfp8LinearKernel)


@pytest.mark.parametrize("kind", ["replicated", "column"])
def test_single_linear_matches_reference(dist_env, kind):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        ReplicatedLinear,
    )
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4Mxfp8LinearMethod

    gen = torch.Generator().manual_seed(0)
    weight = _make_weight(N, K, gen)
    q, e8m0, dequant = _quantize_block32(weight)
    assert e8m0.shape == (N // BLOCK, K // BLOCK)

    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        cls = ReplicatedLinear if kind == "replicated" else ColumnParallelLinear
        layer = cls(
            K,
            N,
            bias=False,
            quant_config=_quant_config(),
            prefix="model.layers.0.attn.wq_b",
            return_bias=False,
        )
    assert isinstance(layer.quant_method, DeepseekV4Mxfp8LinearMethod)
    assert layer.weight_block_size == [1, BLOCK]
    assert layer.weight.shape == (N, K)
    assert layer.weight_scale.shape == (N, K // BLOCK)
    assert layer.weight_scale.dtype == torch.uint8

    layer.weight.weight_loader(layer.weight, q)
    layer.weight_scale.weight_loader(layer.weight_scale, e8m0)

    expected_scale = e8m0.view(torch.uint8).repeat_interleave(BLOCK, dim=0)
    assert torch.equal(layer.weight_scale.data.cpu(), expected_scale)
    assert torch.equal(layer.weight.data.cpu().view(torch.uint8), q.view(torch.uint8))

    layer.quant_method.process_weights_after_loading(layer)
    _assert_marlin_on_sm8x(layer)
    _check_output(layer, dequant, gen)


def test_merged_column_linear_shards_expanded_scale(dist_env):
    """Fused gate/up style loading: each sub-weight's [n_i/32, k/32] scale
    lands at its own output-element offset of the expanded [N, k/32] scale."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    sizes = [N // 2, N // 2]
    gen = torch.Generator().manual_seed(1)
    parts = [_quantize_block32(_make_weight(n, K, gen)) for n in sizes]

    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        layer = MergedColumnParallelLinear(
            K,
            sizes,
            bias=False,
            quant_config=_quant_config(),
            prefix="model.layers.0.attn.fused_wqa_wkv",
            return_bias=False,
        )
    assert layer.weight_scale.shape == (N, K // BLOCK)

    for shard_id, (q, e8m0, _) in enumerate(parts):
        layer.weight.weight_loader(layer.weight, q, shard_id)
        layer.weight_scale.weight_loader(layer.weight_scale, e8m0, shard_id)

    expected_scale = torch.cat(
        [e8m0.view(torch.uint8).repeat_interleave(BLOCK, 0) for _, e8m0, _ in parts]
    )
    assert torch.equal(layer.weight_scale.data.cpu(), expected_scale)

    layer.quant_method.process_weights_after_loading(layer)
    _assert_marlin_on_sm8x(layer)
    dequant = torch.cat([d for _, _, d in parts])
    _check_output(layer, dequant, gen)


def test_bmm_layer_keeps_raw_layout(dist_env):
    """``wo_a`` (is_bmm) is dequantized by the attention einsum from the raw
    weight and scale; Marlin must leave it unpacked."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    gen = torch.Generator().manual_seed(2)
    q, e8m0, dequant = _quantize_block32(_make_weight(N, K, gen))

    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        layer = ColumnParallelLinear(
            K,
            N,
            bias=False,
            quant_config=_quant_config(),
            prefix="model.layers.0.attn.wo_a",
            return_bias=False,
        )
    layer.is_bmm = True
    layer.bmm_batch_size = 4
    layer.weight.weight_loader(layer.weight, q)
    layer.weight_scale.weight_loader(layer.weight_scale, e8m0)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (N, K)
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_scale.shape == (N, K // BLOCK)
    got = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
    torch.testing.assert_close(got, dequant.cuda().to(DTYPE), rtol=1e-2, atol=1e-2)
