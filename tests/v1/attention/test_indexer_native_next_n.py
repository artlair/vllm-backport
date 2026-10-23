# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _paged_mqa_logits_schedule_slots
from vllm.v1.attention.backends.mla import indexer

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)


@pytest.mark.parametrize(
    "family,expected_native",
    [
        # SM90 gained next_n=4 (MTP=3) via 2-CTA multicast, but never 3.
        (9, {1, 2, 4}),
        # SM100 schedules any next_n with multi-atom tiles.
        (10, {1, 2, 3, 4, 5, 8}),
        # SM120 advertises multi-atom too but is unvalidated on hardware, so
        # it stays on the conservative gate. Loosen it only with measurements.
        (12, {1, 2}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, expected_native):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize(
    "cuda,deep_gemm", [(False, True), (True, False), (False, False)]
)
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


def _adaptive_config(num_speculative_tokens: int, enabled: bool = True):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        speculative_config=SimpleNamespace(enable_adaptive_verification=enabled),
    )


@pytest.mark.parametrize(
    "family,expected",
    [
        # SM8x: the flattened rows go through the Triton paged-MQA logits kernel.
        (8, True),
        # Hopper (DeepGEMM) and Blackwell keep their own flattened route.
        (9, False),
        (10, False),
        (12, False),
    ],
)
def test_flattened_triton_query_lens_is_sm8x_only(monkeypatch, family, expected):
    _set_arch(monkeypatch, family, deep_gemm=False)
    assert indexer._supports_flattened_triton_query_lens() == expected
    # The Hopper DeepGEMM flatten gate is unchanged.
    _set_arch(monkeypatch, family, deep_gemm=True)
    assert indexer._supports_flattened_device_query_lens() is (family == 9)


@pytest.mark.parametrize("num_speculative_tokens", [1, 4])
def test_adaptive_flattens_sm8x_even_when_native_decode_is_legal(
    monkeypatch, num_speculative_tokens
):
    """k=1 (next_n=2) must not keep the CPU-decided native decode branch.

    Without DeepGEMM, next_n=2 is native-capable, so a k=1 adaptive run would
    take the native branch whose ``requires_padding`` flag is decided from the
    CPU query lengths at capture and frozen there. The adaptive flatten gate has
    to cover that case so capture and the device-built rows agree.
    """
    _set_arch(monkeypatch, 8, deep_gemm=False)
    config = _adaptive_config(num_speculative_tokens)
    assert indexer._supports_native_decode(1 + num_speculative_tokens) is (
        num_speculative_tokens == 1
    )
    assert indexer._use_flattening(config) is True

    # Standard (non-adaptive) decode keeps the cheapest legal route.
    config.speculative_config.enable_adaptive_verification = False
    assert indexer._use_flattening(config) is (num_speculative_tokens > 1)


@pytest.mark.parametrize("family,expected", [(8, True), (9, False), (10, False)])
def test_v41_indexer_claims_device_query_lens_only_on_sm8x(
    monkeypatch, family, expected
):
    """Capability is per model and per arch, not a blanket sparse-indexer flag."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV4IndexerBackend,
        DeepseekV32IndexerBackend,
        DeepseekV41IndexerBackend,
    )

    _set_arch(monkeypatch, family, deep_gemm=False)
    assert (
        DeepseekV41IndexerBackend.supports_device_cpu_query_lens_mismatch() is expected
    )
    # The V3.2 and V4 indexers never take the SM8x flattened route with a
    # device-decided split, so they stay rejected.
    assert not DeepseekV32IndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert not DeepseekV4IndexerBackend.supports_device_cpu_query_lens_mismatch()
