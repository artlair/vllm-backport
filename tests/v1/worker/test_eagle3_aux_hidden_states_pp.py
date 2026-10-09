# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aux hidden state relay across pipeline stages (upstream #57197).

The MiniMax-M3 flavor is ported from upstream and exercises the
model-agnostic relay machinery (EagleModelMixin slot layout, pack/collect,
reserve) with exact values. The GLM5Next flavor adapts it to this fork's
DFlash2 lane target, whose taps are captured at layer ENTRY (so the slot
base is bisect_left) and whose mHC stream must be contracted at PP entry
boundaries.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.layers.mhc import hc_contract, hc_expand
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.models.glm5next.nvidia import model as glm5next
from vllm.models.glm5next.nvidia.model import Glm5NextModel
from vllm.models.minimax_m3.nvidia import model as minimax_m3
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)


def test_aux_layers_are_sorted_and_deduplicated():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24, 48))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


def test_mimo_does_not_inherit_aux_hidden_state_pp_support():
    inner = MiMoModel.__new__(MiMoModel)
    target = SimpleNamespace(model=inner)

    assert not inner.supports_aux_hidden_states_over_pp
    with pytest.raises(ValueError, match="does not support eagle3"):
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


class _PPAuxLayer(torch.nn.Module):
    """Aliases and mutates residual in place like MiniMaxM3DecoderLayer."""

    ffn_all_reduce_deferred = False

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden_states, residual):
        total = hidden_states if residual is None else hidden_states + residual
        residual = hidden_states if residual is None else residual
        residual.copy_(total + self.index + 1)
        return torch.full_like(hidden_states, 10 * (self.index + 1)), residual


class _PPAuxNorm(torch.nn.Module):
    """Clobbers its inputs so aux taps that alias them would fail."""

    def forward(self, hidden_states, residual):
        output = hidden_states + residual
        hidden_states.fill_(-101)
        residual.fill_(-103)
        return output, residual


@pytest.fixture
def minimax_pp_stage(monkeypatch):
    """Use the real MiniMax constructor/forward and aux helpers on CPU tensors."""
    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    bounds = [0, 3]
    monkeypatch.setattr(minimax_m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        minimax_m3,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        minimax_m3, "MiniMAXGemmaRMSNorm", lambda *args, **kwargs: _PPAuxNorm()
    )
    monkeypatch.setattr(
        minimax_m3,
        "MiniMaxM3DecoderLayer",
        lambda **kwargs: _PPAuxLayer(int(kwargs["prefix"].rsplit(".", 1)[1])),
    )

    def make_layers(num_layers, build, prefix):
        return (
            *bounds,
            torch.nn.ModuleList(
                build(f"{prefix}.{index}")
                if bounds[0] <= index < bounds[1]
                else torch.nn.Identity()
                for index in range(num_layers)
            ),
        )

    monkeypatch.setattr(minimax_m3, "make_layers", make_layers)

    def make_stage(start, end, taps, world_size=2):
        bounds[:] = [start, end]
        group.world_size = world_size
        group.is_first_rank = start == 0
        group.is_last_rank = end == 6
        config = SimpleNamespace(
            vocab_size=16, hidden_size=6144, num_hidden_layers=6, rms_norm_eps=1e-6
        )
        model = minimax_m3.MiniMaxM3Model(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=config),
                quant_config=None,
                speculative_config=None,
                use_v2_model_runner=False,
            )
        )
        model._set_aux_hidden_state_layers(taps)
        target = SimpleNamespace(
            model=model,
            make_empty_intermediate_tensors=model.make_empty_intermediate_tensors,
        )
        reserve_aux_intermediate_tensor_slots(target)
        return model, target

    return make_stage


# Input 7; _PPAuxLayer i adds 11 * (i + 1) to hidden + residual.
def _expected_aux_value(tap):
    return 7 + 11 * tap * (tap + 1) // 2


def _pp_aux_state(value):
    return torch.full((2, 6144), float(value), device="cpu")


@pytest.mark.parametrize("taps", [(0, 3, 6), (0, 1, 3), (4, 5, 6)])
def test_minimax_pp2_preserves_ordered_global_aux_taps(minimax_pp_stage, taps):
    first, _ = minimax_pp_stage(0, 3, taps)
    sent = first(None, None, None, _pp_aux_state(7))
    last, target = minimax_pp_stage(3, 6, taps)
    reserved = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert set(sent.tensors) == set(reserved.tensors)
    result = last(None, None, sent)
    assert isinstance(result, tuple)
    output, aux = result
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))


def test_minimax_missing_upstream_aux_slot_is_not_silently_dropped(minimax_pp_stage):
    model, _ = minimax_pp_stage(3, 6, (0, 3, 6))
    incoming = IntermediateTensors(
        {
            "hidden_states": _pp_aux_state(30),
            "residual": _pp_aux_state(43),
            "aux_hidden_states_0": _pp_aux_state(7),
        }
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
        model(None, None, incoming)


@pytest.mark.parametrize("taps", [(), (0, 3, 6)])
def test_minimax_pp1_output_and_aux_taps(minimax_pp_stage, taps):
    model, _ = minimax_pp_stage(0, 6, taps, world_size=1)
    output = model(None, None, None, _pp_aux_state(7))
    if taps:
        output, aux = output
        for hidden, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


def test_minimax_pp2_without_aux_keeps_hidden_and_residual_transport(
    minimax_pp_stage,
):
    first, _ = minimax_pp_stage(0, 3, ())
    sent = first(None, None, None, _pp_aux_state(7))
    assert set(sent.tensors) == {"hidden_states", "residual"}
    torch.testing.assert_close(sent["hidden_states"], _pp_aux_state(30))
    torch.testing.assert_close(sent["residual"], _pp_aux_state(43))
    last, target = minimax_pp_stage(3, 6, ())
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states", "residual"}
    output = last(None, None, sent)
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


# ---------------------------------------------------------------------------
# GLM5Next flavor: entry-capture taps, bisect_left slot base, mHC contraction
# at PP entry boundaries. Stubs keep everything on CPU with exact values.
# ---------------------------------------------------------------------------


def _cpu_mhc_post():
    from vllm.model_executor.layers.mhc import MHCPostOp

    op = MHCPostOp()
    op._forward_method = op.forward_native
    return op


class _GLMPPStubLayer(torch.nn.Module):
    """Deterministic mHC-shaped stand-in for Glm5NextDecoderLayer.

    Mirrors the real layer's shape contract: ``hidden_states`` leaving a
    layer is the 2-D [s, H] transform output, ``residual`` is the widened
    [s, n, H] stream, and post/comb are zero/identity mixers so a pending
    tap completes to exactly the residual stream. Each layer adds
    ``index + 1`` to both. The last global layer materializes the deferred
    hc_post and contracts, mirroring the real mHC tail.
    """

    def __init__(self, index: int, num_layers: int, n: int):
        super().__init__()
        self.index = index
        self.num_layers = num_layers
        self.n = n
        self.hc_post = _cpu_mhc_post()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ):
        if post is None:
            if self.index == 0:
                # Layer 0 re-enters via the standalone hc_pre: expand the
                # embedding into the widened stream, then mix it down.
                residual = hc_expand(hidden_states, self.n)
            else:
                # PP entry: the shipped stream is the widened residual.
                residual = hidden_states
            x = residual.mean(dim=1) + (self.index + 1)
            residual = residual + (self.index + 1)
        else:
            x = hidden_states + (self.index + 1)
            residual = residual + (self.index + 1)
        if self.index == self.num_layers - 1:
            out = self.hc_post(x, residual, post, comb)
            return hc_contract(out, self.n), None, None, None
        s = hidden_states.shape[0]
        post_out = torch.zeros(s, self.n, 1)
        comb_out = torch.eye(self.n).expand(s, -1, -1)
        return x, residual, post_out, comb_out


def _glm_expected_aux_value(tap: int) -> float:
    """Tap t is the stream entering layer t: embed plus sum(1..t)."""
    return 7 + tap * (tap + 1) // 2


@pytest.fixture
def glm_pp_stage(monkeypatch, default_vllm_config):
    """Build real Glm5NextModel stages with stubbed layers, on CPU tensors.

    The default_vllm_config context satisfies the CustomOp registry when the
    model constructs its MHCPostOp instances. The IR rms_norm priority is
    scoped back to the native impl: an in-process engine boot earlier in the
    same pytest session (e.g. the DFlash2 e2e smoke) flips the process-global
    priority to the CUDA-only vllm_c kernel, which CPU tensors cannot run.
    """
    from vllm.ir.ops.layernorm import rms_norm as ir_rms_norm

    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    bounds = [0, 3]
    num_layers = 6
    n_streams = 2

    monkeypatch.setattr(glm5next, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        glm5next, "VocabParallelEmbedding", lambda *a, **k: torch.nn.Identity()
    )
    monkeypatch.setattr(glm5next, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(glm5next, "MHCPostOp", lambda *a, **k: _cpu_mhc_post())

    def make_stage(start, end, taps, world_size=2):
        bounds[:] = [start, end]
        group.world_size = world_size
        group.is_first_rank = start == 0
        group.is_last_rank = end == num_layers

        def build_layer(**kwargs):
            index = int(kwargs["prefix"].rsplit(".", 1)[1])
            return _GLMPPStubLayer(index, num_layers, n_streams)

        def make_layers(num, build, prefix):
            return (
                bounds[0],
                bounds[1],
                torch.nn.ModuleList(
                    build(prefix=f"{prefix}.{index}")
                    if bounds[0] <= index < bounds[1]
                    else torch.nn.Identity()
                    for index in range(num)
                ),
            )

        monkeypatch.setattr(glm5next, "make_layers", make_layers)
        monkeypatch.setattr(glm5next, "Glm5NextDecoderLayer", build_layer)

        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            index_topk=None,
            mhc=True,
            mhc_num_residual_streams=n_streams,
        )
        model = Glm5NextModel(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=config),
                parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
            )
        )
        model._set_aux_hidden_state_layers(taps)
        target = SimpleNamespace(
            model=model,
            make_empty_intermediate_tensors=model.make_empty_intermediate_tensors,
        )
        reserve_aux_intermediate_tensor_slots(target)
        return model, target

    with ir_rms_norm.set_priority(["native"]):
        yield make_stage


def _glm_aux_state(value: float) -> torch.Tensor:
    return torch.full((2, 8), float(value))


def test_glm5next_slot_base_uses_entry_capture_semantics(glm_pp_stage):
    """GLM taps capture at layer ENTRY, so a tap at a stage's start_layer is
    produced locally: the upstream tap count must be bisect_left, not the
    capture-after bisect_right."""
    # Stage [3, 6) with taps (2, 3, 6): tap 2 is upstream, tap 3 is this
    # stage's own entry capture.
    _, target = glm_pp_stage(3, 6, (2, 3, 6))
    model = target.model
    assert model._aux_slot_base_cached == 1
    assert model._aux_upstream_total_cached == 1
    # First rank produces everything from scratch.
    first, _ = glm_pp_stage(0, 3, (2, 3, 6))
    assert first._aux_slot_base_cached == 0


def test_glm5next_pp2_relays_entry_taps_with_contracted_shapes(glm_pp_stage):
    taps = (2, 3, 6)
    first, _ = glm_pp_stage(0, 3, taps)
    embed = _glm_aux_state(7)
    sent = first(None, torch.arange(2), None, inputs_embeds=embed)
    # Stage 0 captures tap 2 locally (slot 0); tap 3 and 6 belong to stage 1.
    assert set(sent.tensors) == {"hidden_states", "aux_hidden_states_0"}
    # The shipped stream is the widened mHC state.
    assert sent["hidden_states"].shape == (2, 2, 8)
    # The tap must survive the sender's in-flight state: it was cloned or
    # materialized at capture, not aliased into the shipped stream.
    torch.testing.assert_close(
        sent["aux_hidden_states_0"], _glm_aux_state(_glm_expected_aux_value(2))
    )

    last, target = glm_pp_stage(3, 6, taps)
    reserved = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert set(sent.tensors) == set(reserved.tensors)
    result = last(None, torch.arange(2), sent)
    assert isinstance(result, tuple)
    output, aux = result
    assert output.shape == (2, 8)
    assert len(aux) == len(taps)
    for hidden, tap in zip(aux, taps, strict=True):
        # mHC-contracted: the widened stream would be (2, 2, 8).
        assert hidden.shape == (2, 8)
        torch.testing.assert_close(hidden, _glm_aux_state(_glm_expected_aux_value(tap)))


def test_glm5next_pp3_middle_stage_relays_and_packs(glm_pp_stage):
    """A middle stage must forward upstream slots and pack its own taps at
    the continuing slot index; the drafting stage sees every tap in order."""
    taps = (1, 2, 4, 6)
    stage0, _ = glm_pp_stage(0, 2, taps)
    embed = _glm_aux_state(7)
    sent0 = stage0(None, torch.arange(2), None, inputs_embeds=embed)
    assert set(sent0.tensors) == {"hidden_states", "aux_hidden_states_0"}

    stage1, _ = glm_pp_stage(2, 4, taps)
    out1 = stage1(None, torch.arange(2), sent0)
    assert set(out1.tensors) == {"hidden_states", "aux_hidden_states_1"}
    torch.testing.assert_close(
        out1["aux_hidden_states_1"], _glm_aux_state(_glm_expected_aux_value(2))
    )

    # The PPHandler relay re-attaches the upstream slot for the next hop.
    from vllm.v1.worker.gpu.pp_utils import PPHandler

    fake_handler = SimpleNamespace(aux_hidden_state_relay_keys=("aux_hidden_states_0",))
    relayed = PPHandler.relay_aux_hidden_states(fake_handler, sent0, out1)
    assert set(relayed.tensors) == {
        "hidden_states",
        "aux_hidden_states_0",
        "aux_hidden_states_1",
    }

    stage2, target = glm_pp_stage(4, 6, taps)
    reserved = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert set(reserved.tensors) == set(relayed.tensors)
    output, aux = stage2(None, torch.arange(2), relayed)
    assert len(aux) == 4
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _glm_aux_state(_glm_expected_aux_value(tap)))


def test_glm5next_missing_upstream_aux_slot_is_not_silently_dropped(glm_pp_stage):
    model, _ = glm_pp_stage(4, 6, (1, 2, 4, 6))
    incoming = IntermediateTensors(
        {
            "hidden_states": torch.full((2, 2, 8), 30.0),
            "aux_hidden_states_0": _glm_aux_state(7),
        }
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
        model(None, torch.arange(2), incoming)


@pytest.mark.parametrize("taps", [(), (1, 4, 6)])
def test_glm5next_pp1_output_and_aux_taps(glm_pp_stage, taps):
    model, _ = glm_pp_stage(0, 6, taps, world_size=1)
    output = model(None, torch.arange(2), None, inputs_embeds=_glm_aux_state(7))
    if taps:
        output, aux = output
        assert len(aux) == len(taps)
        for hidden, tap in zip(aux, taps, strict=True):
            assert hidden.shape == (2, 8)
            torch.testing.assert_close(
                hidden, _glm_aux_state(_glm_expected_aux_value(tap))
            )
    assert output.shape == (2, 8)


def test_glm5next_pp2_without_aux_keeps_stream_transport(glm_pp_stage):
    first, _ = glm_pp_stage(0, 3, ())
    sent = first(None, torch.arange(2), None, inputs_embeds=_glm_aux_state(7))
    assert set(sent.tensors) == {"hidden_states"}
    last, target = glm_pp_stage(3, 6, ())
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states"}
    output = last(None, torch.arange(2), sent)
    assert output.shape == (2, 8)


def test_glm5next_declares_aux_over_pp_support():
    assert Glm5NextModel.supports_aux_hidden_states_over_pp is True
    assert issubclass(Glm5NextModel, EagleModelMixin)
