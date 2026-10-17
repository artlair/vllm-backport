# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    aux_hidden_state_relay_keys,
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


HC, HIDDEN, NUM_TOKENS, NUM_LAYERS = 4, 8, 3, 10
# DSpark target layers in V4.1's capture-before semantics: layer L's aux state
# is the stream entering L, i.e. captured after layer L-1 runs.
V41_AUX_LAYERS = (7, 8, 9)


class _FakeV41Layer(torch.nn.Module):
    """Deterministic, order-sensitive stand-in for a V4.1 decoder layer."""

    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def forward(
        self,
        x,
        positions,
        input_ids,
        pre_mix,
        post_mix,
        res_mix,
        residual,
        engram_hashes=None,
        engram_mask=None,
        attn_input_out=None,
    ):
        if residual is None:
            residual = x if x.dim() == 3 else x.unsqueeze(1).expand(-1, HC, -1)
        residual = residual * 1.5 + self.layer_id
        zeros = torch.zeros(x.shape[0], HIDDEN)
        return zeros, residual, None, None, torch.full((x.shape[0], HC), 0.25)


def _v41_stage(monkeypatch, partition: list[int], rank: int):
    from vllm.distributed import parallel_state
    from vllm.models.deepseek_v4_1.nvidia import model as v41

    pp = SimpleNamespace(
        world_size=len(partition),
        rank_in_group=rank,
        is_first_rank=rank == 0,
        is_last_rank=rank == len(partition) - 1,
    )
    monkeypatch.setattr(v41, "get_pp_group", lambda: pp)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        v41, "mhc_post_tilelang", lambda x, residual, post_mix, res_mix: residual
    )
    monkeypatch.setattr(v41, "hc_collapse_triton", lambda h, pre_mix: h.mean(dim=1))

    inner = v41.DeepseekV4Model.__new__(v41.DeepseekV4Model)
    torch.nn.Module.__init__(inner)
    inner.start_layer = sum(partition[:rank])
    inner.end_layer = inner.start_layer + partition[rank]
    inner.layers = torch.nn.ModuleList(_FakeV41Layer(i) for i in range(NUM_LAYERS))
    inner.embed_tokens = torch.nn.Embedding(16, HIDDEN)
    with torch.no_grad():
        inner.embed_tokens.weight.copy_(torch.arange(16 * HIDDEN).view(16, HIDDEN))
    inner.norm = torch.nn.Identity()
    inner.use_mega_moe = False
    inner.use_sequence_parallel = False
    inner.engram_hash = None
    inner._mtp_hidden_buffer = None
    # The fake layers share no attention state, so nothing is relayed.
    inner.pp_relay_sources = torch.nn.ModuleDict()
    inner.relay_out_ids = ()
    inner._set_aux_hidden_state_layers(V41_AUX_LAYERS)
    return inner, SimpleNamespace(model=inner)


def _run_v41_pipeline(monkeypatch, partition: list[int]):
    input_ids = torch.arange(NUM_TOKENS)
    positions = torch.arange(NUM_TOKENS)
    intermediate = None
    for rank in range(len(partition)):
        inner, target = _v41_stage(monkeypatch, partition, rank)
        if len(partition) > 1:
            verify_supports_aux_hidden_states_over_pp(target, "dspark")
        with torch.no_grad():
            out = inner(input_ids, positions, intermediate)
        if rank == len(partition) - 1:
            return out
        # Mirrors PPHandler.relay_aux_hidden_states on middle stages.
        relayed = {k: intermediate[k] for k in aux_hidden_state_relay_keys(target)}
        intermediate = IntermediateTensors(out.tensors | relayed)
        assert {"hidden_states", "pre_mix"} <= set(intermediate.tensors)


@pytest.mark.parametrize(
    "partition",
    [
        [4, 3, 3],  # every DSpark target layer on the last stage
        [4, 4, 2],  # layer 7's entry stream is captured on the middle stage
        [3, 4, 3],  # the boundary sits exactly at a target layer (7)
        [4, 5, 1],  # one layer on the last stage: two relayed captures
        [7, 2, 1],  # captured on the first stage, relayed through the middle
    ],
)
def test_v41_dspark_aux_hidden_states_survive_pp_partition(monkeypatch, partition):
    """DSpark reads the streams entering V4.1 layers 37/38/39. Under PP the
    stage holding the layer before a target layer captures it, so wherever the
    partition cuts, the last stage must still see the same aux states, in
    order, as a single-stage run."""
    ref_hidden, ref_aux = _run_v41_pipeline(monkeypatch, [NUM_LAYERS])
    hidden, aux = _run_v41_pipeline(monkeypatch, partition)

    assert len(aux) == len(V41_AUX_LAYERS)
    torch.testing.assert_close(hidden, ref_hidden)
    for got, want in zip(aux, ref_aux):
        torch.testing.assert_close(got, want)
