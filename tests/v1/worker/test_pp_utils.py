# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry, and the PPHandler
sampled-token / draft-token relay under spec decode."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.distributed.parallel_state import GroupCoordinator, _split_tensor_dict
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu import model_runner, pp_utils
from vllm.v1.worker.gpu.pp_utils import PPHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PPHandler drives a side CUDA stream"
)


@pytest.fixture
def pp_worker(monkeypatch):
    """Run Worker.execute_model without initializing a device or process group."""
    worker = object.__new__(gpu_worker.Worker)
    worker._pp_step_id = 0
    worker._pp_send_work = []
    worker.use_v2_model_runner = False
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, distributed_executor_backend="mp"
        ),
    )
    worker.model_runner = Mock(execute_model=Mock(return_value=None))
    worker.annotate_profile = lambda _: nullcontext()
    group = Mock(is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: None)
    return worker, group


@pytest.mark.parametrize("sender_step", [0, 2, None])
def test_pp_step_skew_fails_before_forward(pp_worker, sender_step):
    """Even equal-shaped steps must fail closed on missing or skewed metadata."""
    worker, group = pp_worker
    tensors = {"hidden_states": torch.zeros(4, 2)}
    if sender_step is not None:
        tensors["__pp_step_id__"] = sender_step
    group.irecv_tensor_dict.return_value = (tensors, [], [])

    with pytest.raises(RuntimeError) as exc_info:
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))

    message = str(exc_info.value)
    assert "PP step 1" in message
    assert f"PP step {sender_step}" in message
    assert "num_scheduled_tokens=4" in message
    assert "skipped or double-sent" in message
    worker.model_runner.execute_model.assert_not_called()


def test_pp_step_metadata_is_removed_without_waiting(pp_worker, monkeypatch):
    """The integer survives transport but never enters the lazy tensor wrapper."""
    worker, group = pp_worker
    payload = {
        "__pp_step_id__": 1,
        "hidden_states": torch.empty(4, 2),
        "relay_attn_input.0": torch.empty(4, 2),
        model_runner._PP_ADAPTIVE_PLAN_KEY: torch.empty(4, 2),
    }
    metadata, tensors = _split_tensor_dict(payload)
    assert metadata[0] == ("__pp_step_id__", 1)
    assert len(tensors) == 3

    receiver = GroupCoordinator.__new__(GroupCoordinator)
    receiver.world_size = 2
    receiver.rank_in_group = 1
    receiver.ranks = [0, 1]
    receiver.use_cpu_custom_send_recv = False
    receiver.device_group = receiver.cpu_group = None
    receiver.recv_object = Mock(return_value=metadata)
    work = Mock()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "irecv", Mock(return_value=work))
    received, handles, postprocess = receiver.irecv_tensor_dict()
    assert received["__pp_step_id__"] == 1
    postprocess.append(Mock())
    group.irecv_tensor_dict.return_value = (received, handles, postprocess)

    worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))

    intermediate = worker.model_runner.execute_model.call_args.args[1]
    assert isinstance(intermediate, gpu_worker.AsyncIntermediateTensors)
    assert set(vars(intermediate)["tensors"]) == set(payload) - {"__pp_step_id__"}
    work.wait.assert_not_called()
    postprocess[0].assert_not_called()
    assert intermediate.tensors is received
    assert work.wait.call_count == len(tensors)
    postprocess[0].assert_called_once_with()


def test_pp_step_ids_ignore_zero_token_steps(pp_worker):
    """All stages count scheduled forwards, not rank-local work or idle steps."""
    worker, group = pp_worker
    group.is_first_rank = True
    group.is_last_rank = False
    output = IntermediateTensors({"hidden_states": torch.zeros(4, 2)})
    worker.model_runner.execute_model.side_effect = [output, None, output]
    metadata_work, tensor_work = Mock(), Mock()
    group.isend_tensor_dict.return_value = [metadata_work, tensor_work]

    for num_tokens in (4, 0, 4):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=num_tokens))

    sent = [call.args[0] for call in group.isend_tensor_dict.call_args_list]
    assert [payload["__pp_step_id__"] for payload in sent] == [1, 2]
    assert "__pp_step_id__" not in output.tensors
    group.irecv_tensor_dict.assert_not_called()
    metadata_work.wait.assert_not_called()
    tensor_work.wait.assert_called_once_with()

    group.is_first_rank = False
    group.is_last_rank = True
    worker._pp_step_id = 0
    worker.model_runner.execute_model.side_effect = [None, None, None]
    group.irecv_tensor_dict.side_effect = [
        ({"__pp_step_id__": step}, [], []) for step in (1, 2)
    ]
    for num_tokens in (4, 0, 4):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=num_tokens))
    assert group.irecv_tensor_dict.call_count == 2
    assert worker._pp_step_id == 2


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


# ---------------------------------------------------------------------------
# PPHandler relay: sender and receiver must stay in lockstep on every step
# ---------------------------------------------------------------------------


def make_handler(
    monkeypatch,
    *,
    is_last_rank: bool,
    num_speculative_steps: int,
    world_size: int = 2,
) -> PPHandler:
    """Build a real PPHandler with the PP group stubbed out."""
    pp_group = SimpleNamespace(
        is_last_rank=is_last_rank,
        last_rank=world_size - 1,
        world_size=world_size,
        make_sibling_device_group=lambda group_desc: object(),
    )
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: pp_group)
    return PPHandler(
        max_num_reqs=8,
        num_speculative_steps=num_speculative_steps,
        device=torch.device("cuda"),
    )


def record_broadcasts(monkeypatch, fill_value: int | None = None) -> list:
    """Capture every tensor handed to the collective, in call order.

    With `fill_value`, also fill the tensor so a receiver observes "data" as if
    the last rank had sent it."""
    calls: list[torch.Tensor] = []

    def fake_broadcast(tensor, src, group):
        calls.append(tensor)
        if fill_value is not None:
            tensor.fill_(fill_value)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    return calls


def make_input_batch(num_reqs: int = 3, *, needs_sample: bool = True):
    # compute_need_sampled_mask only reads the three numpy fields. With
    # needs_sample=False every request is a non-final prefill chunk, so no
    # sample is produced this step.
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, 4 if needs_sample else 4096, dtype=np.int32),
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device="cuda"),
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
    )


def send_step(handler, input_batch, *, width: int, with_draft: bool):
    num_reqs = input_batch.num_reqs
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device="cuda")
    counts = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    handler.broadcast(sampled, counts, counts, input_batch)
    if with_draft:
        # The runner relays its persistent [max_num_reqs, num_spec] table.
        draft_tokens = torch.zeros(
            8, handler.num_speculative_steps, dtype=torch.int64, device="cuda"
        )
        handler.broadcast_drafts(draft_tokens, input_batch)


def make_snapshot_runner(handler, input_batch, speculator):
    """A GPUModelRunner stub carrying only what ``sample_tokens`` reads on the
    last PP rank, so the confidence snapshot runs in its real call site."""
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    num_reqs = input_batch.num_reqs
    sampled = torch.zeros(num_reqs, 1, dtype=torch.int64, device="cuda")
    num_sampled = torch.ones(num_reqs, dtype=torch.int32, device="cuda")
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cuda")
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.pp_handler = handler
    runner.speculator = speculator
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(8, device="cuda")),
            seeds=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int64, device="cuda")),
        )
    )
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.zeros(8, 3, dtype=torch.int64, device="cuda"),
        last_sampled_tokens=torch.zeros(8, dtype=torch.int64, device="cuda"),
        next_prefill_tokens=torch.zeros(1, 8, dtype=torch.int64, device="cuda"),
        all_token_ids=SimpleNamespace(
            gpu=torch.zeros(8, dtype=torch.int64, device="cuda")
        ),
        num_computed_tokens=SimpleNamespace(
            gpu=torch.zeros(8, dtype=torch.int32, device="cuda")
        ),
        prompt_len=SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {}
    )
    runner.main_stream = torch.cuda.current_stream()
    runner.output_copy_stream = torch.cuda.Stream()
    runner.check_ep_fault = False
    runner.postprocess_sampled = Mock()
    runner.adaptive_verification = SimpleNamespace(record_confidences=Mock())
    runner.num_speculative_steps = 3
    runner.draft_tokens_handler = Mock()
    runner.kv_connector = Mock()
    runner.eplb = Mock()
    runner._draft_workspace_lane = 0
    runner.sample = Mock(return_value=(sampler_output, num_sampled, num_rejected))

    def make_execute_model_state():
        return SimpleNamespace(
            input_batch=input_batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=torch.zeros(1, 1, device="cuda"),
            aux_hidden_states=None,
            dp_sync=None,
            finished_req_ids=set(),
            ec_connector_output=None,
            routed_experts=None,
            cudagraph_stats=None,
        )

    runner.make_execute_model_state = make_execute_model_state
    return runner


@requires_cuda
@pytest.mark.parametrize("width,num_spec", [(1, 1), (1, 3), (2, 3)])
def test_broadcast_pads_sampled_tokens_to_max_sample_len(monkeypatch, width, num_spec):
    """The sampler emits width 1 on steps with no draft tokens (prefill, first
    decode) and num_spec+1 only after rejection sampling. The receiver always
    posts a max_sample_len buffer, so an unpadded send is a count mismatch."""
    handler = make_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=num_spec
    )
    calls = record_broadcasts(monkeypatch)
    input_batch = make_input_batch()

    send_step(handler, input_batch, width=width, with_draft=False)

    sent_sampled = calls[0]
    assert sent_sampled.shape == (input_batch.num_reqs, handler.max_sample_len)
    # The real columns come first; the padding columns are ignored by
    # post_update, which advances each request by its own num_sampled count.
    assert (sent_sampled[:, :width] == 0).all()


@requires_cuda
def test_send_and_recv_op_counts_match_with_speculator(monkeypatch):
    """With a speculator the step is three broadcasts: sampled, combined, draft,
    matched strictly by order on the communicator."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=3)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=True)
    assert len(calls) == 3
    assert calls[0].shape == (3, sender.max_sample_len)
    assert calls[2].shape == (3, sender.num_speculative_steps)

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=3)
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 3
    assert calls[0].shape == (3, sender.max_sample_len)
    assert calls[2].shape == (3, sender.num_speculative_steps)


@requires_cuda
def test_send_and_recv_op_counts_match_without_speculator(monkeypatch):
    """Without spec decode the step is two broadcasts on both sides."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=0)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=False)
    assert len(calls) == 2

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=0)
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 2
    assert receiver.queue[-1].draft_tokens is None


@requires_cuda
def test_both_ranks_skip_when_no_request_needs_sampling(monkeypatch):
    """The skip gate must be symmetric, or the ranks desynchronize."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=3)
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(needs_sample=False), width=1, with_draft=True)
    assert calls == []

    receiver = make_handler(monkeypatch, is_last_rank=False, num_speculative_steps=3)
    calls.clear()
    assert not receiver.receive(make_input_batch(needs_sample=False))
    assert calls == []


@requires_cuda
def test_relayed_draft_tokens_are_scattered_on_consume(monkeypatch):
    """Non-last ranks verify against req_states.draft_tokens, which only the
    last rank can compute. The relayed drafts must land in that table when the
    deferred entry is consumed pp_size steps later, or the earlier stages verify
    against a zero-initialised table (near-zero acceptance, corrupt output)."""
    receiver = make_handler(
        monkeypatch, is_last_rank=False, num_speculative_steps=3, world_size=2
    )
    record_broadcasts(monkeypatch, fill_value=7)
    input_batch = make_input_batch(num_reqs=2)
    assert receiver.receive(input_batch)

    draft_tokens = torch.zeros(8, 3, dtype=torch.int64, device="cuda")
    # The queue is pre-seeded with pp_size placeholders: the first consume is
    # a no-op, the second returns the entry pushed above.
    assert receiver.get_prev_sampled_outputs(draft_tokens) is None
    outputs = receiver.get_prev_sampled_outputs(draft_tokens)
    torch.cuda.synchronize()

    assert outputs is not None
    assert (draft_tokens[:2] == 7).all()
    assert (draft_tokens[2:] == 0).all()
    assert (outputs["sampled_tokens"] == 7).all()


def test_disabled_handler_skips_broadcast_and_receive(monkeypatch):
    """While disabled (warmup), neither side enqueues a broadcast op."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    handler = object.__new__(pp_utils.PPHandler)
    handler.set_disabled(True)

    handler.is_last_rank = False
    assert handler.receive(Mock()) is False

    handler.is_last_rank = True
    assert handler.broadcast(Mock(), Mock(), Mock(), Mock()) is None
    handler.max_sample_len = 6
    assert handler.broadcast_drafts(Mock(), Mock()) is None

    assert sent == []

    handler.set_disabled(False)
    assert handler.disabled is False


def test_alloc_combined_keeps_unbind_views_16_byte_aligned():
    """Triton specializes on pointer alignment: an unaligned `num_rejected`
    would compile a second `_post_update_kernel` variant at serving time,
    where the in-flight broadcast NCCL kernel can block the module load."""
    for num_reqs in range(1, 9):
        combined = pp_utils._alloc_combined(num_reqs, torch.device("cpu"))
        num_sampled, num_rejected = combined.unbind(dim=0)
        assert num_sampled.data_ptr() % 16 == 0
        assert num_rejected.data_ptr() % 16 == 0
        assert combined.shape[1] >= num_reqs


def test_warmup_pp_decode_update_matches_serving_specialization(monkeypatch):
    """The warmup launch must hit the same triton specialization as serving.

    A mismatch means the first real ``update_pp_decode_requests`` recompiles
    mid-serving, where the in-flight broadcast NCCL kernel blocks the CUDA
    module load and deadlocks the pipeline.
    """
    calls = []
    monkeypatch.setattr(model_runner, "post_update", lambda *args: calls.append(args))

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pp_handler = Mock(max_sample_len=3)
    runner.req_states = Mock()

    runner.warmup_pp_decode_update()

    assert len(calls) == 1
    args = calls[0]
    idx_mapping, _, _, output_bin_counts = args[:4]
    sampled_tokens, num_sampled, num_rejected, query_start_loc = args[4:8]
    broadcast_drafts, draft_tokens_out = args[10:12]
    assert idx_mapping.tolist() == [-1] and idx_mapping.dtype == torch.int64
    assert output_bin_counts is None
    assert query_start_loc is None
    assert sampled_tokens.shape == (1, 3) and sampled_tokens.dtype == torch.int64
    assert num_sampled.dtype == torch.int32
    assert num_rejected.dtype == torch.int32
    # Spec-enabled PP handlers receive drafts over the broadcast; the warmup
    # must compile that specialization (non-None draft pointers) too.
    assert broadcast_drafts.shape == (1, 2) and broadcast_drafts.dtype == torch.int64
    assert draft_tokens_out is runner.req_states.draft_tokens


@requires_cuda
def test_draft_relay_snapshots_before_the_side_stream_stalls(monkeypatch):
    """The sampled-token sends ahead of the draft relay can hold the side stream
    back on peer progress while the next step rewrites the persistent draft
    table on the main stream. The relay must still carry this step's drafts."""
    sender = make_handler(monkeypatch, is_last_rank=True, num_speculative_steps=3)
    sent: list[torch.Tensor] = []

    def stalled_broadcast(tensor, src, group):
        torch.cuda._sleep(100_000_000)
        sent.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, "broadcast", stalled_broadcast)
    input_batch = make_input_batch()
    sampled = torch.zeros(3, 1, dtype=torch.int64, device="cuda")
    counts = torch.zeros(3, dtype=torch.int32, device="cuda")
    persistent = torch.full((8, 3), 7, dtype=torch.int64, device="cuda")

    sender.broadcast(sampled, counts, counts, input_batch)
    sender.broadcast_drafts(persistent, input_batch)
    persistent.fill_(-1)
    torch.accelerator.synchronize()

    assert len(sent) == 3
    assert (sent[2] == 7).all()


# ---------------------------------------------------------------------------
# Adaptive-verification confidence payload: the ring and its filtering
# ---------------------------------------------------------------------------


def make_confidence_ring(depth: int, num_spec: int = 2, max_num_reqs: int = 4):
    """A CPU ring: the slot bookkeeping is device-independent."""
    return pp_utils.AdaptiveConfidenceRing(
        depth, max_num_reqs, num_spec, torch.device("cpu")
    )


def confidence_block(num_rows: int, num_spec: int, value: float) -> torch.Tensor:
    return torch.full((num_rows, num_spec), value, dtype=torch.float32)


def push_block(ring, value, idx_mapping_np, req_idx_gen_np, mask=None):
    num_rows = len(idx_mapping_np)
    if mask is None:
        mask = np.ones(num_rows, dtype=bool)
    return ring.push(
        confidence_block(num_rows, ring._slots[0].host.shape[1], value),
        idx_mapping_np,
        req_idx_gen_np[idx_mapping_np],
        mask,
    )


@pytest.mark.parametrize("depth", [1, 2, 6])
def test_confidence_ring_keeps_every_in_flight_step_alive(depth):
    """Each in-flight step owns its pinned block: writing step T+depth must not
    disturb the payloads of the `depth` steps still in flight."""
    ring = make_confidence_ring(depth)
    req_idx_gen_np = np.zeros(4, dtype=np.int32)
    idx_mapping_np = np.arange(4, dtype=np.intp)

    slots = [
        push_block(ring, float(step), idx_mapping_np, req_idx_gen_np)
        for step in range(depth)
    ]
    assert len({id(slot) for slot in slots}) == depth

    # One more push reuses the oldest slot only.
    push_block(ring, -1.0, idx_mapping_np, req_idx_gen_np)
    for step, slot in enumerate(slots):
        payload = slot.collect(req_idx_gen_np)
        assert payload is not None
        # Only the oldest slot was reused by the extra push.
        expected = -1.0 if step == 0 else float(step)
        assert (payload.confidences_np == expected).all()
        assert (payload.confidences == expected).all()


@pytest.mark.parametrize("depth", [1, 2, 6])
def test_confidence_ring_reads_the_payload_pushed_depth_steps_earlier(depth):
    """The consumer of step S must see the block proposed at S-depth, which is
    the whole alignment argument for carrying confidences on the relay."""
    ring = make_confidence_ring(depth)
    req_idx_gen_np = np.zeros(4, dtype=np.int32)
    idx_mapping_np = np.arange(2, dtype=np.intp)
    queue: deque = deque([None] * depth)

    for step in range(3 * depth + 2):
        slot = queue.popleft()
        queue.append(None)
        payload = ring.collect(slot, req_idx_gen_np)
        expected = step - depth
        if expected < 0:
            assert payload is None
        else:
            assert payload is not None
            assert (payload.confidences_np == float(expected)).all()
        queue[-1] = push_block(ring, float(step), idx_mapping_np, req_idx_gen_np)


def test_confidence_ring_drops_rows_freed_or_unsampled_since_push():
    """A slot freed and reused must not receive the old request's block, and a
    row that produces no sample must not overwrite its slot either."""
    ring = make_confidence_ring(2)
    req_idx_gen_np = np.zeros(4, dtype=np.int32)
    idx_mapping_np = np.array([3, 1, 2], dtype=np.intp)
    gen_np = req_idx_gen_np[idx_mapping_np]
    slot = ring.push(
        confidence_block(3, 2, 0.5),
        idx_mapping_np,
        gen_np,
        np.array([True, False, True]),
    )

    # Slot 1 finishes and is reused between push and consume.
    req_idx_gen_np[1] += 1
    payload = slot.collect(req_idx_gen_np)

    assert payload is not None
    # Row 1 needs no sample, so it is dropped before its generation matters.
    assert payload.idx_mapping_np.tolist() == [3, 2]
    assert payload.confidences.shape == (2, 2)
    assert payload.confidences_np.shape == (2, 2)

    # A freed row that also needed a sample is dropped by the generation check.
    gen_np = req_idx_gen_np[idx_mapping_np]
    slot = ring.push(
        confidence_block(3, 2, 0.5), idx_mapping_np, gen_np, np.ones(3, dtype=bool)
    )
    req_idx_gen_np[3] += 1
    payload = slot.collect(req_idx_gen_np)
    assert payload is not None
    assert payload.idx_mapping_np.tolist() == [1, 2]


def test_confidence_ring_collects_nothing_when_every_row_is_gone():
    ring = make_confidence_ring(2)
    req_idx_gen_np = np.zeros(4, dtype=np.int32)
    idx_mapping_np = np.array([0, 1], dtype=np.intp)
    slot = ring.push(
        confidence_block(2, 2, 0.5),
        idx_mapping_np,
        req_idx_gen_np[idx_mapping_np],
        np.zeros(2, dtype=bool),
    )
    assert slot.collect(req_idx_gen_np) is None


# ---------------------------------------------------------------------------
# Adaptive-verification confidence payload: on the wire
# ---------------------------------------------------------------------------


def make_adaptive_handler(
    monkeypatch,
    *,
    is_last_rank: bool,
    num_speculative_steps: int = 3,
    world_size: int = 2,
) -> PPHandler:
    handler = make_handler(
        monkeypatch,
        is_last_rank=is_last_rank,
        num_speculative_steps=num_speculative_steps,
        world_size=world_size,
    )
    handler.set_adaptive_confidences(True)
    return handler


def confidence_payload(handler, input_batch, value: float = 0.5) -> torch.Tensor:
    return torch.full(
        (input_batch.num_reqs, handler.num_speculative_steps),
        value,
        dtype=torch.float32,
        device="cuda",
    )


@requires_cuda
def test_adaptive_confidence_payload_is_the_fourth_broadcast(monkeypatch):
    """The confidences ride the same FIFO as the block they score, so the ops
    must line up strictly by order on both sides."""
    input_batch = make_input_batch(num_reqs=2)
    sender = make_adaptive_handler(monkeypatch, is_last_rank=True)
    calls = record_broadcasts(monkeypatch, fill_value=7)
    sampled = torch.zeros(2, 1, dtype=torch.int64, device="cuda")
    counts = torch.zeros(2, dtype=torch.int32, device="cuda")
    sender.broadcast(sampled, counts, counts, input_batch)
    sender.broadcast_drafts(
        torch.zeros(8, 3, dtype=torch.int64, device="cuda"),
        input_batch,
        confidence_payload(sender, input_batch),
    )
    assert len(calls) == 4
    assert calls[3].shape == (2, sender.num_speculative_steps)
    assert calls[3].dtype == torch.float32

    receiver = make_adaptive_handler(monkeypatch, is_last_rank=False)
    calls.clear()
    assert receiver.receive(input_batch)
    assert len(calls) == 4
    assert calls[3].shape == (2, sender.num_speculative_steps)

    # The first consume is the pre-seeded placeholder; the second one is this
    # step's payload.
    assert receiver.get_prev_sampled_outputs() is None
    outputs = receiver.get_prev_sampled_outputs()
    assert outputs is not None
    confidences = outputs.pop("confidences")
    assert confidences is not None
    assert confidences.idx_mapping_np.tolist() == input_batch.idx_mapping_np.tolist()
    assert (confidences.confidences_np == 7.0).all()
    assert (confidences.confidences == 7.0).all()


@requires_cuda
def test_adaptive_confidence_payload_follows_the_skip_gate(monkeypatch):
    """No sample this step means no payload either; a rank that posted the
    extra broadcast would desynchronize the pp_broadcast FIFO."""
    input_batch = make_input_batch(needs_sample=False)

    sender = make_adaptive_handler(monkeypatch, is_last_rank=True)
    calls = record_broadcasts(monkeypatch, fill_value=7)
    sender.broadcast_drafts(
        torch.zeros(8, 3, dtype=torch.int64, device="cuda"),
        input_batch,
        confidence_payload(sender, input_batch),
    )
    assert calls == []
    assert sender.pop_self_confidences() is None

    receiver = make_adaptive_handler(monkeypatch, is_last_rank=False)
    calls.clear()
    assert not receiver.receive(input_batch)
    assert calls == []


@requires_cuda
def test_last_rank_self_ring_returns_the_payload_pushed_pp_size_steps_ago(
    monkeypatch,
):
    """The last rank has no deferred queue, so it must replay its own payload
    through a ring of the same depth rather than read the fresh one."""
    pp_size = 3
    handler = make_adaptive_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=2, world_size=pp_size
    )
    record_broadcasts(monkeypatch, fill_value=None)
    input_batch = make_input_batch(num_reqs=2)

    for step in range(2 * pp_size + 1):
        payload = handler.pop_self_confidences()
        if step < pp_size:
            assert payload is None
        else:
            assert payload is not None
            assert (payload.confidences_np == float(step - pp_size)).all()
        handler.broadcast_drafts(
            torch.zeros(8, 2, dtype=torch.int64, device="cuda"),
            input_batch,
            confidence_payload(handler, input_batch, float(step)),
        )


def stall_fourth_broadcast(monkeypatch) -> list[torch.Tensor]:
    """Clone each step's confidence op after a side-stream stall.

    The confidence payload is the fourth broadcast of a step, so stalling only
    that op leaves the sampled-token sends ahead of it unaffected."""
    sent: list[torch.Tensor] = []
    ops: list[torch.Tensor] = []

    def stalling_broadcast(tensor, src, group):
        ops.append(tensor)
        if len(ops) % 4 == 0:
            assert tensor.dtype == torch.float32
            torch.cuda._sleep(100_000_000)
            sent.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, "broadcast", stalling_broadcast)
    return sent


@requires_cuda
def test_confidence_relay_snapshots_before_the_side_stream_stalls(monkeypatch):
    """The speculator reuses its confidence buffer next step, while this step's
    confidence broadcast can still be waiting on the side stream. The runner
    must relay a snapshot, not the live buffer, or earlier stages price the
    wrong generation. This drives the real ``sample_tokens`` snapshot path, not
    a test-side clone, and checks both the broadcast and the last rank's
    self-ring payload."""
    num_reqs, num_spec = 3, 3
    handler = make_adaptive_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=num_spec, world_size=2
    )
    input_batch = make_input_batch(num_reqs=num_reqs)
    input_batch.req_ids = [f"req-{i}" for i in range(num_reqs)]
    input_batch.query_start_loc = torch.arange(
        num_reqs + 1, dtype=torch.int32, device="cuda"
    )

    probs = torch.full((8, num_spec), -1.0, dtype=torch.float32, device="cuda")
    draft_tokens = torch.zeros(8, num_spec, dtype=torch.int64, device="cuda")
    expected: list[torch.Tensor] = []

    def propose(*args, **kwargs):
        values = torch.arange(
            num_reqs * num_spec, dtype=torch.float32, device="cuda"
        ).view(num_reqs, num_spec) + 100 * len(expected)
        probs[:num_reqs] = values
        expected.append(values)
        return draft_tokens[:num_reqs]

    speculator = SimpleNamespace(
        supports_mm_inputs=False,
        draft_token_confidence_probs=probs,
        propose=propose,
    )
    runner = make_snapshot_runner(handler, input_batch, speculator)

    sent = stall_fourth_broadcast(monkeypatch)

    runner.execute_model_state = runner.make_execute_model_state()
    async_outputs = [runner.sample_tokens(None)]
    # The speculator reuses the live buffer before the side-stream send and its
    # host copy can finish.
    probs.fill_(-1.0)
    assert handler.pop_self_confidences() is None

    runner.execute_model_state = runner.make_execute_model_state()
    async_outputs.append(runner.sample_tokens(None))
    payload = handler.pop_self_confidences()
    torch.accelerator.synchronize()
    assert all(output is not None for output in async_outputs)

    assert len(sent) == 2
    for got, want in zip(sent, expected):
        assert got.shape == (num_reqs, num_spec)
        torch.testing.assert_close(got, want)

    assert payload is not None
    assert payload.idx_mapping_np.tolist() == input_batch.idx_mapping_np.tolist()
    assert payload.confidences.shape == (num_reqs, num_spec)
    torch.testing.assert_close(payload.confidences, expected[0])
    torch.testing.assert_close(
        torch.from_numpy(payload.confidences_np), expected[0].cpu()
    )


@requires_cuda
def test_live_confidence_buffer_would_alias_the_next_step(monkeypatch):
    """Negative control for the snapshot test: relaying the live buffer is not
    safe, because the side-stream send and host copy observe next step's rewrite.
    Kept separate so the snapshot test can assert the positive behaviour."""
    handler = make_adaptive_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=3, world_size=2
    )
    input_batch = make_input_batch(num_reqs=3)
    sent = stall_fourth_broadcast(monkeypatch)
    live = torch.arange(9, dtype=torch.float32, device="cuda").view(3, 3)

    send_step(handler, input_batch, width=1, with_draft=False)
    handler.broadcast_drafts(
        torch.zeros(8, 3, dtype=torch.int64, device="cuda"), input_batch, live
    )
    assert handler.pop_self_confidences() is None
    live.fill_(-1.0)
    torch.accelerator.synchronize()

    assert len(sent) == 1
    assert (sent[0] == -1.0).all()
    payload = handler.pop_self_confidences()
    assert payload is not None
    assert (payload.confidences == -1.0).all()
    assert (payload.confidences_np == -1.0).all()


@requires_cuda
def test_adaptive_confidences_fail_closed_without_a_payload(monkeypatch):
    """A draft block without its confidences would let the ranks price
    different generations, so the relay refuses to post it."""
    handler = make_adaptive_handler(monkeypatch, is_last_rank=True)
    record_broadcasts(monkeypatch)
    with pytest.raises(AssertionError):
        handler.broadcast_drafts(
            torch.zeros(8, 3, dtype=torch.int64, device="cuda"), make_input_batch()
        )


def test_pp_adaptive_plan_debug_compare_covers_per_request_boundaries():
    """A total-token check alone would miss stages that trim the same total into
    different per-request blocks, so the debug plan carries the request order,
    the query boundaries and the logits boundaries."""
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 4
    runner._pp_adaptive_debug = True

    def batch(query_start_loc, cu_num_logits, idx_mapping, num_tokens=5):
        return SimpleNamespace(
            num_reqs=len(query_start_loc) - 1,
            num_tokens=num_tokens,
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
            cu_num_logits=torch.tensor(cu_num_logits, dtype=torch.int32),
            idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64),
        )

    batch_state = batch([0, 3, 5], [0, 2, 3], [1, 0])
    plan = runner._pp_adaptive_plan(batch_state)
    assert plan[0, :2].tolist() == [2, 5]
    assert plan[3, :2].tolist() == [1, 0]

    # Same total, different per-request split.
    runner._check_pp_adaptive_plan(plan, batch_state)
    with pytest.raises(RuntimeError):
        runner._check_pp_adaptive_plan(plan, batch([0, 2, 5], [0, 2, 3], [1, 0]))

    # A stage that forgot to send the plan must not pass silently.
    with pytest.raises(RuntimeError):
        runner._check_pp_adaptive_plan(None, batch_state)

    # The flag is required; without it the payload is neither sent nor checked.
    runner._pp_adaptive_debug = False
    runner._check_pp_adaptive_plan(plan, batch([0, 2, 5], [0, 2, 3], [1, 0]))
    runner._check_pp_adaptive_plan(None, batch_state)


def test_pp_adaptive_flag_broadcast_uses_the_last_local_rank(monkeypatch):
    """`GroupCoordinator.broadcast_object` takes a local group rank, while
    `PPHandler` needs the global last rank for its NCCL broadcast. On a
    non-contiguous PP group (TP4 x PP6) the two differ, and passing the global
    rank here trips the coordinator's `src < world_size` assert."""

    class FakePPGroup:
        world_size = 6
        ranks = [2, 6, 10, 14, 18, 22]
        last_rank = 22

        def broadcast_object(self, obj, src):
            assert 0 <= src < self.world_size, f"invalid local src rank ({src})"
            self.src_global_rank = self.ranks[src]
            # Only the last stage knows the effective value.
            return obj if src != self.world_size - 1 else False

    pp_group = FakePPGroup()
    monkeypatch.setattr(model_runner, "get_pp_group", lambda: pp_group)

    # A non-last stage sees the last stage's answer.
    assert model_runner.broadcast_pp_adaptive_verification_flag(True) is False
    assert pp_group.src_global_rank == 22
    assert model_runner.broadcast_pp_adaptive_verification_flag(False) is False
