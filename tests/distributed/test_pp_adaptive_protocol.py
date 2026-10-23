# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The adaptive PP relay over a real gloo transport.

Two spawned ranks run the production ``PPHandler`` send/receive path, the
production confidence ring, and the production confidence install and draft
budget. The payload crosses a real gloo broadcast, so a sender and receiver
that disagree about the four-op FIFO, the skip gate or the generation a block
belongs to hang or fail instead of agreeing with themselves in one process.

CPU only: the CUDA stream/event/record_stream calls are stubbed inside the
child processes. This proves the host-side protocol and generation alignment,
not NCCL stream safety or device-side masking.
"""

import multiprocessing as mp
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.utils.network_utils import get_file_store_init_method

_PP_SIZE = 2
_NUM_SPEC = 2
_MAX_REQS = 4
_REQ_IDS = tuple(f"r{slot}" for slot in range(_MAX_REQS))
_JOIN_TIMEOUT_S = 120
_FAIL = "adaptive pp relay"


@dataclass(frozen=True)
class _Step:
    """One scheduler step: which request slots it runs and how many drafts it
    schedules for each. A prefill step produces no sample and therefore posts
    no relay payload. `free` slots are released after the step's consume."""

    slots: tuple[int, ...]
    capacities: tuple[int, ...] = ()
    prefill: bool = False
    free: tuple[int, ...] = ()


_SCRIPT = (
    _Step((0, 1, 2), (2, 2, 2)),
    # Reordered batch and a request whose draft capacity shrank.
    _Step((2, 0, 1), (1, 2, 2)),
    # No sample this step; slot 0 is freed and reused at the same point.
    _Step((0, 1, 2), prefill=True, free=(0,)),
    _Step((1, 2, 0), (2, 2, 2)),
    _Step((0, 2), (2, 1), free=(1,)),
    _Step((2, 1, 0), (1, 1, 1)),
    _Step((0, 1, 2), (2, 2, 2)),
    # Drain the last two in-flight generations.
    _Step(()),
    _Step(()),
)

_MUTATIONS = ("drop_confidences", "no_generation_filter")


@dataclass(frozen=True)
class _Proposal:
    slots: tuple[int, ...]
    gens: tuple[int, ...]
    confidences: "object"
    drafts: "object"
    sampled: "object"
    num_sampled: "object"
    num_rejected: "object"


def _check(condition: bool, message: str) -> None:
    assert condition, f"{_FAIL}: {message}"


def _install_cpu_shims() -> None:
    """Confine the CUDA stream/event surface to these child processes."""
    import torch

    class _Event:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class _Stream:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def wait_stream(self, other) -> None:
            pass

        def wait_event(self, event) -> None:
            pass

        def record_event(self) -> _Event:
            return _Event()

        def synchronize(self) -> None:
            pass

    class _StreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *exc_info):
            return False

    torch.Tensor.record_stream = lambda self, stream: None
    torch.cuda.current_stream = lambda device=None: _Stream()
    torch.cuda.Stream = _Stream
    torch.cuda.Event = _Event
    torch.cuda.stream = lambda stream: _StreamContext(stream)


def _cpu_copy(x, out=None, device=None):
    """The relay's index copy without pinned memory or a device context."""
    import numpy as np
    import torch

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if out is None:
        out = torch.empty_like(x)
    return out.copy_(x)


def _make_manager(rank: int):
    import numpy as np
    import torch

    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
        build_cost_tables_from_curves,
    )

    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = _NUM_SPEC
    manager.num_bonus_tokens = 1
    manager.pp_world_size = _PP_SIZE
    manager.pp_rank = rank
    manager._confidence_cpu = np.ones((_MAX_REQS, _NUM_SPEC), dtype=np.float32)
    manager._confidence_probs = torch.ones((_MAX_REQS, _NUM_SPEC), dtype=torch.float32)
    manager._batch_draft_capacity = torch.zeros(_MAX_REQS, dtype=torch.int32)
    manager._num_non_draft_tokens = torch.zeros(_MAX_REQS, dtype=torch.int32)
    manager._cu_num_logits = torch.zeros(_MAX_REQS + 1, dtype=torch.int32)
    manager.query_start_loc = torch.zeros(_MAX_REQS + 1, dtype=torch.int32)
    manager._stale_confidences = []
    manager._stale_idx = 0
    manager._max_total_logits = 1 << 30
    manager._batch_budget = None
    manager.cost_tables = build_cost_tables_from_curves(
        draft_curve=[(1, 0.2), (4, 0.8)],
        verify_curve=[(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (8, 2.0)],
        max_num_reqs=_MAX_REQS,
        max_batch_tokens=32,
    )
    manager.req_states = SimpleNamespace(
        req_id_to_index={req_id: slot for slot, req_id in enumerate(_REQ_IDS)},
        num_computed_tokens_np=np.full(_MAX_REQS, 100, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.full(_MAX_REQS, 100, dtype=np.int32)),
        max_num_reqs=_MAX_REQS,
        max_num_batched_tokens=32,
        num_speculative_steps=_NUM_SPEC,
    )
    return manager


def _make_handler(rank: int, pp_utils):
    import torch
    import torch.distributed as dist

    pp_group = SimpleNamespace(
        is_last_rank=rank == _PP_SIZE - 1,
        last_rank=_PP_SIZE - 1,
        world_size=_PP_SIZE,
        make_sibling_device_group=lambda group_desc: dist.group.WORLD,
    )
    pp_utils.get_pp_group = lambda: pp_group
    handler = pp_utils.PPHandler(_MAX_REQS, _NUM_SPEC, torch.device("cpu"))
    handler.set_adaptive_confidences(True)
    return handler


def _make_batch(step: _Step):
    import numpy as np
    import torch

    num_reqs = len(step.slots)
    if step.prefill:
        computed = np.zeros(num_reqs, dtype=np.int32)
        prefill_len = np.full(num_reqs, 4096, dtype=np.int32)
        scheduled = np.full(num_reqs, 4, dtype=np.int32)
    else:
        computed = np.full(num_reqs, 100, dtype=np.int32)
        prefill_len = np.full(num_reqs, 100, dtype=np.int32)
        scheduled = np.array(step.capacities, dtype=np.int32) + 1
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=computed,
        prefill_len_np=prefill_len,
        num_scheduled_tokens=scheduled,
        idx_mapping=torch.tensor(step.slots, dtype=torch.int64),
        idx_mapping_np=np.array(step.slots, dtype=np.intp),
    )


def _confidence_block(slots: tuple[int, ...], step: int):
    """Per-slot survival probabilities, all strictly inside [0, 1]."""
    import numpy as np

    rows = []
    for slot in slots:
        base = 0.9 - 0.1 * slot - 0.01 * (step % 3)
        rows.append([base - 0.08 * position for position in range(_NUM_SPEC)])
    return np.array(rows, dtype=np.float32)


def _proposal(step_index: int, slot_ids: tuple[int, ...], test_gen):
    import numpy as np
    import torch

    width = 1 + step_index % (_NUM_SPEC + 1)
    sampled = np.array(
        [
            [1000 + 10 * step_index + slot + column for column in range(width)]
            for slot in slot_ids
        ],
        dtype=np.int64,
    )
    drafts = np.array(
        [
            [
                2000 + 10 * step_index + slot * 3 + position
                for position in range(_NUM_SPEC)
            ]
            for slot in slot_ids
        ],
        dtype=np.int64,
    )
    return _Proposal(
        slots=slot_ids,
        gens=tuple(int(test_gen[slot]) for slot in slot_ids),
        confidences=_confidence_block(slot_ids, step_index),
        drafts=drafts,
        sampled=sampled,
        num_sampled=torch.ones(len(slot_ids), dtype=torch.int32),
        num_rejected=torch.zeros(len(slot_ids), dtype=torch.int32),
    )


def _consume(handler, manager, pending, test_gen, local_draft_table):
    import numpy as np
    import torch

    outputs = handler.get_prev_sampled_outputs(local_draft_table)
    confidences = outputs.pop("confidences") if outputs is not None else None
    if confidences is None:
        confidences = handler.pop_self_confidences()

    proposal = pending.popleft()
    pending.append(None)
    if proposal is None:
        _check(outputs is None, "postprocess payload without a proposal")
        _check(confidences is None, "confidence payload without a proposal")
        return

    live = np.array(
        [test_gen[slot] == gen for slot, gen in zip(proposal.slots, proposal.gens)],
        dtype=bool,
    )
    expected_slots = np.array(proposal.slots, dtype=np.intp)[live]
    if expected_slots.size == 0:
        _check(outputs is None, "a fully freed block still produced a sample payload")
        _check(confidences is None, "a fully freed block still produced a payload")
        return

    _check(confidences is not None, "missing generation-aligned confidence payload")
    _check(
        confidences.idx_mapping_np.tolist() == expected_slots.tolist(),
        "payload rows do not match the still-live proposal slots",
    )
    np.testing.assert_allclose(confidences.confidences_np, proposal.confidences[live])
    manager.install_confidences(confidences)

    for row, slot in enumerate(proposal.slots):
        if live[row]:
            np.testing.assert_allclose(
                manager._confidence_cpu[slot], proposal.confidences[row]
            )
        else:
            _check(
                np.allclose(manager._confidence_cpu[slot], 1.0),
                "a freed slot inherited an in-flight confidence block",
            )

    if outputs is None:
        return
    width = proposal.sampled.shape[1]
    np.testing.assert_array_equal(
        outputs["sampled_tokens"][:, :width].numpy(), proposal.sampled
    )
    _check(
        torch.equal(
            outputs["num_sampled"][: len(proposal.slots)], proposal.num_sampled
        ),
        "num_sampled did not survive the relay",
    )
    _check(
        torch.equal(
            outputs["num_rejected"][: len(proposal.slots)], proposal.num_rejected
        ),
        "num_rejected did not survive the relay",
    )
    for row, slot in enumerate(proposal.slots):
        if live[row]:
            np.testing.assert_array_equal(
                local_draft_table[slot].numpy(), proposal.drafts[row]
            )
        else:
            _check(
                torch.equal(
                    local_draft_table[slot],
                    torch.full_like(local_draft_table[slot], -1),
                ),
                "a freed slot inherited an in-flight draft block",
            )


def _drive(pp_utils, adaptive_module, rank: int, case: str) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    if case == "drop_confidences":
        original_push = pp_utils.AdaptiveConfidenceRing.push

        def drop_push(ring, *args, **kwargs):
            original_push(ring, *args, **kwargs)
            return None

        pp_utils.AdaptiveConfidenceRing.push = drop_push
    elif case == "no_generation_filter":
        pp_utils.PPHandler.on_req_idx_freed = lambda self, req_idx: None

    # The eager ranking kernel is the body the compiled wrapper runs, without a
    # CPU torch.compile inside the protocol test.
    adaptive_module._assign_draft_token_budget_compiled = (
        adaptive_module._assign_draft_token_budget
    )

    handler = _make_handler(rank, pp_utils)
    manager = _make_manager(rank)
    is_last_rank = rank == _PP_SIZE - 1
    local_draft_table = torch.full((_MAX_REQS, _NUM_SPEC), -1, dtype=torch.int64)
    test_gen = np.zeros(_MAX_REQS, dtype=np.int32)
    pending = deque([None] * _PP_SIZE)
    broadcast_calls: list = []
    real_broadcast = torch.distributed.broadcast

    def counted_broadcast(tensor, src, group, **kwargs):
        broadcast_calls.append((tuple(tensor.shape), tensor.dtype))
        return real_broadcast(tensor, src=src, group=group, **kwargs)

    torch.distributed.broadcast = counted_broadcast

    for step_index, step in enumerate(_SCRIPT):
        batch = _make_batch(step)
        _consume(handler, manager, pending, test_gen, local_draft_table)

        # Production frees and re-adds before this step's batch is planned, but
        # after the deferred payload was installed.
        for slot in step.free:
            test_gen[slot] += 1
            handler.on_req_idx_freed(slot)
            manager.add_request(slot)
            local_draft_table[slot].fill_(-1)

        if step.prefill or not step.slots:
            if step.slots:
                before = len(broadcast_calls)
                if is_last_rank:
                    handler.broadcast(
                        torch.zeros(len(step.slots), 1, dtype=torch.int64),
                        torch.zeros(len(step.slots), dtype=torch.int32),
                        torch.zeros(len(step.slots), dtype=torch.int32),
                        batch,
                    )
                    handler.broadcast_drafts(local_draft_table, batch)
                else:
                    _check(not handler.receive(batch), "prefill step gathered samples")
                _check(
                    len(broadcast_calls) == before,
                    "the no-sample skip gate posted a broadcast",
                )
            _check(pending[-1] is None, "a skipped step queued a payload")
            continue

        # The step's plan is priced from the block this generation verifies.
        req_ids = [_REQ_IDS[slot] for slot in step.slots]
        idx_mapping = torch.tensor(step.slots, dtype=torch.int64)
        capacity_np = np.array(step.capacities, dtype=np.int32)
        num_tokens_per_req = {
            req_id: int(capacity) + 1
            for req_id, capacity in zip(req_ids, step.capacities)
        }
        draft_tokens = {
            req_id: list(range(int(capacity)))
            for req_id, capacity in zip(req_ids, step.capacities)
        }
        cu_num_logits_np = np.zeros(len(req_ids) + 1, dtype=np.int32)
        np.cumsum(capacity_np + 1, out=cu_num_logits_np[1:])
        num_tokens = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
        compacted, cu_num_logits_cpu = manager.compact_batch(
            capacity_np,
            np.array([capacity + 1 for capacity in step.capacities], dtype=np.int32),
            cu_num_logits_np,
        )
        cu_num_logits, query_start_loc, draft_budget = manager.reallocate_drafts(
            req_ids, idx_mapping
        )
        admitted = manager._batch_draft_capacity[: len(req_ids)].tolist()
        plan = (
            int(num_tokens),
            tuple(int(value) for value in compacted.tolist()),
            tuple(int(value) for value in cu_num_logits_cpu.tolist()),
            int(draft_budget),
            tuple(int(value) for value in admitted),
            tuple(int(value) for value in cu_num_logits.tolist()),
            tuple(int(value) for value in query_start_loc.tolist()),
        )
        gathered: list = [None] * _PP_SIZE
        dist.all_gather_object(gathered, plan)
        _check(gathered[0] == gathered[1], "the two ranks priced different plans")
        _check(plan[3] == sum(plan[4]), "admitted capacity does not match the budget")
        _check(
            all(
                admitted_count <= capacity
                for admitted_count, capacity in zip(plan[4], step.capacities)
            ),
            "the budget admitted more drafts than the scheduler scheduled",
        )

        proposal = _proposal(step_index, step.slots, test_gen)
        before = len(broadcast_calls)
        if is_last_rank:
            sampled = torch.from_numpy(proposal.sampled)
            handler.broadcast(
                sampled, proposal.num_sampled, proposal.num_rejected, batch
            )
            local_draft_table[list(step.slots)] = torch.from_numpy(proposal.drafts)
            handler.broadcast_drafts(
                local_draft_table, batch, torch.from_numpy(proposal.confidences)
            )
        else:
            _check(handler.receive(batch), "receive dropped a row that needed a sample")
        _check(
            len(broadcast_calls) - before == 4,
            "the adaptive relay did not post four broadcasts",
        )
        pending[-1] = proposal

    _check(all(proposal is None for proposal in pending), "unconsumed relay payload")
    torch.distributed.broadcast = real_broadcast


def _run_rank(rank: int, case: str, init_method: str, report_conn) -> None:
    import torch.distributed as dist

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=_PP_SIZE,
        )
        _install_cpu_shims()
        from vllm.v1.worker.gpu import pp_utils
        from vllm.v1.worker.gpu.spec_decode import adaptive_verification

        pp_utils.async_copy_to_gpu = _cpu_copy
        adaptive_verification.async_copy_to_gpu = _cpu_copy

        try:
            _drive(pp_utils, adaptive_verification, rank, case)
        except AssertionError as exc:
            if case not in _MUTATIONS:
                raise
            if _FAIL not in str(exc):
                raise
        else:
            if case in _MUTATIONS:
                raise AssertionError(f"{_FAIL}: mutation {case} survived")
    except BaseException:
        report = traceback.format_exc()
        print(report, file=sys.stderr, flush=True)
        report_conn.send(report)
        raise
    else:
        report_conn.send("")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_ranks(case: str) -> None:
    init_method = get_file_store_init_method()
    context = mp.get_context("spawn")
    pipes = [context.Pipe(duplex=False) for _ in range(_PP_SIZE)]
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, case, init_method, pipes[rank][1]),
        )
        for rank in range(_PP_SIZE)
    ]
    for process in processes:
        process.start()
    for _, send_conn in pipes:
        send_conn.close()
    for process in processes:
        process.join(timeout=_JOIN_TIMEOUT_S)
    stuck = [process for process in processes if process.is_alive()]
    for process in stuck:
        process.kill()
        process.join()
    reports = []
    for recv_conn, _ in pipes:
        reports.append(recv_conn.recv() if recv_conn.poll(5) else "")
        recv_conn.close()
    failed = [report for report in reports if report]
    assert not stuck, f"gloo rank did not finish within {_JOIN_TIMEOUT_S}s"
    assert all(process.exitcode == 0 for process in processes), (
        "gloo rank failed:\n" + "\n".join(failed)
    )


@pytest.mark.parametrize("case", ["normal", *_MUTATIONS])
def test_pp_adaptive_relay_over_gloo(case):
    _run_ranks(case)
