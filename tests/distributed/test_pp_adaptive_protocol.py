# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The adaptive PP relay over a real gloo transport.

Spawned ranks run the production ``PPHandler`` send/receive path, the
production confidence ring, and the production confidence install and draft
budget. The payload crosses a real gloo broadcast, so a sender and receiver
that disagree about the four-op FIFO, the skip gate or the generation a block
belongs to hang or fail instead of agreeing with themselves in one process.
The consume side calls the real ``GPUModelRunner.update_pp_decode_requests``
on a stub runner whose only non-production part is the CUDA postprocess kernel.

The happy path runs at two world sizes, including a mixed batch with sampled
decode rows and a non-final prefill row. CPU only: the CUDA
stream/event/record_stream calls are stubbed inside the child processes. This
proves the host-side protocol, per-row masking and generation alignment, not
NCCL stream safety or device-side masking.
"""

import multiprocessing as mp
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.utils.network_utils import get_file_store_init_method

_PP_SIZES = (2, 6)
_MUTATION_PP_SIZE = 2
_NUM_SPEC = 2
_MAX_REQS = 4
_REQ_IDS = tuple(f"r{slot}" for slot in range(_MAX_REQS))
_JOIN_TIMEOUT_S = 120
_REPORT_TIMEOUT_S = 5
_FAIL = "adaptive pp relay"


@dataclass(frozen=True)
class _Step:
    """One scheduler step: the request slots it runs and the draft capacity of
    each decode row. Slots named in `nonfinal_prefill` are non-final prefill
    chunks: they produce no sample and schedule no drafts. `free` slots are
    released after the step's consume."""

    slots: tuple[int, ...]
    capacities: tuple[int, ...] = ()
    nonfinal_prefill: tuple[int, ...] = ()
    free: tuple[int, ...] = ()

    def need_sampled(self) -> tuple[bool, ...]:
        return tuple(slot not in self.nonfinal_prefill for slot in self.slots)


_SCRIPT = (
    _Step((0, 1, 2), (2, 2, 2)),
    # Reordered batch and a request whose draft capacity shrank.
    _Step((2, 0, 1), (1, 2, 2)),
    # Mixed step: slot 1 is a non-final prefill row and schedules no drafts.
    # Verification rows stay at the front, as production sorts them.
    _Step((0, 2, 1), (2, 2, 0), nonfinal_prefill=(1,)),
    # All-prefill step; slot 0 is freed and reused at the same point.
    _Step((0, 1, 2), nonfinal_prefill=(0, 1, 2), free=(0,)),
    # Slot 1's first decode row after its prefill chunk.
    _Step((1, 2, 0), (2, 2, 2)),
    _Step((0, 2), (2, 1), free=(1,)),
    _Step((2, 1, 0), (1, 1, 1)),
    _Step((0, 1, 2), (2, 2, 2)),
) + (_Step(()),) * 6

_MUTATIONS = ("drop_confidences", "no_generation_filter", "asymmetric_mask_filter")


@dataclass(frozen=True)
class _Proposal:
    step_index: int
    slots: tuple[int, ...]
    gens: tuple[int, ...]
    need_sampled: tuple[bool, ...]
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


def _make_manager(rank: int, pp_size: int):
    import numpy as np
    import torch

    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
        build_cost_tables_from_curves,
    )

    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = _NUM_SPEC
    manager.num_bonus_tokens = 1
    manager.pp_world_size = pp_size
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


def _make_handler(rank: int, pp_utils, pp_size: int):
    import torch
    import torch.distributed as dist

    pp_group = SimpleNamespace(
        is_last_rank=rank == pp_size - 1,
        last_rank=pp_size - 1,
        world_size=pp_size,
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
    nonfinal = np.array(
        [slot in step.nonfinal_prefill for slot in step.slots], dtype=bool
    )
    capacities = np.array(step.capacities or (0,) * num_reqs, dtype=np.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.where(nonfinal, 0, 100).astype(np.int32),
        prefill_len_np=np.where(nonfinal, 4096, 100).astype(np.int32),
        num_scheduled_tokens=np.where(nonfinal, 4, capacities + 1).astype(np.int32),
        idx_mapping=torch.tensor(step.slots, dtype=torch.int64),
        idx_mapping_np=np.array(step.slots, dtype=np.intp),
    )


def _confidence_block(slots: tuple[int, ...], step: int):
    """Per-slot survival probabilities, all inside [0, 1]."""
    import numpy as np

    rows = []
    for slot in slots:
        base = 0.9 - 0.1 * slot - 0.01 * (step % 3)
        rows.append([base - 0.08 * position for position in range(_NUM_SPEC)])
    block = np.array(rows, dtype=np.float32)
    _check(
        bool(((block >= 0.0) & (block <= 1.0)).all()),
        "confidence block left [0, 1]",
    )
    return block


def _proposal(step_index: int, step: _Step, test_gen):
    import numpy as np
    import torch

    slot_ids = step.slots
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
        step_index=step_index,
        slots=slot_ids,
        gens=tuple(int(test_gen[slot]) for slot in slot_ids),
        need_sampled=step.need_sampled(),
        confidences=_confidence_block(slot_ids, step_index),
        drafts=drafts,
        sampled=sampled,
        num_sampled=torch.ones(len(slot_ids), dtype=torch.int32),
        num_rejected=torch.zeros(len(slot_ids), dtype=torch.int32),
    )


def _consume(
    runner,
    GPUModelRunner,
    manager,
    pending,
    test_gen,
    local_draft_table,
    order,
):
    import numpy as np
    import torch

    proposal = pending.popleft()
    pending.append(None)
    order.events.clear()
    order.budget_started = False
    order.installed.clear()
    postprocess_calls: list[dict] = []

    def postprocess_sampled(**kwargs):
        _check("confidences" not in kwargs, "postprocess received confidences")
        postprocess_calls.append(kwargs)

    runner.postprocess_sampled = postprocess_sampled
    before_confidences = manager._confidence_cpu.copy()
    before_drafts = local_draft_table.clone()
    GPUModelRunner.update_pp_decode_requests(runner)

    if proposal is None:
        _check(not postprocess_calls, "postprocess ran without an in-flight proposal")
        _check(
            not order.installed,
            "confidence installed without an in-flight proposal",
        )
        return None

    live = np.array(
        [test_gen[slot] == gen for slot, gen in zip(proposal.slots, proposal.gens)],
        dtype=bool,
    )
    keep = live & np.array(proposal.need_sampled, dtype=bool)
    expected_slots = np.array(proposal.slots, dtype=np.intp)[keep]
    if expected_slots.size == 0:
        _check(not postprocess_calls, "a fully dropped block still ran postprocess")
        _check(not order.installed, "a fully dropped block still installed confidences")
        return proposal, keep

    _check(len(order.installed) == 1, "missing generation-aligned confidence payload")
    confidences = order.installed[0]
    _check(
        confidences.idx_mapping_np.tolist() == expected_slots.tolist(),
        "payload rows do not match the live sampled proposal slots",
    )
    np.testing.assert_allclose(confidences.confidences_np, proposal.confidences[keep])
    for row, slot in enumerate(proposal.slots):
        if keep[row]:
            np.testing.assert_allclose(
                manager._confidence_cpu[slot], proposal.confidences[row]
            )
        else:
            _check(
                np.allclose(manager._confidence_cpu[slot], before_confidences[slot]),
                "a masked or freed row overwrote its confidence block",
            )

    if not postprocess_calls:
        # The last rank has no deferred sample queue; its self-ring carries only
        # confidences, so there is no sample payload to postprocess here.
        return proposal, keep
    _check(len(postprocess_calls) == 1, "postprocess ran more than once for a payload")
    outputs = postprocess_calls[0]
    expected_idx = np.where(keep, np.array(proposal.slots), -1)
    _check(
        outputs["idx_mapping"].numpy().tolist() == expected_idx.tolist(),
        "postprocess rows do not match the per-row sample mask",
    )
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
        if keep[row]:
            np.testing.assert_array_equal(
                local_draft_table[slot].numpy(), proposal.drafts[row]
            )
        else:
            _check(
                torch.equal(local_draft_table[slot], before_drafts[slot]),
                "a masked or freed slot inherited an in-flight draft block",
            )
    return proposal, keep


def _drive(pp_utils, adaptive_module, rank: int, pp_size: int, case: str) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if case == "drop_confidences":
        original_push = pp_utils.AdaptiveConfidenceRing.push

        def drop_push(ring, *args, **kwargs):
            original_push(ring, *args, **kwargs)
            return None

        pp_utils.AdaptiveConfidenceRing.push = drop_push
    elif case == "no_generation_filter":
        pp_utils.PPHandler.on_req_idx_freed = lambda self, req_idx: None
    elif case == "asymmetric_mask_filter":
        original_collect = pp_utils.ConfidenceSlot.collect

        def collect_without_row_mask(slot, req_idx_gen_np):
            # Break the mask asymmetrically across ranks only on mixed steps:
            # the receiver keeps every row, the last rank drops every row.
            mask = slot.need_sampled_mask
            if not (mask.any() and not mask.all()):
                return original_collect(slot, req_idx_gen_np)
            slot.need_sampled_mask = (
                np.ones_like(mask) if rank != pp_size - 1 else np.zeros_like(mask)
            )
            try:
                return original_collect(slot, req_idx_gen_np)
            finally:
                slot.need_sampled_mask = mask

        pp_utils.ConfidenceSlot.collect = collect_without_row_mask

    # The eager ranking kernel is the body the compiled wrapper runs, without a
    # CPU torch.compile inside the protocol test.
    adaptive_module._assign_draft_token_budget_compiled = (
        adaptive_module._assign_draft_token_budget
    )

    handler = _make_handler(rank, pp_utils, pp_size)
    manager = _make_manager(rank, pp_size)
    is_last_rank = rank == pp_size - 1
    local_draft_table = torch.full((_MAX_REQS, _NUM_SPEC), -1, dtype=torch.int64)
    test_gen = np.zeros(_MAX_REQS, dtype=np.int32)
    pending = deque([None] * pp_size)
    broadcast_calls: list = []
    real_broadcast = torch.distributed.broadcast

    # Install must run before this step prices its budget.
    order = SimpleNamespace(events=[], budget_started=False, installed=[])
    real_install = manager.install_confidences

    def install_confidences(payload):
        _check(not order.budget_started, "confidence installed after budget pricing")
        order.events.append("install")
        order.installed.append(payload)
        real_install(payload)

    manager.install_confidences = install_confidences
    for name in ("get_num_tokens", "compact_batch", "reallocate_drafts"):
        real_budget_step = getattr(manager, name)

        def budget_step(*args, _real=real_budget_step, **kwargs):
            order.budget_started = True
            return _real(*args, **kwargs)

        setattr(manager, name, budget_step)

    # The self-ring is only a fallback for a step whose deferred payload did not
    # carry confidences (always true on the last rank, which has no queue).
    relay = SimpleNamespace(
        self_ring_calls=0,
        deferred_confidences=False,
        deferred_steps=0,
        steps=0,
    )
    real_get = handler.get_prev_sampled_outputs

    def get_prev_sampled_outputs(draft_tokens):
        relay.steps += 1
        outputs = real_get(draft_tokens)
        relay.deferred_confidences = (
            outputs is not None and outputs.get("confidences") is not None
        )
        if relay.deferred_confidences:
            relay.deferred_steps += 1
        return outputs

    handler.get_prev_sampled_outputs = get_prev_sampled_outputs
    real_pop_self = handler.pop_self_confidences

    def pop_self_confidences():
        relay.self_ring_calls += 1
        _check(
            not relay.deferred_confidences,
            "self-ring used while the deferred payload carried confidences",
        )
        return real_pop_self()

    handler.pop_self_confidences = pop_self_confidences

    runner = SimpleNamespace(
        pp_handler=handler,
        adaptive_verification=manager,
        req_states=SimpleNamespace(draft_tokens=local_draft_table),
    )

    def counted_broadcast(tensor, src, group, **kwargs):
        broadcast_calls.append((tuple(tensor.shape), tensor.dtype))
        return real_broadcast(tensor, src=src, group=group, **kwargs)

    torch.distributed.broadcast = counted_broadcast

    used_slots = {slot for step in _SCRIPT for slot in step.slots}
    kept_steps = {slot: set() for slot in used_slots}
    masked_slots: set[int] = set()
    mixed_consumed = False
    transition_consumed = False

    for step_index, step in enumerate(_SCRIPT):
        batch = _make_batch(step)
        consumed = _consume(
            runner,
            GPUModelRunner,
            manager,
            pending,
            test_gen,
            local_draft_table,
            order,
        )
        if consumed is not None:
            proposal, keep = consumed
            for row, slot in enumerate(proposal.slots):
                if not proposal.need_sampled[row]:
                    mixed_consumed = True
                    masked_slots.add(slot)
                elif keep[row] and slot in masked_slots:
                    transition_consumed = True
                if keep[row]:
                    kept_steps[slot].add(proposal.step_index)

        # Production frees and re-adds before this step's batch is planned, but
        # after the deferred payload was installed.
        for slot in step.free:
            test_gen[slot] += 1
            handler.on_req_idx_freed(slot)
            manager.add_request(slot)
            local_draft_table[slot].fill_(-1)

        if not any(step.need_sampled()):
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
        manager.req_states.num_computed_tokens_np[batch.idx_mapping_np] = (
            batch.num_computed_tokens_np
        )
        manager.req_states.prefill_len.np[batch.idx_mapping_np] = batch.prefill_len_np
        req_ids = [_REQ_IDS[slot] for slot in step.slots]
        idx_mapping = torch.tensor(step.slots, dtype=torch.int64)
        nonfinal = np.array(
            [slot in step.nonfinal_prefill for slot in step.slots], dtype=bool
        )
        capacity_np = np.array(step.capacities, dtype=np.int32)
        draft_capacity_np = np.where(nonfinal, 0, capacity_np).astype(np.int32)
        scheduled_np = np.where(nonfinal, 4, capacity_np + 1).astype(np.int32)
        num_tokens_per_req = {
            req_id: int(scheduled) for req_id, scheduled in zip(req_ids, scheduled_np)
        }
        draft_tokens = {
            req_id: list(range(int(capacity)))
            for req_id, capacity in zip(req_ids, draft_capacity_np)
        }
        cu_num_logits_np = np.zeros(len(req_ids) + 1, dtype=np.int32)
        np.cumsum(scheduled_np, out=cu_num_logits_np[1:])
        num_tokens = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
        compacted, cu_num_logits_cpu = manager.compact_batch(
            draft_capacity_np,
            scheduled_np,
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
        gathered: list = [None] * pp_size
        dist.all_gather_object(gathered, plan)
        _check(
            all(other == plan for other in gathered),
            "the ranks priced different plans",
        )
        _check(plan[3] == sum(plan[4]), "admitted capacity does not match the budget")
        _check(
            all(
                admitted_count <= capacity
                for admitted_count, capacity in zip(plan[4], draft_capacity_np)
            ),
            "the budget admitted more drafts than the scheduler scheduled",
        )

        proposal = _proposal(step_index, step, test_gen)
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
            _check(
                handler.receive(batch) == all(proposal.need_sampled),
                "receive return does not match the per-row sample mask",
            )
        _check(
            len(broadcast_calls) - before == 4,
            "the adaptive relay did not post four broadcasts",
        )
        pending[-1] = proposal

    _check(all(proposal is None for proposal in pending), "unconsumed relay payload")
    _check(mixed_consumed, "no mixed batch exposed a per-row sample mask")
    _check(transition_consumed, "a masked prefill row never decoded later")
    _check(
        all(kept_steps[slot] for slot in used_slots),
        "a request never consumed a generation",
    )
    _check(
        all(len(kept_steps[slot]) >= 2 for slot in used_slots),
        "fewer than two request generations were consumed",
    )
    _check(
        relay.self_ring_calls == relay.steps - relay.deferred_steps,
        "self-ring fallback did not match the missing confidence payloads",
    )
    torch.distributed.broadcast = real_broadcast


def _run_rank(
    rank: int, pp_size: int, case: str, init_method: str, report_conn
) -> None:
    import torch.distributed as dist

    expect_failure = case in _MUTATIONS
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=pp_size,
        )
        _install_cpu_shims()
        from vllm.v1.worker.gpu import pp_utils
        from vllm.v1.worker.gpu.spec_decode import adaptive_verification

        pp_utils.async_copy_to_gpu = _cpu_copy
        adaptive_verification.async_copy_to_gpu = _cpu_copy

        try:
            _drive(pp_utils, adaptive_verification, rank, pp_size, case)
        except AssertionError as exc:
            if not expect_failure or _FAIL not in str(exc):
                raise
        else:
            if expect_failure:
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


def _run_ranks(pp_size: int, case: str) -> None:
    init_method = get_file_store_init_method()
    context = mp.get_context("spawn")
    pipes = [context.Pipe(duplex=False) for _ in range(pp_size)]
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, pp_size, case, init_method, pipes[rank][1]),
        )
        for rank in range(pp_size)
    ]
    for process in processes:
        process.start()
    for _, send_conn in pipes:
        send_conn.close()
    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    stuck = [process for process in processes if process.is_alive()]
    for process in stuck:
        process.kill()
    for process in stuck:
        process.join(timeout=_REPORT_TIMEOUT_S)
    reports = []
    for recv_conn, _ in pipes:
        report = ""
        if recv_conn.poll(_REPORT_TIMEOUT_S):
            try:
                report = recv_conn.recv()
            except (EOFError, OSError):
                # The child died before it could report; its exit code below is
                # the only diagnosis we get.
                report = ""
        reports.append(report)
        recv_conn.close()
    failures = []
    if stuck:
        failures.append(f"ranks did not finish within {_JOIN_TIMEOUT_S}s")
    for rank, (process, report) in enumerate(zip(processes, reports)):
        if report:
            failures.append(f"rank {rank} report:\n{report}")
        if process.exitcode != 0:
            failures.append(f"rank {rank} exited with {process.exitcode}")
    assert not failures, "gloo ranks failed:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    ("pp_size", "case"),
    [(pp_size, "normal") for pp_size in _PP_SIZES]
    + [(_MUTATION_PP_SIZE, case) for case in _MUTATIONS],
)
def test_pp_adaptive_relay_over_gloo(pp_size, case):
    _run_ranks(pp_size, case)
