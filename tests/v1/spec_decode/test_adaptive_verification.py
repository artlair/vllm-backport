# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
from vllm.v1.worker.gpu.pp_utils import (
    AdaptiveConfidenceRing,
    PendingConfidences,
)
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    # PP1, i.e. the last-rank-only path these cases pin.
    manager.pp_world_size = 1
    manager.pp_rank = 0
    manager._confidence_cpu = None
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


def test_manager_scopes_varlen_check_without_weakening_runner_cg_mode(monkeypatch):
    class Backend:
        @classmethod
        def supports_device_cpu_query_lens_mismatch(cls):
            return True

    class Builder:
        def __init__(self, support):
            self.support = support

        def get_cudagraph_support(self, *_args):
            return self.support

    def group(layer_name, support):
        builder = Builder(support)
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=Backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group("target", AttentionCGSupport.ALWAYS),
            group("draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    runner_support = AttentionCGSupportInfo(
        AttentionCGSupport.UNIFORM_BATCH, "DraftBackend"
    )
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    manager = maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=True,
        attn_groups=groups,
        attn_cg_support=runner_support,
        req_states=object(),
        query_start_loc=object(),
        num_bonus_tokens=1,
        max_total_logits=1,
        vllm_config=None,
        target_layer_names={"target"},
    )

    assert manager is created
    assert runner_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    manager.pp_world_size = 1
    manager.pp_rank = 0
    manager._confidence_cpu = None
    curves: dict[str, list[tuple[int, float]]] = {}
    manager.set_cost_curves = lambda draft, verify: curves.update(
        draft=draft, verify=verify
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.array([1.0] * 44 + [100.0] * 3),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def test_zero_budget_keeps_one_grammar_row_per_scheduled_draft():
    # The scheduler sizes the grammar bitmask from the *scheduled* drafts
    # (len(drafts) + 1 rows per request), but a zero budget rewrites
    # cu_num_logits_np to bonus-only. Deriving the bitmask -> logits mapping
    # from those rewritten offsets drops rows and trips the
    # `num_masks == len(mapping)` assert in apply_grammar_bitmask.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
        np.ones(64),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    scheduled_spec_decode_tokens = {"low": [1, 2], "high": [3, 4]}
    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40}, scheduled_spec_decode_tokens
    )
    assert manager._batch_budget[2] == 0

    req_ids = ["low", "high", "prefill"]
    num_draft_tokens_per_req = np.array([2, 2, 0], dtype=np.int32)
    _, cu_num_logits_np = manager.compact_batch(
        num_draft_tokens_per_req,
        np.array([3, 3, 40], dtype=np.int32),
        np.array([0, 3, 6, 7], dtype=np.int32),
    )

    mask_stride = manager.num_speculative_steps + manager.num_bonus_tokens
    mapping = _build_grammar_mapping(
        req_ids,
        req_ids,
        cu_num_logits_np,
        num_draft_tokens_per_req,
        manager.num_bonus_tokens,
        mask_stride,
    )

    num_bitmask_rows = sum(
        len(scheduled_spec_decode_tokens.get(req_id, ())) + 1 for req_id in req_ids
    )
    assert len(mapping) == num_bitmask_rows
    # (request, position) keys, so the kernel can mask rows the compacted
    # device layout no longer has room for.
    assert mapping == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# PP protocol: one generation-aligned confidence block per in-flight step
# ---------------------------------------------------------------------------

PP_NUM_SPEC = 2
PP_MAX_REQS = 4
PP_REQ_IDS = ("a", "b", "c", "d")


def pp_cost_tables():
    """A verify curve with a real gradient: the budget is not trivially 0 or
    "verify everything" for any confidence block."""
    return adaptive_module.build_cost_tables_from_curves(
        draft_curve=[(1, 0.2), (4, 0.8)],
        verify_curve=[(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (8, 2.0)],
        max_num_reqs=PP_MAX_REQS,
        max_batch_tokens=32,
    )


def make_pp_manager(
    pp_world_size, stale_confidences, num_spec=PP_NUM_SPEC, device="cpu"
):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_spec
    manager.num_bonus_tokens = 1
    manager.pp_world_size = pp_world_size
    manager.pp_rank = 0
    manager._confidence_cpu = np.ones((PP_MAX_REQS, num_spec), dtype=np.float32)
    manager._confidence_probs = torch.ones(
        (PP_MAX_REQS, num_spec), dtype=torch.float32, device=device
    )
    manager._batch_draft_capacity = torch.zeros(
        PP_MAX_REQS, dtype=torch.int32, device=device
    )
    manager._num_non_draft_tokens = torch.zeros(
        PP_MAX_REQS, dtype=torch.int32, device=device
    )
    manager._cu_num_logits = torch.zeros(
        PP_MAX_REQS + 1, dtype=torch.int32, device=device
    )
    manager.query_start_loc = torch.zeros(
        PP_MAX_REQS + 1, dtype=torch.int32, device=device
    )
    # The PP budget must never read these (they are the PP1 ping-pong slots).
    # Poisoning them differently per rank makes any such read visible.
    manager._stale_confidences = [
        SimpleNamespace(
            np=np.full((PP_MAX_REQS, num_spec), stale_confidences, np.float32)
        )
    ]
    manager._stale_idx = 0
    manager._max_total_logits = 1 << 30
    manager.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(PP_REQ_IDS)},
        num_computed_tokens_np=np.ones(PP_MAX_REQS, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(PP_MAX_REQS, dtype=np.int32)),
        max_num_reqs=PP_MAX_REQS,
        max_num_batched_tokens=32,
        num_speculative_steps=num_spec,
        device=device,
    )
    manager.cost_tables = pp_cost_tables()
    return manager


class PPRelay:
    """Two PP ranks driving the real ring, collect and install path on CPU.

    Step S consumes the payload pushed at S-pp_size and then proposes the one
    for this step, exactly like `get_prev_sampled_outputs` followed by the
    draft relay on a serving step.
    """

    def __init__(self, pp_world_size):
        self.pp_size = pp_world_size
        self.req_idx_gen_np = np.zeros(PP_MAX_REQS, dtype=np.int32)
        # Rank 0 (an earlier stage) and the last rank read different ping-pong
        # slots in PP1, so they are poisoned differently here.
        self.managers = [
            make_pp_manager(pp_world_size, stale_confidences)
            for stale_confidences in (1.0, 0.0)
        ]
        self.rings = [
            AdaptiveConfidenceRing(
                pp_world_size, PP_MAX_REQS, PP_NUM_SPEC, torch.device("cpu")
            )
            for _ in self.managers
        ]
        self.queues = [deque([None] * pp_world_size) for _ in self.managers]

    def consume(self):
        """Every rank installs the block of the generation it verifies."""
        installed = []
        for rank, queue in enumerate(self.queues):
            slot = queue.popleft()
            queue.append(None)
            payload = self.rings[rank].collect(slot, self.req_idx_gen_np)
            if payload is not None:
                self.managers[rank].install_confidences(payload)
            installed.append(payload)
        return installed

    def propose(self, confidences, idx_mapping_np, need_sample=None):
        gen_np = self.req_idx_gen_np[idx_mapping_np]
        if need_sample is None:
            need_sample = np.ones(len(idx_mapping_np), dtype=bool)
        for rank, queue in enumerate(self.queues):
            queue[-1] = self.rings[rank].push(
                confidences, idx_mapping_np, gen_np, need_sample
            )

    def tables(self, req_ids=PP_REQ_IDS):
        slots = [self.managers[0].req_states.req_id_to_index[r] for r in req_ids]
        return [manager._confidence_cpu[slots] for manager in self.managers]


def pp_plans(relay, num_tokens_per_req, draft_tokens):
    """CPU and device plans for the same step on both ranks."""
    req_ids = list(num_tokens_per_req)
    num_reqs = len(req_ids)
    scheduled_drafts = np.array(
        [len(draft_tokens[req_id]) for req_id in req_ids], dtype=np.int32
    )
    num_scheduled = np.fromiter(num_tokens_per_req.values(), np.int32, num_reqs)
    cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(scheduled_drafts + 1, out=cu_num_logits_np[1:])

    plans = []
    for manager in relay.managers:
        num_tokens = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
        compacted, cu_num_logits_cpu = manager.compact_batch(
            scheduled_drafts, num_scheduled, cu_num_logits_np
        )
        _, _, draft_budget = manager._batch_budget
        idx_mapping = torch.as_tensor(
            [manager.req_states.req_id_to_index[req_id] for req_id in req_ids],
            dtype=torch.int64,
        )
        capacities = torch.from_numpy(scheduled_drafts.copy())
        if draft_budget == 0:
            capacities.zero_()
        elif draft_budget < int(scheduled_drafts.sum()):
            # The eager ranking kernel `reallocate_drafts` runs (through its
            # compiled wrapper), so the comparison covers the device plan too.
            adaptive_module._assign_draft_token_budget(
                manager._confidence_probs,
                idx_mapping,
                capacities,
                draft_budget,
                manager.num_speculative_steps,
            )
        plans.append(
            SimpleNamespace(
                num_tokens=num_tokens,
                draft_budget=draft_budget,
                capacities=capacities.tolist(),
                compacted=compacted.tolist(),
                cu_num_logits=cu_num_logits_cpu.tolist(),
            )
        )
    return plans


def assert_same_plan(relay, num_tokens_per_req, draft_tokens):
    last_rank, earlier_rank = pp_plans(relay, num_tokens_per_req, draft_tokens)
    assert vars(last_rank) == vars(earlier_rank)
    return earlier_rank


def decode_batch(num_reqs=3):
    # Every request is one bonus row plus PP_NUM_SPEC drafts.
    num_tokens_per_req = {req_id: 1 + PP_NUM_SPEC for req_id in PP_REQ_IDS[:num_reqs]}
    draft_tokens = {req_id: [1, 2] for req_id in PP_REQ_IDS[:num_reqs]}
    return num_tokens_per_req, draft_tokens


@pytest.mark.parametrize("pp_size", [2, 6])
def test_pp_relay_prices_the_generation_the_step_verifies(pp_size):
    relay = PPRelay(pp_size)
    num_tokens_per_req, draft_tokens = decode_batch()

    scheduled_ids = PP_REQ_IDS[: len(num_tokens_per_req)]
    for step in range(3 * pp_size + 2):
        relay.consume()
        last_rank_table, earlier_rank_table = relay.tables(scheduled_ids)
        # Both ranks read the same generation-aligned block, and neither reads
        # its own (differently poisoned) PP1 ping-pong slot.
        assert np.array_equal(last_rank_table, earlier_rank_table)

        proposed_step = step - pp_size
        if proposed_step < 0:
            # First generations fall back to "verify everything".
            assert (last_rank_table == 1.0).all()
        else:
            expected = [[0.5 + 0.01 * proposed_step, 0.4] for _ in scheduled_ids]
            assert np.allclose(last_rank_table, expected)

        # The step's plan (CPU and device) is identical on every rank.
        plan = assert_same_plan(relay, num_tokens_per_req, draft_tokens)
        assert sum(plan.capacities) == plan.draft_budget

        batch = PP_REQ_IDS[: len(num_tokens_per_req)]
        confidences = torch.tensor(
            [[0.5 + 0.01 * step, 0.4] for _ in batch], dtype=torch.float32
        )
        idx_mapping_np = np.array(
            [relay.managers[0].req_states.req_id_to_index[r] for r in batch],
            dtype=np.intp,
        )
        relay.propose(confidences, idx_mapping_np)


@pytest.mark.parametrize("pp_size", [2, 6])
def test_pp_steps_without_a_draft_block_install_nothing(pp_size):
    """Pure-prefill steps post no payload. Every rank must then keep verifying
    the full block instead of adopting a neighbour's older generation."""
    relay = PPRelay(pp_size)
    num_tokens_per_req, draft_tokens = decode_batch()

    for _ in range(2 * pp_size):
        relay.consume()
    for table in relay.tables():
        assert (table == 1.0).all()

    # The fail-open default is the "verify more" end of the cost model: a block
    # of low confidences must trim strictly more.
    fail_open = assert_same_plan(relay, num_tokens_per_req, draft_tokens)
    relay.propose(
        torch.full((3, PP_NUM_SPEC), 0.2, dtype=torch.float32),
        np.arange(3, dtype=np.intp),
    )
    for _ in range(pp_size):
        relay.consume()
    trimmed = assert_same_plan(relay, num_tokens_per_req, draft_tokens)
    assert trimmed.draft_budget < fail_open.draft_budget


@pytest.mark.parametrize("pp_size", [2, 6])
def test_pp_install_never_overwrites_a_reused_request_slot(pp_size):
    """A payload pushed before a request was freed and its slot reused must not
    land in the new request's row, and the admission reset must survive."""
    relay = PPRelay(pp_size)
    idx_mapping_np = np.array([0], dtype=np.intp)
    block = torch.full((1, PP_NUM_SPEC), 0.9, dtype=torch.float32)

    for _ in range(pp_size):
        relay.consume()
        relay.propose(block, idx_mapping_np)

    # The first payload lands on its own generation.
    relay.consume()
    for table in relay.tables(("a",)):
        assert np.allclose(table[0], 0.9)

    # Preemption is remove-then-add: the slot's generation moves and the new
    # request starts from the fail-open default.
    relay.req_idx_gen_np[0] += 1
    for manager in relay.managers:
        manager.add_request(0)
    for table in relay.tables(("a",)):
        assert (table[0] == 1.0).all()

    # The in-flight payloads of the previous life are filtered, not installed.
    for _ in range(pp_size - 1):
        relay.consume()
        for table in relay.tables(("a",)):
            assert (table[0] == 1.0).all()

    # The re-added request's own block is installed once it is pp_size old.
    relay.propose(
        torch.full((1, PP_NUM_SPEC), 0.7, dtype=torch.float32), idx_mapping_np
    )
    for step in range(pp_size):
        relay.consume()
        expected = 1.0 if step < pp_size - 1 else 0.7
        for table in relay.tables(("a",)):
            assert np.allclose(table[0], expected)


def test_pp_zero_budget_is_identical_on_every_rank():
    """A budget of zero rebuilds the CPU logits layout exactly, so every rank
    has to reach it from the same block."""
    relay = PPRelay(2)
    # One bonus row per request already fills the rejection sampler's chunk, so
    # no draft can be verified regardless of the confidences.
    for manager in relay.managers:
        manager._max_total_logits = 3
    idx_mapping_np = np.arange(3, dtype=np.intp)
    block = torch.full((3, PP_NUM_SPEC), 0.9, dtype=torch.float32)
    for _ in range(2):
        relay.consume()
        relay.propose(block, idx_mapping_np)
    relay.consume()

    plan = assert_same_plan(relay, *decode_batch())
    assert plan.draft_budget == 0
    assert plan.capacities == [0, 0, 0]
    assert plan.compacted == [1, 1, 1]
    assert plan.cu_num_logits == [0, 1, 2, 3]


def test_pp_capacity_shrink_near_max_tokens_bounds_the_plan():
    """The budget is clamped by what the scheduler schedules at the verify step,
    not by what the draft block contained when it was proposed."""
    relay = PPRelay(2)
    idx_mapping_np = np.arange(3, dtype=np.intp)
    block = torch.full((3, PP_NUM_SPEC), 0.9, dtype=torch.float32)
    for _ in range(2):
        relay.consume()
        relay.propose(block, idx_mapping_np)
    relay.consume()

    # "a" is one token from its cap, so only one of its two drafts is scheduled.
    num_tokens_per_req = {"a": 2, "b": 3, "c": 3}
    draft_tokens = {"a": [1], "b": [2, 3], "c": [4, 5]}
    plan = assert_same_plan(relay, num_tokens_per_req, draft_tokens)

    assert plan.capacities[0] <= 1
    assert sum(plan.capacities) <= sum(len(d) for d in draft_tokens.values())
    assert plan.num_tokens == sum(plan.compacted)


def test_pp_confidence_table_is_request_keyed_not_row_keyed():
    """The install is keyed by the slot the block was proposed for, so a
    reordered verification batch cannot pick up another request's confidences."""
    relay = PPRelay(2)
    # Propose for "c" then "a"; verify in the opposite order later.
    idx_mapping_np = np.array([2, 0], dtype=np.intp)
    block = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=torch.float32)
    for _ in range(2):
        relay.consume()
        relay.propose(block, idx_mapping_np)
    for _ in range(2):
        relay.consume()

    for table in relay.tables():
        assert np.allclose(table[2], 0.2)
        assert np.allclose(table[0], 0.8)
        assert np.allclose(table[1], 1.0)


def test_pp_record_confidences_leaves_both_tables_to_the_install():
    """The relay payload is the only PP writer of the device and host tables.

    Publishing this step's freshly proposed block here would put the last rank a
    generation ahead of its peers on any step that proposes but ships no payload.
    """
    relay = PPRelay(2)
    manager = relay.managers[1]
    manager._confidence_probs.fill_(0.25)

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
    )
    manager.record_confidences(
        torch.full((PP_MAX_REQS, PP_NUM_SPEC), 0.9, dtype=torch.float32), input_batch
    )

    assert (manager._confidence_probs == 0.25).all()
    assert (manager._confidence_cpu == 1.0).all()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="checks the device-side scatter"
)


def test_pp_budget_table_differs_from_the_pingpong_slots():
    """The PP budget reads the installed host table, not the PP1 ping-pong
    slots; the two disagree in this test, so a regression is visible."""
    relay = PPRelay(2)
    idx_mapping_np = np.arange(3, dtype=np.intp)
    for _ in range(2):
        relay.consume()
        relay.propose(
            torch.full((3, PP_NUM_SPEC), 0.1, dtype=torch.float32), idx_mapping_np
        )
    relay.consume()

    for manager in relay.managers:
        assert manager._stale_confidences[0].np[0, 0] != pytest.approx(0.1)
        assert manager._confidence_cpu[0, 0] == pytest.approx(0.1)
    # The trim the poisoned slot would not produce.
    plan = assert_same_plan(relay, *decode_batch())
    assert plan.draft_budget < 6


@requires_cuda
def test_pp_install_scatters_into_the_ranking_table_on_device():
    """The budget reads the host table and the ranking kernel reads the device
    table; both must come from the same payload rows."""
    manager = make_pp_manager(2, stale_confidences=1.0, device="cuda")
    payload = PendingConfidences(
        confidences=torch.tensor([[0.3, 0.3], [0.6, 0.6]], device="cuda"),
        confidences_np=np.array([[0.3, 0.3], [0.6, 0.6]], dtype=np.float32),
        idx_mapping_np=np.array([3, 1], dtype=np.intp),
    )

    manager.install_confidences(payload)

    for table in (manager._confidence_cpu, manager._confidence_probs):
        assert np.allclose(table[3].tolist(), [0.3, 0.3])
        assert np.allclose(table[1].tolist(), [0.6, 0.6])
        assert np.allclose(table[0].tolist(), [1.0, 1.0])
