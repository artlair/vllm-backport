# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    aggregate_pp_cost_curves,
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
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
    curves: dict[str, list[tuple[int, float]]] = {}
    manager.set_cost_curves = lambda draft, verify: curves.update(
        draft=draft, verify=verify
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_reqs"],
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
    assert curves["draft"] == [(1, 1.0), (64, 1.0)]


def make_profile_manager(
    capture_sizes: list[int],
    *,
    max_num_reqs: int = 16,
    max_num_batched_tokens: int = 2048,
    num_speculative_steps: int = 4,
    num_bonus_tokens: int = 1,
    max_model_len: int | None = None,
) -> AdaptiveVerificationManager:
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_reqs=max_num_reqs,
    )
    if max_model_len is not None:
        manager.req_states.max_model_len = max_model_len
    manager.num_speculative_steps = num_speculative_steps
    manager.num_bonus_tokens = num_bonus_tokens
    manager._cudagraph_limit = 0
    return manager


def profile_samples(
    batches: list[dict[str, int]],
    *,
    forward_scale: float = 1.0,
    drafter_ms: float = 0.0,
    full_cudagraph: bool = True,
) -> list[StepTimingSample]:
    """One stage's profile samples for a batch program."""
    return [
        StepTimingSample(
            forward_ms=forward_scale * batch["num_tokens"],
            drafter_ms=drafter_ms,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_reqs"],
            full_cudagraph=full_cudagraph,
        )
        for batch in batches
    ]


def test_profile_batches_are_decode_shaped():
    # A trimmed verify step shrinks a decode block over a fixed request count,
    # so each profiled size must be spread over decode-shaped requests: a whole
    # block per request at most, and no rounding of the sub-block rungs.
    manager = make_profile_manager([1, 2, 3, 4, 5, 10, 15, 40])
    decode_query_len = manager.num_speculative_steps + manager.num_bonus_tokens

    batches = list(manager.batches_to_profile([1, 2, 3, 4, 5, 10, 15, 40]))
    shape = {(batch["num_tokens"], batch["num_reqs"]) for batch in batches}

    # One request per block up to a full decode block, then as many requests as
    # the size needs, never rounding a small rung up to a full block.
    assert shape == {
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (10, 2),
        (15, 3),
        (40, 8),
        # Tail sizes past the capture limit stay mixed/piecewise: 16 requests
        # carrying more than one block each.
        (60, 12),
        (80, 16),
        (160, 16),
        (320, 16),
        (640, 16),
        (1280, 16),
        (2048, 16),
    }
    for batch in batches:
        num_reqs = batch["num_reqs"]
        if batch["num_tokens"] <= manager.req_states.max_num_reqs:
            assert -(-batch["num_tokens"] // num_reqs) <= decode_query_len
        context_len = batch["context_len"]
        assert context_len > 0


def test_profile_context_respects_max_model_len():
    # The context is fabricated from the KV pool, so it must fit the model's
    # servable length at the profiled query length instead of relying on the
    # downstream clamp.
    manager = make_profile_manager(
        [5, 40], max_model_len=2048, max_num_batched_tokens=2304
    )
    batches = {
        (batch["num_tokens"], batch["num_reqs"]): batch["context_len"]
        for batch in manager.batches_to_profile([5, 40])
    }
    assert batches[(5, 1)] == 2048 - 5
    # The eager tail spreads the token count evenly over every request, which
    # shrinks the context by the resulting query length.
    assert batches[(2304, 16)] == 2048 - 2304 // 16


def test_profile_context_keeps_the_configured_value_when_it_fits(monkeypatch):
    # The knob stays opt-in: a deployment that raises it gets exactly what it
    # asked for as long as it fits the model length.
    monkeypatch.setattr(
        adaptive_module.envs, "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", 131072
    )
    manager = make_profile_manager([5, 40], max_model_len=524288)
    assert {batch["context_len"] for batch in manager.batches_to_profile([5, 40])} == {
        131072
    }


@pytest.mark.parametrize("num_stages", [1, 2, 6])
def test_pp_curves_sum_verify_costs_and_keep_the_last_draft_curve(num_stages):
    # A verify step serializes through the stages, so its cost is the sum of the
    # stage costs; the drafter only runs on the last stage.
    capture_sizes = [1, 2, 3, 4, 5, 10]
    manager = make_profile_manager(capture_sizes)
    batches = list(manager.batches_to_profile(capture_sizes))
    stage_samples = [
        profile_samples(
            batches,
            forward_scale=float(stage + 1),
            drafter_ms=0.0 if stage < num_stages - 1 else 4.0,
        )
        for stage in range(num_stages)
    ]

    draft_curve, verify_curve = aggregate_pp_cost_curves(stage_samples)

    cost_sum = sum(range(1, num_stages + 1))
    num_tokens_seen = sorted({batch["num_tokens"] for batch in batches})
    assert verify_curve == [
        (num_tokens, float(cost_sum * num_tokens)) for num_tokens in num_tokens_seen
    ]
    num_reqs_seen = sorted({batch["num_reqs"] for batch in batches})
    assert draft_curve == [(num_reqs, 4.0) for num_reqs in num_reqs_seen]


def test_single_stage_curves_match_the_stage_medians():
    manager = make_profile_manager([5, 10])
    batches = list(manager.batches_to_profile([5, 10]))
    samples = profile_samples(batches, forward_scale=2.0, drafter_ms=3.0)

    assert aggregate_pp_cost_curves([samples]) == adaptive_module.stage_cost_curves(
        samples
    )


def test_pp_curves_reject_stages_that_profiled_different_sizes():
    manager = make_profile_manager([1, 2, 3, 4, 5, 10])
    batches = list(manager.batches_to_profile([1, 2, 3, 4, 5, 10]))
    complete = profile_samples(batches, drafter_ms=1.0)
    # A stage that captured a different ladder, or dropped a rung's replays,
    # would silently under-count the pipeline cost.
    truncated = profile_samples(batches[:-1], drafter_ms=1.0)

    with pytest.raises(RuntimeError, match="must capture and profile the same"):
        aggregate_pp_cost_curves([complete, truncated])
    # A stage with no samples at all is the same failure, not a zero-cost stage.
    with pytest.raises(RuntimeError, match="must capture and profile the same"):
        aggregate_pp_cost_curves([complete, []])


def test_pp_curves_reject_stages_that_group_tokens_into_other_requests():
    # Same token counts, different request shapes: the stage priced a batch this
    # feature never runs, so its cost cannot enter the total.
    manager = make_profile_manager([1, 2, 3, 4, 5, 10])
    batches = list(manager.batches_to_profile([1, 2, 3, 4, 5, 10]))
    decode_shaped = profile_samples(batches, drafter_ms=1.0)
    one_token_per_request = profile_samples(
        [{**batch, "num_reqs": batch["num_tokens"]} for batch in batches],
        drafter_ms=1.0,
    )

    with pytest.raises(RuntimeError, match="must capture and profile the same"):
        aggregate_pp_cost_curves([decode_shaped, one_token_per_request])


def test_pp_curves_reject_a_last_stage_without_draft_costs():
    # Only graph-replay samples price the draft curve, so a stage that never
    # replayed one has no drafter cost to offer.
    manager = make_profile_manager([5, 10])
    eager_samples = profile_samples(
        list(manager.batches_to_profile([5, 10])), full_cudagraph=False
    )

    with pytest.raises(RuntimeError, match="could not profile draft step costs"):
        aggregate_pp_cost_curves([eager_samples, eager_samples])


def test_pp_curves_reject_an_empty_profile():
    with pytest.raises(RuntimeError, match="could not profile step costs"):
        aggregate_pp_cost_curves([])


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
