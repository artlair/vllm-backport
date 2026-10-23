# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.pp_utils import PendingConfidences
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so a global top-k always admits
    continuously along steps with a request.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


_assign_draft_token_budget_compiled = torch.compile(
    _assign_draft_token_budget, dynamic=True
)


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cost tables: graph-padded below the capture limit, smooth above.

    Args:
        cudagraph_limit: Largest cudagraph-captured size. At or below it,
            execution pads up to the next captured size, so cost is a step
            function. Above it there is no padding, so cost is continuous.
    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # Execution pads to the next captured size, so cost is a step
        # function of the padded size: smooth interpolation would invent
        # marginal per-token costs that don't exist within a pad bucket.
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Past the capture limit nothing pads, so cost really is continuous in
        # size; snapping to the next profiled point overestimates badly when
        # the profiled points are far apart. Interpolate only between points
        # that are themselves past the limit: crossing the limit loses
        # cudagraphs entirely, which is a genuine discontinuity, so the first
        # point above it must not be blended with the last one below.
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


_UNPROFILED_COSTS = (
    "Adaptive verification could not profile step costs. Pass "
    "`enable_adaptive_verification=false` in the speculative config to "
    "verify a fixed number of drafts instead."
)


def _median_cost_curve(
    points: Iterable[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Median cost per distinct size, in ascending size order."""
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for size, cost_ms in points:
        grouped[size].append(cost_ms)
    return [(size, float(np.median(v))) for size, v in sorted(grouped.items())]


def stage_cost_curves(
    samples: Sequence[StepTimingSample],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """(draft, verify) cost curves measured by one PP stage.

    Eager-target steps inflate drafter timings (the CPU is still launching
    kernels, opening gaps between the drafter's events), and request counts,
    unlike token counts, collide across execution modes, so only graph-replay
    samples may price the draft curve. A stage that runs no drafter reports 0.0
    for every step, which only the last stage's curve may be read from.
    """
    draft_curve = _median_cost_curve(
        (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph
    )
    verify_curve = _median_cost_curve(
        (s.num_target_tokens, s.forward_ms) for s in samples
    )
    return draft_curve, verify_curve


def _profile_signature(
    samples: Sequence[StepTimingSample],
) -> tuple[tuple[int, int, bool], ...]:
    """The shape cases one stage profiled, for cross-stage agreement."""
    return tuple(
        sorted((s.num_reqs, s.num_target_tokens, s.full_cudagraph) for s in samples)
    )


def _describe_profile(signature: Sequence[tuple[int, int, bool]]) -> str:
    token_counts = sorted({num_tokens for _, num_tokens, _ in signature})
    return f"{len(signature)} dummy steps over token counts {token_counts}"


def aggregate_pp_cost_curves(
    stage_samples: Sequence[Sequence[StepTimingSample]],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Combine per-stage profile samples into pipeline cost curves.

    A verification step runs serially through the pipeline stages, so its cost
    is the sum of the per-stage costs at the same token count. Only the last
    stage runs the drafter, so only its draft curve prices drafting.

    Summing is deliberately conservative: a steady-state microbatch is bounded
    by the slowest stage rather than the sum, so this over-prices verification
    and therefore trims at least as much as single-stage adaptive verification
    would. It also omits the activation hand-off between stages and the
    pipeline fill/drain, which dummy runs never issue. Both omissions are
    performance-only: the budget never exceeds the scheduled draft count.
    """
    if not stage_samples:
        raise RuntimeError(_UNPROFILED_COSTS)
    signatures = [_profile_signature(samples) for samples in stage_samples]
    reference = signatures[0]
    for stage, signature in enumerate(signatures[1:], start=1):
        if signature != reference:
            raise RuntimeError(
                "Adaptive verification sums per-stage verify costs, so every PP "
                "stage must capture and profile the same shapes, but PP stage "
                f"{stage} profiled {_describe_profile(signature)} while PP stage "
                f"0 profiled {_describe_profile(reference)}."
            )
    stage_curves = [stage_cost_curves(samples) for samples in stage_samples]
    draft_curve = stage_curves[-1][0]
    if not draft_curve:
        raise RuntimeError(
            "Adaptive verification could not profile draft step costs on the "
            "stage that runs the drafter. Pass "
            "`enable_adaptive_verification=false` in the speculative config to "
            "verify a fixed number of drafts instead."
        )
    verify_costs: dict[int, float] = {}
    for _, verify_curve in stage_curves:
        for num_tokens, cost_ms in verify_curve:
            verify_costs[num_tokens] = verify_costs.get(num_tokens, 0.0) + cost_ms
    return draft_curve, sorted(verify_costs.items())


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        pp_world_size: int = 1,
        pp_rank: int = 0,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

        # Under PP every rank prices the generation this step verifies, which
        # arrives through the relay's confidence payload. That payload is the
        # only writer of this per-request host table, so all ranks read the same
        # values for the same generation.
        self.pp_world_size = pp_world_size
        self.pp_rank = pp_rank
        self._confidence_cpu: np.ndarray | None = None
        if pp_world_size > 1:
            self._confidence_cpu = np.ones(
                (max_num_reqs, self.num_speculative_steps), dtype=np.float32
            )

        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        if self._confidence_cpu is not None:
            # The install runs before requests are added, so a slot reused by a
            # new request never inherits its predecessor's block.
            self._confidence_cpu[req_idx] = 1.0
        else:
            self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
            self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def install_confidences(self, payload: "PendingConfidences") -> None:
        """Adopt the confidences of the generation this step verifies.

        Called on every PP rank with the payload the draft relay carried, before
        the budget prices it and before requests can be added or freed.
        """
        assert self._confidence_cpu is not None
        assert payload.confidences.shape[1] == self.num_speculative_steps
        assert payload.confidences_np.shape == payload.confidences.shape
        assert payload.idx_mapping_np.shape == (payload.confidences.shape[0],)
        if payload.confidences.shape[0] == 0:
            return
        # Fail closed on rows that cannot be survival probabilities: a NaN or an
        # out-of-range value would silently collapse the budget.
        assert np.isfinite(payload.confidences_np).all(), (
            "adaptive verification received non-finite draft confidences"
        )
        assert (payload.confidences_np >= 0.0).all(), (
            "adaptive verification received negative draft confidences"
        )
        assert (payload.confidences_np <= 1.0).all(), (
            "adaptive verification received draft confidences above one"
        )
        idx_mapping_np = payload.idx_mapping_np
        if self._confidence_probs.is_cuda:
            # A NumPy index would stage an unpinned H2D copy, which the GPU sync
            # checker flags in the middle of the step.
            idx_mapping = async_copy_to_gpu(
                idx_mapping_np, device=self.req_states.device
            )
        else:
            idx_mapping = idx_mapping_np
        self._confidence_probs[idx_mapping] = payload.confidences
        self._confidence_cpu[idx_mapping_np] = payload.confidences_np

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Dummy-run kwargs whose step timings seed the cost tables.

        Run these inside StepTimingCollector.collect(), then hand the block's
        timings to set_initial_cost_curves.

        Batches are decode-shaped: a trimmed verification step is a few token
        rows spread over its requests, so each batch carries up to
        decode_query_len rows per request (a single request for the smallest
        rungs) and only falls back to an even split once the token count no
        longer fits that many requests.
        """
        max_num_reqs = self.req_states.max_num_reqs
        max_num_tokens = self.req_states.max_num_batched_tokens
        decode_query_len = self.num_speculative_steps + self.num_bonus_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # Also profile beyond the capture limit: real steps run there
        # (piecewise/eager) and linear extrapolation badly underestimates
        # them. These runs double as JIT warmup for the piecewise shapes.
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            num_reqs = min(max_num_reqs, cdiv(num_tokens, decode_query_len))
            context_len = self._profile_context_len(cdiv(num_tokens, num_reqs))
            for _ in range(_PROFILE_REPLAYS):
                yield {
                    "num_tokens": num_tokens,
                    "num_reqs": num_reqs,
                    "context_len": context_len,
                }

    def _profile_context_len(self, max_query_len: int) -> int:
        """Profiled decode context, clamped to what a dummy batch can carry.

        set_dummy_context fabricates this much context per request from the KV
        pool, so the clamp keeps the profiled request inside max_model_len.
        """
        context_len = envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
        max_model_len = getattr(self.req_states, "max_model_len", 0)
        if max_model_len:
            clamped = max(min(context_len, max_model_len - max_query_len), 0)
            if clamped != context_len:
                logger.debug_once(
                    "Clamping the adaptive verification profile context to %d "
                    "tokens to fit max_model_len=%d at query length %d.",
                    clamped,
                    max_model_len,
                    max_query_len,
                )
            context_len = clamped
        return context_len

    def profile_shape(self, capture_sizes: list[int]) -> dict[str, int | list[int]]:
        """The shape of this stage's profile program, for stage agreement."""
        return {
            "capture_sizes": list(capture_sizes),
            "decode_query_len": self.num_speculative_steps + self.num_bonus_tokens,
            "max_num_reqs": self.req_states.max_num_reqs,
            "max_num_batched_tokens": self.req_states.max_num_batched_tokens,
            "max_model_len": getattr(self.req_states, "max_model_len", 0),
            "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
        }

    def _pp_cpu_group(self) -> torch.distributed.ProcessGroup:
        """This rank's pipeline-stage CPU group, for startup object collectives."""
        return get_pp_group().cpu_group

    def check_pp_profile_agreement(self, capture_sizes: list[int]) -> None:
        """Fail unless every PP stage profiles the same batches.

        The verify curve is summed across stages, so a stage that captured a
        different ladder or schedules different shapes would produce a
        meaningless total. The gather runs before any check, so no stage can
        leave its peers waiting inside the profiling loop.
        """
        pp_world_size = getattr(self, "pp_world_size", 1)
        if pp_world_size <= 1:
            return
        local = self.profile_shape(capture_sizes)
        shapes: list[dict[str, int | list[int]]] = [local for _ in range(pp_world_size)]
        torch.distributed.all_gather_object(shapes, local, group=self._pp_cpu_group())
        reference = shapes[0]
        mismatched = [stage for stage, shape in enumerate(shapes) if shape != reference]
        if mismatched:
            stage = mismatched[0]
            raise RuntimeError(
                "Adaptive verification sums per-stage verify costs, so every PP "
                "stage must capture and profile the same shapes, but PP stage "
                f"{stage} profiles {shapes[stage]} while PP stage 0 profiles "
                f"{reference}."
            )

    def _gather_stage_samples(
        self, samples: list[StepTimingSample]
    ) -> list[list[StepTimingSample]]:
        """Every PP stage's profile samples, in stage order."""
        pp_world_size = getattr(self, "pp_world_size", 1)
        if pp_world_size <= 1:
            return [samples]
        stage_samples: list[list[StepTimingSample]] = [[] for _ in range(pp_world_size)]
        torch.distributed.all_gather_object(
            stage_samples, samples, group=self._pp_cpu_group()
        )
        return stage_samples

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        """Turn this startup's profile samples into the cost tables.

        Under pipeline parallelism every stage contributes its curve to the
        total verify cost, so the stage samples are gathered before the curves
        are assembled and validated.
        """
        draft_curve, verify_curve = aggregate_pp_cost_curves(
            self._gather_stage_samples(samples)
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def record_confidences(
        self,
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        if self._confidence_cpu is not None:
            # Under PP the relay payload is the only writer of both tables: this
            # step's block reaches every rank, this one included, when the step
            # that verifies it runs. Publishing it here would leave this rank a
            # generation ahead of its peers on any step that proposes without
            # shipping a payload.
            return
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        host_confidences = (
            self._confidence_cpu[slots]
            if self._confidence_cpu is not None
            else self._stale_confidences[self._stale_idx].np[slots]
        )
        survival_probability = np.cumprod(host_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # The draft budget is 0, so we can know cu_num_logits_np exactly. This helps
            # when we would exceed the sampler logit chunk size.
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids keeps verification requests at the front.
        assert np.all(is_verification_request[:num_verification_reqs])
        # for the CPU side buffer we distribute draft tokens evenly
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

        # Rank draft slots by survival probability and admit the best prefix.
        # capacities enters holding each request's valid draft count (the kernel
        # uses it to bound eligible slots) and leaves holding the admitted count.
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_copy_to_gpu(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                _assign_draft_token_budget_compiled(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self.num_speculative_steps,
                )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_copy_to_gpu(
            num_non_draft_tokens,
            out=num_non_draft_tokens_gpu,
        )
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + num_non_draft_tokens_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
        )


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list["AttentionGroup"]],
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    vllm_config: "VllmConfig",
    target_layer_names: set[str] | None = None,
    additional_attn_cg_support: tuple[AttentionCGSupport, str | None] | None = None,
    pp_world_size: int = 1,
    pp_rank: int = 0,
) -> AdaptiveVerificationManager | None:
    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends, but models that
    # hard-wire theirs (e.g. DeepSeek-V4) never go through it.
    backend = get_query_lens_mismatch_unsupported_backend(
        attn_groups,
        checked_layer_names=target_layer_names,
    )
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    target_attn_cg_support = attn_cg_support
    if target_layer_names is not None:
        target_attn_cg_support = get_attn_cg_support(
            attn_groups,
            vllm_config,
            checked_layer_names=target_layer_names,
        )
        if additional_attn_cg_support is not None:
            target_attn_cg_support = target_attn_cg_support.narrow(
                *additional_attn_cg_support
            )
    if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS:
        raise ValueError(
            "Adaptive verification captures varlen decode cudagraphs, so every"
            " target attention builder must report AttentionCGSupport.ALWAYS, but "
            f"{target_attn_cg_support.min_cg_attn_backend} reports "
            f"{target_attn_cg_support.min_cg_support}. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
        pp_world_size=pp_world_size,
        pp_rank=pp_rank,
    )
