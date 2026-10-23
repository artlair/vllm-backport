# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]
    confidences: "ConfidenceSlot | None" = None


@dataclass
class PendingConfidences:
    """One generation's draft confidences, aligned with the draft block it
    scores. Rows are in batch order; rows whose request is gone or that need no
    sample were dropped when the payload was collected."""

    confidences: torch.Tensor  # [num_rows, num_speculative_steps] f32
    confidences_np: np.ndarray  # host copy of the same rows
    idx_mapping_np: np.ndarray  # [num_rows] request slots


class ConfidenceSlot:
    """One in-flight PP step's confidence payload.

    Owns its pinned host block, so a payload is never overwritten while another
    in-flight step (or a host view of it) still refers to it: the ring that
    hands these out is as deep as the number of in-flight steps.
    """

    def __init__(self, max_num_reqs: int, num_speculative_steps: int, device):
        self.device = device
        pinned = device.type == "cuda"
        # these buffers are mutable runtime state, so allocate them as normal
        with torch.inference_mode(False):
            self.host = torch.zeros(
                max_num_reqs,
                num_speculative_steps,
                dtype=torch.float32,
                pin_memory=pinned,
            )
        self.host_np: np.ndarray = self.host.numpy()
        # Covers the side-stream D2H. Without a device there is nothing to wait
        # for, which is what the CPU protocol tests rely on.
        self.event: torch.cuda.Event | None = (
            torch.cuda.Event(blocking=True) if pinned else None
        )
        self.confidences: torch.Tensor | None = None
        self.idx_mapping_np: np.ndarray | None = None
        self.gen_np: np.ndarray | None = None
        self.need_sampled_mask: np.ndarray | None = None
        self.num_rows = 0

    def write(
        self,
        confidences: torch.Tensor,
        idx_mapping_np: np.ndarray,
        gen_np: np.ndarray,
        need_sampled_mask: np.ndarray,
    ) -> None:
        """Park `confidences` for this slot and start copying them to the host.

        Runs on the caller's current stream; the event then covers the copy so a
        consumer only has to synchronize it.
        """
        self.num_rows = confidences.shape[0]
        self.confidences = confidences
        self.idx_mapping_np = np.array(idx_mapping_np, copy=True)
        self.gen_np = np.array(gen_np, copy=True)
        self.need_sampled_mask = np.array(need_sampled_mask, copy=True)
        self.host[: self.num_rows].copy_(confidences, non_blocking=True)
        if self.event is not None:
            self.event.record()

    def collect(self, req_idx_gen_np: np.ndarray) -> PendingConfidences | None:
        """Build the payload, dropping rows freed or unsampled since `write`."""
        assert self.confidences is not None
        assert self.idx_mapping_np is not None
        assert self.gen_np is not None
        assert self.need_sampled_mask is not None
        num_rows = self.num_rows
        idx_mapping_np = self.idx_mapping_np[:num_rows]
        exclude_mask = (req_idx_gen_np[idx_mapping_np] != self.gen_np) | (
            ~self.need_sampled_mask
        )
        confidences = self.confidences[:num_rows]
        confidences_np = self.host_np[:num_rows]
        if exclude_mask.any():
            keep = ~exclude_mask
            if not keep.any():
                return None
            keep_t = torch.as_tensor(keep, device=self.device)
            confidences = confidences[keep_t]
            confidences_np = confidences_np[keep]
            idx_mapping_np = idx_mapping_np[keep]
        return PendingConfidences(
            confidences=confidences,
            confidences_np=np.array(confidences_np, copy=True),
            idx_mapping_np=idx_mapping_np,
        )


class AdaptiveConfidenceRing:
    """Bounded ring of confidence slots sized to the PP in-flight depth.

    The slot written when step T proposes is read when step T+pp_size verifies,
    and pushes happen at most once per step, so a slot is reused only after the
    step it held has been consumed. That keeps every in-flight payload (and its
    pinned host block) alive for as long as it can be read.
    """

    def __init__(self, depth: int, max_num_reqs: int, num_spec: int, device):
        assert depth > 0
        self._slots = [
            ConfidenceSlot(max_num_reqs, num_spec, device) for _ in range(depth)
        ]
        self._next = 0

    def push(
        self,
        confidences: torch.Tensor,
        idx_mapping_np: np.ndarray,
        gen_np: np.ndarray,
        need_sampled_mask: np.ndarray,
    ) -> ConfidenceSlot:
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        slot.write(confidences, idx_mapping_np, gen_np, need_sampled_mask)
        return slot

    def collect(
        self, slot: ConfidenceSlot | None, req_idx_gen_np: np.ndarray
    ) -> PendingConfidences | None:
        """Wait for the slot's copy and return its payload.

        The wait is on data pp_size steps old, so in steady state it returns
        immediately.
        """
        if slot is None:
            return None
        if slot.event is not None:
            with gpu_sync_allowed():
                slot.event.synchronize()
        return slot.collect(req_idx_gen_np)


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


def _alloc_combined(num_reqs: int, device: torch.device) -> torch.Tensor:
    """Allocate the (2, N) int32 buffer broadcast alongside sampled tokens.

    The inner dim is padded to a multiple of 4 so that both `unbind` views
    stay 16-byte aligned for any `num_reqs`: triton specializes on pointer
    alignment, and a misaligned `num_rejected` would JIT-compile a second
    `_post_update_kernel` variant at serving time. The padding is broadcast
    but never read. Sender and receiver must both use this allocation.
    """
    return torch.empty(2, -(-num_reqs // 4) * 4, dtype=torch.int32, device=device)


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

        # Warmup steps run the pipeline with synthetic batches whose outputs are
        # discarded; the sampled-token broadcast is disabled there so its
        # side-stream NCCL ops cannot overlap the next step's activation p2p.
        self.disabled = False

        # Adaptive verification: the confidence payload rides the draft relay so
        # that every rank prices the generation this step actually verifies.
        # Built only once the draft model confirms it can publish confidences.
        self.adaptive_confidences = False
        self._conf_ring: AdaptiveConfidenceRing | None = None
        # The last rank has no deferred queue; it replays its own payload
        # through a self-ring of the same depth.
        self._self_confidences: deque[ConfidenceSlot | None] = deque()

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled

    def set_adaptive_confidences(self, enabled: bool) -> None:
        """Attach the confidence payload to the draft relay.

        The flag is agreed by every rank at startup, so the extra broadcast is
        posted on exactly the steps the receiver posts it.
        """
        if enabled and self._conf_ring is None:
            assert self.num_speculative_steps > 0
            depth = get_pp_group().world_size
            self._conf_ring = AdaptiveConfidenceRing(
                depth, self.max_num_reqs, self.num_speculative_steps, self.device
            )
            self._self_confidences = deque([None] * depth)
        self.adaptive_confidences = enabled

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens

        assert self._conf_ring is not None or slot.confidences is None
        confidences = None
        if slot.confidences is not None:
            assert self._conf_ring is not None
            confidences = self._conf_ring.collect(slot.confidences, self.req_idx_gen_np)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            broadcast_drafts=slot.draft_tokens,
            confidences=confidences,
        )

    def pop_self_confidences(self) -> PendingConfidences | None:
        """Consume the last rank's own payload for this in-flight step.

        Mirrors `get_prev_sampled_outputs` so the last rank prices the same
        generation as its peers instead of its freshly proposed one.
        """
        if not self.is_last_rank or self._conf_ring is None:
            return None
        slot = self._self_confidences.popleft()
        self._self_confidences.append(None)
        return self._conf_ring.collect(slot, self.req_idx_gen_np)

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        if self.disabled:
            return False
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = _alloc_combined(num_reqs, self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            confidences = None
            if self.adaptive_confidences and draft_tokens is not None:
                assert self._conf_ring is not None
                confidences_tensor = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.float32,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    confidences_tensor, src=self.last_rank, group=self.broadcast_group
                )
                confidences_tensor.record_stream(self.main_stream)
                # Enqueued before the event below, so waiting on it also covers
                # the host copy the budget will read.
                confidences = self._conf_ring.push(
                    confidences_tensor,
                    input_batch.idx_mapping_np,
                    gen_at_receive_np,
                    need_sampled_mask,
                )

            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
            confidences,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if self.disabled:
            return
        mask = compute_need_sampled_mask(input_batch)
        if mask is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
        assert num_sampled.dtype == torch.int32
        assert num_rejected.dtype == torch.int32

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = _alloc_combined(num_sampled.shape[0], self.device)
            combined[0, : num_sampled.shape[0]] = num_sampled
            combined[1, : num_sampled.shape[0]] = num_rejected
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)

    def broadcast_drafts(
        self,
        draft_token_table: torch.Tensor,
        input_batch: InputBatch,
        confidences: torch.Tensor | None = None,
    ) -> None:
        """Broadcast the speculator's freshly proposed draft tokens.

        Runs after propose() on the last rank; the send is stream-ordered
        after broadcast()'s sends, matching receive()'s enqueue order. The
        payload is gathered from the runner's draft table (just updated from
        propose()'s output) into a fresh compact tensor: the speculator
        overwrites its own persistent buffer on the next step, possibly
        before this async send completes.

        `confidences` ([num_reqs, num_speculative_steps], batch order) scores
        that same block. It is broadcast as a fourth op under the same gate, so
        the pp_broadcast FIFO stays aligned, and the last rank parks its own copy
        in the self-ring instead of reading it fresh next step.
        """
        assert self.is_last_rank
        if self.disabled or self.max_sample_len == 1:
            return
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return
        assert draft_token_table.dtype == torch.int64
        assert draft_token_table.shape[1] == self.max_sample_len - 1
        if self.adaptive_confidences:
            # Fail closed: a step that broadcasts drafts must carry their
            # confidences, or the ranks would price different generations.
            assert confidences is not None, (
                "adaptive verification needs the confidences of the draft block"
            )
            assert self._conf_ring is not None
            assert confidences.shape == (
                input_batch.num_reqs,
                self.num_speculative_steps,
            )

        # Gather on the main stream so the payload is ordered after this
        # step's table update and before any later one.
        drafts = draft_token_table[input_batch.idx_mapping]
        assert drafts.shape[0] == input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                drafts, src=self.last_rank, group=self.broadcast_group
            )
            drafts.record_stream(self.broadcast_stream)
            if self.adaptive_confidences:
                assert confidences is not None
                torch.distributed.broadcast(
                    confidences, src=self.last_rank, group=self.broadcast_group
                )
                confidences.record_stream(self.broadcast_stream)
                gen_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
                self._self_confidences[-1] = self._conf_ring.push(
                    confidences,
                    input_batch.idx_mapping_np,
                    gen_np,
                    need_sampled_mask,
                )
