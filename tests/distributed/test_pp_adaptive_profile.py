# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup agreement for the adaptive verification cost profile over PP.

The verify cost of a verification step is the sum of the per-stage costs, and
every rank rebuilds the same cost tables from that sum, so the startup
collectives have to hand every stage the same data and fail symmetrically when
they cannot. These tests run the real ``all_gather_object`` on gloo: each
process owns one stage's manager and drives the startup path with synthetic
profile samples, so no GPU, cudagraph or draft model is involved.
"""

import multiprocessing as mp
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm.utils.network_utils import get_file_store_init_method
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)

_CAPTURE_SIZES = [1, 2, 3, 4, 5, 10]
_JOIN_TIMEOUT_S = 120


def _make_stage_manager(rank: int, world_size: int) -> AdaptiveVerificationManager:
    """A manager carrying only what the startup profile path reads."""
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.pp_world_size = world_size
    manager.pp_rank = rank
    manager.req_states = SimpleNamespace(
        max_num_reqs=16,
        max_num_batched_tokens=2048,
        max_model_len=524288,
    )
    manager.num_speculative_steps = 4
    manager.num_bonus_tokens = 1
    manager._cudagraph_limit = 0
    # The stage group is the whole gloo world here.
    manager._pp_cpu_group = lambda: None
    return manager


def _stage_samples(
    manager: AdaptiveVerificationManager,
    capture_sizes: list[int],
    rank: int,
    world_size: int,
    drop_last: bool = False,
) -> list[StepTimingSample]:
    batches = list(manager.batches_to_profile(capture_sizes))
    if drop_last:
        batches = batches[:-1]
    return [
        StepTimingSample(
            forward_ms=float(rank + 1) * batch["num_tokens"],
            drafter_ms=4.0 if rank == world_size - 1 else 0.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_reqs"],
            full_cudagraph=True,
        )
        for batch in batches
    ]


def _run_stage(rank: int, world_size: int, init_method: str, case: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        manager = _make_stage_manager(rank, world_size)
        capture_sizes = _CAPTURE_SIZES
        if case == "capture_mismatch" and rank != 0:
            capture_sizes = [5, 10]
        if case == "capture_mismatch":
            with pytest.raises(RuntimeError, match="same shapes"):
                manager.check_pp_profile_agreement(capture_sizes)
            return

        manager.check_pp_profile_agreement(capture_sizes)
        samples = _stage_samples(
            manager,
            capture_sizes,
            rank,
            world_size,
            drop_last=case == "samples_mismatch" and rank != 0,
        )
        curves: dict[str, list[tuple[int, float]]] = {}
        manager.set_cost_curves = lambda draft, verify: curves.update(
            draft=draft, verify=verify
        )
        if case == "samples_mismatch":
            # Gathering first is what lets the peers fail instead of hanging.
            with pytest.raises(RuntimeError, match="same shapes"):
                manager.set_initial_cost_curves(samples)
            return

        manager.set_initial_cost_curves(samples)

        # Every stage prices the same total: the sum of the per-stage costs at
        # the same token count, with only the last stage's drafter cost.
        batches = list(manager.batches_to_profile(capture_sizes))
        cost_sum = sum(range(1, world_size + 1))
        num_tokens_seen = sorted({batch["num_tokens"] for batch in batches})
        assert curves["verify"] == [
            (num_tokens, float(cost_sum * num_tokens)) for num_tokens in num_tokens_seen
        ]
        num_reqs_seen = sorted({batch["num_reqs"] for batch in batches})
        assert curves["draft"] == [(num_reqs, 4.0) for num_reqs in num_reqs_seen]
    finally:
        dist.destroy_process_group()


def _run_stages(world_size: int, case: str) -> None:
    init_method = get_file_store_init_method()
    # Spawn: the parent has torch threads running, so fork is unsafe here.
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for rank in range(world_size):
        process = context.Process(
            target=_run_stage, args=(rank, world_size, init_method, case)
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join(timeout=_JOIN_TIMEOUT_S)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0, (
            f"profile startup rank failed with exit code {process.exitcode}"
        )


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("case", ["agreement", "capture_mismatch", "samples_mismatch"])
def test_pp_profile_startup_agrees_across_stages(world_size: int, case: str):
    _run_stages(world_size, case)
