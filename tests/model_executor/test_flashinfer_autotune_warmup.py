# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-PP-stage FlashInfer autotune grouping (upstream #57197).

With PP > 1 each stage runs different layers and may profile different ops,
so tuning must happen per TP group with a per-group cache file; the world
group would either mix incompatible tactic timings or hang on ranks that
never reach the same op. These tests simulate a pp*tp rank mesh with fake
collective groups and assert the group wiring, cache isolation, and PP=1
back-compatibility.
"""

import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup import kernel_warmup as warmup
from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune

pytestmark = pytest.mark.cpu_test


def _make_runner(run) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        get_model=Mock(return_value=SimpleNamespace(modules=Mock(return_value=[]))),
        _dummy_run=lambda **kwargs: run.dummy_runs(),
    )


class _AutotuneGroup:
    def __init__(self, run, ranks):
        self.run = run
        self.ranks = tuple(ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = self.ranks.index(run.rank)
        self.cpu_group = self

    def record(self, operation):
        self.run.collectives[self.ranks][self.run.rank].append(operation)

    def broadcast_object(self, obj, src=0):
        assert src == 0
        self.record(("broadcast", src))
        if self.rank_in_group == src:
            self.run.broadcasts[self.ranks] = obj
        return self.run.broadcasts[self.ranks]

    def barrier(self):
        self.record(("barrier",))


class _AutotuneTuner:
    def __init__(self, run):
        self.run = run
        self.cache = {}
        self.loaded = None

    def load_configs(self, path):
        self.loaded = json.loads(Path(path).read_text())
        self.cache.update(self.loaded)

    def save_configs(self, path):
        self.run.saves.append((self.run.rank, Path(path), dict(self.cache)))
        Path(path).write_text(json.dumps(self.cache))

    def profile(self, operation):
        if operation in self.cache:
            return
        group = self.run.tuning_group
        self.run.profile_groups[self.run.rank].append(
            None if group is None else group.ranks
        )
        for tactic in range(2):
            if group is not None:
                group.record(("all_reduce", operation, tactic))
        self.cache[operation] = self.run.rank // self.run.tp


class _AutotuneRun:
    def __init__(self, pp, tp):
        self.pp, self.tp = pp, tp
        self.rank = 0
        self.tuning_group = None
        self.collectives: dict[tuple[int, ...], dict[int, list[tuple[Any, ...]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.broadcasts = {}
        self.tuners = {}
        self.saves = []
        self.profile_groups = defaultdict(list)

    def world(self):
        return _AutotuneGroup(self, range(self.pp * self.tp))

    def tensor_group(self):
        start = self.rank // self.tp * self.tp
        return _AutotuneGroup(self, range(start, start + self.tp))

    def pipeline_group(self):
        return SimpleNamespace(world_size=self.pp)

    def set_group(self, group):
        self.tuning_group = group

    def dummy_runs(self):
        # Stand-in for the dummy forward: stage 0 owns an op the other
        # stages never run, mirroring heterogeneous PP stages.
        self.tuners[self.rank].profile("shared_gemm")
        if self.pp > 1 and self.rank // self.tp == 0:
            self.tuners[self.rank].profile("pp0_extra_gemm")

    def execute(self):
        for rank in range(self.pp * self.tp):
            self.rank = rank
            self.tuners[rank] = _AutotuneTuner(self)
            flashinfer_autotune(_make_runner(self))
        return self

    def assert_collectives_match(self):
        for ranks, traces in self.collectives.items():
            assert set(traces) == set(ranks)
            expected = traces[ranks[0]]
            assert all(trace == expected for trace in traces.values()), dict(traces)


@pytest.fixture
def autotune_run(monkeypatch, tmp_path):
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    def make_run(*, pp=2, tp=4):
        run = _AutotuneRun(pp, tp)
        autotuner = ModuleType("flashinfer.autotuner")
        monkeypatch.setattr(
            autotuner,
            "AutoTuner",
            SimpleNamespace(get=lambda: run.tuners[run.rank]),
            raising=False,
        )
        monkeypatch.setattr(
            autotuner, "set_autotune_process_group", run.set_group, raising=False
        )
        monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        monkeypatch.setattr(parallel_state, "get_world_group", run.world)
        monkeypatch.setattr(parallel_state, "get_tp_group", run.tensor_group)
        monkeypatch.setattr(parallel_state, "get_pp_group", run.pipeline_group)
        monkeypatch.setattr(fi_utils, "autotune", lambda **kwargs: nullcontext())
        monkeypatch.setattr(
            warmup,
            "resolve_flashinfer_autotune_file",
            lambda runner: tmp_path / "autotune_configs.json",
        )
        monkeypatch.setattr(
            warmup, "_flashinfer_autotune_skip_ops", lambda runner: None
        )
        return run

    return make_run


@pytest.mark.parametrize("tp", [1, 4])
def test_heterogeneous_pp_stages_have_compatible_collectives(autotune_run, tp):
    run = autotune_run(tp=tp).execute()
    run.assert_collectives_match()
    for rank, groups in run.profile_groups.items():
        expected = tuple(range(rank // tp * tp, (rank // tp + 1) * tp))
        assert groups and all(
            group == (expected if tp > 1 else None) for group in groups
        )


def test_pp_stage_cache_roundtrip_isolated_and_asymmetric_hits_safe(
    autotune_run, tmp_path
):
    legacy = tmp_path / "autotune_configs.json"
    legacy.write_text('{"legacy_world_cache": 99}')
    cold = autotune_run().execute()
    assert [rank for rank, _, _ in cold.saves] == [0, 4]
    paths = [path for _, path, _ in cold.saves]
    assert len(set(paths)) == 2 and legacy not in paths
    assert json.loads(legacy.read_text()) == {"legacy_world_cache": 99}
    assert cold.saves[0][2] == {"shared_gemm": 0, "pp0_extra_gemm": 0}
    assert cold.saves[1][2] == {"shared_gemm": 1}
    cold.assert_collectives_match()
    warm = autotune_run().execute()
    warm.assert_collectives_match()
    assert not warm.profile_groups
    for rank, tuner in warm.tuners.items():
        assert tuner.loaded == cold.saves[rank // 4][2]
    paths[1].unlink()
    mixed = autotune_run().execute()
    mixed.assert_collectives_match()
    assert set(mixed.profile_groups) == {4, 5, 6, 7}
    assert all(mixed.tuners[rank].loaded is not None for rank in range(4))
    assert all(mixed.tuners[rank].loaded is None for rank in range(4, 8))


def test_pp1_retains_world_synchronization_and_existing_cache_name(autotune_run):
    run = autotune_run(pp=1, tp=4).execute()
    run.assert_collectives_match()
    assert [(rank, path.name) for rank, path, _ in run.saves] == [
        (0, "autotune_configs.json")
    ]
    assert all(groups == [(0, 1, 2, 3)] for groups in run.profile_groups.values())
