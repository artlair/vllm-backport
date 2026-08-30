"""Lightweight host-side phase timing for the v2 GPU worker.

Enabled with VLLM_BUILDER_TRACE=1. Collects wall-clock durations for named
phases of the per-step host work (input prep, attention metadata builders,
forward launch) and logs rolling p50/p90/max per name every
VLLM_BUILDER_TRACE_INTERVAL traced steps. Default-off; record() is a no-op
cheap enough to leave in the hot path.
"""

import os
from collections import defaultdict

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_BUILDER_TRACE", "0") == "1"
_INTERVAL = int(os.environ.get("VLLM_BUILDER_TRACE_INTERVAL", "500"))

_samples: dict[str, list[float]] = defaultdict(list)
_steps = 0


def record(name: str, dur_s: float) -> None:
    if ENABLED:
        _samples[name].append(dur_s * 1e3)


def tick(tag: str = "") -> None:
    """Count one traced step; emit and reset stats every _INTERVAL steps."""
    global _steps
    if not ENABLED:
        return
    _steps += 1
    if _steps % _INTERVAL:
        return
    parts = []
    for name in sorted(_samples):
        vals = sorted(_samples[name])
        n = len(vals)
        if not n:
            continue
        parts.append(
            f"{name} p50={vals[n // 2]:.2f} p90={vals[int(n * 0.9)]:.2f} "
            f"max={vals[-1]:.2f} n={n}"
        )
    _samples.clear()
    logger.info("BTRACE %s%s", tag, " | ".join(parts))
