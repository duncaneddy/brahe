"""Shared helpers for Python profile workloads.

Mirrors profiles/rust/src/common.rs: provider initialization, a fixed initial
state, and the `run_until_elapsed` driver. Imported by every script under
profiles/python/.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

import brahe as bh

# Pinned ISS-like TLE used by SGP4 profile workloads. Mirrors the Rust common
# module — the epoch in the TLE is irrelevant for profiling but pinning it
# keeps results comparable across runs.
DEFAULT_ISS_TLE_LINE1 = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
)
DEFAULT_ISS_TLE_LINE2 = (
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
)


def setup_providers() -> None:
    """Install global EOP and space-weather providers.

    Idempotent — brahe's providers are write-once.
    """
    bh.initialize_eop()
    bh.initialize_sw()


def default_leo_state() -> tuple[bh.Epoch, np.ndarray]:
    """Fixed 500 km sun-sync LEO at 2024-01-01 UTC. Matches the Rust harness."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    return epoch, state


def duration_from_env(default: float = 10.0) -> float:
    """Read PROFILE_DURATION_S, falling back to `default`."""
    raw = os.environ.get("PROFILE_DURATION_S")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def run_until_elapsed(duration_s: float, body: Callable[[], None]) -> int:
    """Call `body()` back-to-back until `duration_s` seconds have elapsed.

    Returns the iteration count. Does not sleep — the profiler must see the
    workload at 100% duty cycle.
    """
    deadline = time.perf_counter() + duration_s
    iters = 0
    while time.perf_counter() < deadline:
        body()
        iters += 1
    return iters
