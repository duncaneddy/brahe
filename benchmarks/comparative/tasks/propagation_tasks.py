"""
Propagation benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.tasks.base import BenchmarkTask

R_EARTH = 6378137.0  # meters
GM = 3.986004418e14  # m^3/s^2


class KeplerianSingleTask(BenchmarkTask):
    """Benchmark Keplerian propagation of multiple orbits to a single future epoch."""

    @property
    def name(self) -> str:
        return "propagation.keplerian_single"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return "Propagate 20 orbits each to a single future epoch using two-body analytical"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        cases = []
        # Base epoch: 2024-01-01T00:00:00 UTC = JD 2460310.5
        base_jd = 2460310.5

        for _ in range(20):
            jd = base_jd + rng.uniform(0.0, 365.0)
            a = R_EARTH + rng.uniform(200e3, 36000e3)
            e = rng.uniform(0.001, 0.3)
            i = rng.uniform(0.0, 180.0)  # degrees
            raan = rng.uniform(0.0, 360.0)
            argp = rng.uniform(0.0, 360.0)
            M = rng.uniform(0.0, 360.0)
            dt = rng.uniform(3600.0, 86400.0)  # 1h to 1 day

            cases.append(
                {
                    "jd": jd,
                    "elements": [a, e, i, raan, argp, M],
                    "dt": dt,
                }
            )
        return {"cases": cases}


class KeplerianTrajectoryTask(BenchmarkTask):
    """Benchmark Keplerian propagation of 1 orbit over ~90 time steps."""

    @property
    def name(self) -> str:
        return "propagation.keplerian_trajectory"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return (
            "Propagate 1 LEO orbit over ~90 steps (1 orbital period at 60s intervals)"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        # Fixed LEO orbit
        a = R_EARTH + 500e3
        e = 0.01
        i = 97.8  # degrees
        raan = 15.0
        argp = 30.0
        M = 45.0
        step_size = 60.0  # seconds

        # Compute orbital period
        period = 2.0 * math.pi * math.sqrt(a**3 / GM)
        n_steps = int(period / step_size)

        return {
            "jd": 2460310.5,  # 2024-01-01T00:00:00 UTC
            "elements": [a, e, i, raan, argp, M],
            "step_size": step_size,
            "n_steps": n_steps,
        }


# ISS TLE for SGP4 benchmarks (well-known, stable orbit)
ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009"
ISS_TLE_LINE2 = "2 25544  51.6400 208.9163 0006703 307.5225  52.5371 15.49560812484108"


class Sgp4SingleTask(BenchmarkTask):
    """Benchmark SGP4 propagation of ISS TLE to 50 future epochs."""

    @property
    def name(self) -> str:
        return "propagation.sgp4_single"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return "Propagate ISS TLE to 50 future time offsets using SGP4"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        offsets = sorted([rng.uniform(0.0, 86400.0) for _ in range(50)])
        return {
            "line1": ISS_TLE_LINE1,
            "line2": ISS_TLE_LINE2,
            "time_offsets_seconds": offsets,
        }


class Sgp4TrajectoryTask(BenchmarkTask):
    """Benchmark SGP4 propagation of ISS TLE over 1 day at 60s steps."""

    @property
    def name(self) -> str:
        return "propagation.sgp4_trajectory"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return "Propagate ISS TLE over 1 day at 60s intervals using SGP4 (1440 steps)"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {
            "line1": ISS_TLE_LINE1,
            "line2": ISS_TLE_LINE2,
            "step_size": 60.0,
            "n_steps": 1440,
        }


class NumericalTwobodyTask(BenchmarkTask):
    """Benchmark numerical two-body propagation over 1 orbital period."""

    @property
    def name(self) -> str:
        return "propagation.numerical_twobody"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return "Numerical integration of two-body problem over 1 orbital period at 60s steps"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        # Fixed LEO orbit — same as keplerian_trajectory for comparison
        a = R_EARTH + 500e3
        e = 0.01
        i = 97.8  # degrees
        raan = 15.0
        argp = 30.0
        M = 45.0
        step_size = 60.0

        period = 2.0 * math.pi * math.sqrt(a**3 / GM)
        n_steps = int(period / step_size)

        return {
            "jd": 2460310.5,
            "elements": [a, e, i, raan, argp, M],
            "step_size": step_size,
            "n_steps": n_steps,
        }
