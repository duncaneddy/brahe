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


# Shared LEO orbit used for the high-fidelity numerical propagation comparison.
# Fixed parameters across implementations so brahe and Orekit see identical inputs.
_NUMERICAL_LEO = {
    "jd": 2460310.5,  # 2024-01-01T00:00:00 UTC
    "elements_deg": [R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0],
    "step_size": 30.0,  # RK4 step size
    "n_steps": 180,  # ~90 minutes of LEO propagation (~1 orbit)
    # Spacecraft mass and surfaces, used by drag and SRP.
    "mass": 1000.0,  # kg
    "drag_area": 10.0,  # m^2
    "cd": 2.2,
    "srp_area": 10.0,  # m^2
    "cr": 1.3,
}


def _numerical_leo_params() -> dict:
    """Common parameter dict for the RK4 force-model benchmarks."""
    p = dict(_NUMERICAL_LEO)
    # Layout matches brahe's DefaultParameterLayout: [mass, drag_area, Cd, srp_area, Cr]
    p["params"] = [p["mass"], p["drag_area"], p["cd"], p["srp_area"], p["cr"]]
    return p


class NumericalRk4Grav5x5Task(BenchmarkTask):
    """RK4 + 5x5 spherical-harmonic gravity over 1 LEO revolution."""

    @property
    def name(self) -> str:
        return "propagation.numerical_rk4_grav5x5"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return "Numerical RK4 propagation with 5x5 spherical-harmonic gravity over ~1 LEO orbit"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    @property
    def timeout(self) -> int:
        return 600

    def generate_params(self, seed: int) -> dict:
        p = _numerical_leo_params()
        p["gravity_degree"] = 5
        p["gravity_order"] = 5
        return p


class NumericalRk4Grav20x20SunMoonTask(BenchmarkTask):
    """RK4 + 20x20 gravity + Sun/Moon third-body over 1 LEO revolution."""

    @property
    def name(self) -> str:
        return "propagation.numerical_rk4_grav20x20_sun_moon"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return (
            "Numerical RK4 propagation with 20x20 gravity and Sun/Moon third-body "
            "over ~1 LEO orbit"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    @property
    def timeout(self) -> int:
        return 600

    def generate_params(self, seed: int) -> dict:
        p = _numerical_leo_params()
        p["gravity_degree"] = 20
        p["gravity_order"] = 20
        p["third_body_sun"] = True
        p["third_body_moon"] = True
        return p


class NumericalRk4Grav80x80FullTask(BenchmarkTask):
    """RK4 + 80x80 gravity + Sun/Moon + drag + SRP over 1 LEO revolution."""

    @property
    def name(self) -> str:
        return "propagation.numerical_rk4_grav80x80_full"

    @property
    def module(self) -> str:
        return "propagation"

    @property
    def description(self) -> str:
        return (
            "Numerical RK4 propagation with 80x80 gravity, Sun/Moon third-body, "
            "NRLMSISE-00 drag, and SRP over ~1 LEO orbit"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    @property
    def timeout(self) -> int:
        return 1200

    def generate_params(self, seed: int) -> dict:
        p = _numerical_leo_params()
        p["gravity_degree"] = 80
        p["gravity_order"] = 80
        p["third_body_sun"] = True
        p["third_body_moon"] = True
        p["drag"] = True
        p["srp"] = True
        return p
