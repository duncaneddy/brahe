"""
Frame transformation benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask, _flatten

R_EARTH = 6378137.0  # meters
GM = 3.986004418e14  # m^3/s^2


def _generate_frame_params(seed: int) -> dict:
    """Generate 50 test cases with JD epochs and 6D state vectors."""
    rng = random.Random(seed)
    cases = []

    for idx in range(50):
        # Spread epochs over 2024 (JD 2460310.5 = 2024-01-01T00:00:00 UTC)
        jd = 2460310.5 + idx * 7.0 + rng.uniform(0.0, 7.0)  # ~weekly over a year

        # Generate random Keplerian elements and convert to Cartesian
        a = R_EARTH + rng.uniform(200e3, 36000e3)
        e = rng.uniform(0.001, 0.3)
        i = math.radians(rng.uniform(0.0, 180.0))
        raan = math.radians(rng.uniform(0.0, 360.0))
        argp = math.radians(rng.uniform(0.0, 360.0))
        nu = math.radians(rng.uniform(0.0, 360.0))

        p = a * (1.0 - e * e)
        r = p / (1.0 + e * math.cos(nu))
        r_pqw = [r * math.cos(nu), r * math.sin(nu), 0.0]
        v_mag = math.sqrt(GM / p)
        v_pqw = [-v_mag * math.sin(nu), v_mag * (e + math.cos(nu)), 0.0]

        cos_raan, sin_raan = math.cos(raan), math.sin(raan)
        cos_argp, sin_argp = math.cos(argp), math.sin(argp)
        cos_i, sin_i = math.cos(i), math.sin(i)

        r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
        r12 = -(cos_raan * sin_argp + sin_raan * cos_argp * cos_i)
        r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
        r22 = -(sin_raan * sin_argp - cos_raan * cos_argp * cos_i)
        r31 = sin_argp * sin_i
        r32 = cos_argp * sin_i

        x = r11 * r_pqw[0] + r12 * r_pqw[1]
        y = r21 * r_pqw[0] + r22 * r_pqw[1]
        z = r31 * r_pqw[0] + r32 * r_pqw[1]
        vx = r11 * v_pqw[0] + r12 * v_pqw[1]
        vy = r21 * v_pqw[0] + r22 * v_pqw[1]
        vz = r31 * v_pqw[0] + r32 * v_pqw[1]

        cases.append(
            {
                "jd": jd,
                "state": [x, y, z, vx, vy, vz],
            }
        )

    return {"cases": cases}


class StateEciToEcefTask(BenchmarkTask):
    """Benchmark ECI to ECEF 6D state transformation."""

    @property
    def name(self) -> str:
        return "frames.state_eci_to_ecef"

    @property
    def module(self) -> str:
        return "frames"

    @property
    def description(self) -> str:
        return "Transform 6D state vector from ECI to ECEF at given epoch"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return _generate_frame_params(seed)

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare with wider tolerance due to EOP differences."""
        flat_a = _flatten(results_a)
        flat_b = _flatten(results_b)

        if not flat_a or not flat_b:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        n = min(len(flat_a), len(flat_b))
        abs_errors = [abs(flat_a[i] - flat_b[i]) for i in range(n)]
        rel_errors = [
            abs(flat_a[i] - flat_b[i]) / max(abs(flat_a[i]), 1e-30) for i in range(n)
        ]
        rms = math.sqrt(sum(e * e for e in abs_errors) / n)

        return AccuracyComparison(
            task_name=self.name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=max(abs_errors),
            max_rel_error=max(rel_errors),
            rms_error=rms,
        )


class StateEcefToEciTask(BenchmarkTask):
    """Benchmark ECEF to ECI 6D state transformation."""

    @property
    def name(self) -> str:
        return "frames.state_ecef_to_eci"

    @property
    def module(self) -> str:
        return "frames"

    @property
    def description(self) -> str:
        return "Transform 6D state vector from ECEF to ECI at given epoch"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return _generate_frame_params(seed)

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare with wider tolerance due to EOP differences."""
        flat_a = _flatten(results_a)
        flat_b = _flatten(results_b)

        if not flat_a or not flat_b:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        n = min(len(flat_a), len(flat_b))
        abs_errors = [abs(flat_a[i] - flat_b[i]) for i in range(n)]
        rel_errors = [
            abs(flat_a[i] - flat_b[i]) / max(abs(flat_a[i]), 1e-30) for i in range(n)
        ]
        rms = math.sqrt(sum(e * e for e in abs_errors) / n)

        return AccuracyComparison(
            task_name=self.name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=max(abs_errors),
            max_rel_error=max(rel_errors),
            rms_error=rms,
        )
