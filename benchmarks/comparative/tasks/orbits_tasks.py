"""
Orbital element conversion benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask

R_EARTH = 6378137.0  # meters


class KeplerianToCartesianTask(BenchmarkTask):
    """Keplerian orbital elements to Cartesian state conversion benchmark."""

    @property
    def name(self) -> str:
        return "orbits.keplerian_to_cartesian"

    @property
    def module(self) -> str:
        return "orbits"

    @property
    def description(self) -> str:
        return "Convert Keplerian elements [a,e,i,raan,argp,M] to Cartesian state [x,y,z,vx,vy,vz]"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        elements = []
        for _ in range(50):
            a = R_EARTH + rng.uniform(200e3, 36000e3)  # LEO to GEO
            e = rng.uniform(0.001, 0.5)
            i = rng.uniform(0.0, 180.0)  # degrees
            raan = rng.uniform(0.0, 360.0)
            argp = rng.uniform(0.0, 360.0)
            M = rng.uniform(0.0, 360.0)
            elements.append([a, e, i, raan, argp, M])
        return {"elements": elements}


class CartesianToKeplerianTask(BenchmarkTask):
    """Cartesian state to Keplerian orbital elements conversion benchmark."""

    @property
    def name(self) -> str:
        return "orbits.cartesian_to_keplerian"

    @property
    def module(self) -> str:
        return "orbits"

    @property
    def description(self) -> str:
        return "Convert Cartesian state [x,y,z,vx,vy,vz] to Keplerian elements [a,e,i,raan,argp,M]"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare orbital elements with angle normalization.

        Keplerian elements [a, e, i, raan, argp, M] have angles in degrees.
        Different libraries may return angles in different ranges (e.g., -180..180 vs 0..360).
        Normalize angular differences before computing errors.
        """
        n = min(len(results_a), len(results_b))
        if n == 0:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        abs_errors = []
        rel_errors = []

        for i in range(n):
            oe_a = results_a[i]
            oe_b = results_b[i]
            if not isinstance(oe_a, (list, tuple)) or len(oe_a) < 6:
                continue

            for j in range(6):
                a_val = oe_a[j]
                b_val = oe_b[j]

                if j >= 2:
                    # Angular elements (i, raan, argp, M) — normalize difference
                    diff = (a_val - b_val) % 360.0
                    if diff > 180.0:
                        diff -= 360.0
                    abs_err = abs(diff)
                else:
                    # a, e — direct comparison
                    abs_err = abs(a_val - b_val)

                abs_errors.append(abs_err)
                denom = max(abs(a_val), 1e-30)
                rel_errors.append(abs_err / denom)

        if not abs_errors:
            return AccuracyComparison(
                task_name=self.name,
                reference_language=language_a,
                comparison_language=language_b,
                max_abs_error=float("nan"),
                max_rel_error=float("nan"),
                rms_error=float("nan"),
            )

        rms = math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
        return AccuracyComparison(
            task_name=self.name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=max(abs_errors),
            max_rel_error=max(rel_errors),
            rms_error=rms,
        )

    def generate_params(self, seed: int) -> dict:
        # Generate Cartesian states by converting from known Keplerian elements

        rng = random.Random(seed)
        GM = 3.986004418e14  # m^3/s^2
        states = []
        for _ in range(50):
            a = R_EARTH + rng.uniform(200e3, 36000e3)
            e = rng.uniform(0.001, 0.5)
            i = math.radians(rng.uniform(0.0, 180.0))
            raan = math.radians(rng.uniform(0.0, 360.0))
            argp = math.radians(rng.uniform(0.0, 360.0))
            nu = math.radians(rng.uniform(0.0, 360.0))

            # Perifocal coordinates
            p = a * (1.0 - e * e)
            r = p / (1.0 + e * math.cos(nu))
            r_pqw = [r * math.cos(nu), r * math.sin(nu), 0.0]
            v_mag = math.sqrt(GM / p)
            v_pqw = [
                -v_mag * math.sin(nu),
                v_mag * (e + math.cos(nu)),
                0.0,
            ]

            # Rotation to ECI
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
            states.append([x, y, z, vx, vy, vz])

        return {"states": states}
