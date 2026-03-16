"""
Attitude conversion benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask


def _generate_random_quaternions(seed: int, count: int = 50) -> list[list[float]]:
    """Generate random unit quaternions [w, x, y, z]."""
    rng = random.Random(seed)
    quats = []
    for _ in range(count):
        # Random Gaussian components, then normalize
        w = rng.gauss(0, 1)
        x = rng.gauss(0, 1)
        y = rng.gauss(0, 1)
        z = rng.gauss(0, 1)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        quats.append([w / norm, x / norm, y / norm, z / norm])
    return quats


def _quaternion_to_matrix(q: list[float]) -> list[list[float]]:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix (row-major)."""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _compare_quaternions(
    results_a: list, results_b: list, task_name: str, language_a: str, language_b: str
) -> AccuracyComparison:
    """Compare quaternion results accounting for sign ambiguity (q == -q)."""
    n = min(len(results_a), len(results_b))
    if n == 0:
        return AccuracyComparison(
            task_name=task_name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=float("nan"),
            max_rel_error=float("nan"),
            rms_error=float("nan"),
        )

    abs_errors = []
    rel_errors = []

    for i in range(n):
        qa = results_a[i]
        qb = results_b[i]
        if not isinstance(qa, (list, tuple)) or len(qa) < 4:
            continue

        # Compute errors for both q and -q, take minimum
        for j in range(4):
            err_pos = abs(qa[j] - qb[j])
            err_neg = abs(qa[j] + qb[j])
            abs_err = min(err_pos, err_neg)
            abs_errors.append(abs_err)
            denom = max(abs(qa[j]), 1e-30)
            rel_errors.append(abs_err / denom)

    if not abs_errors:
        return AccuracyComparison(
            task_name=task_name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=float("nan"),
            max_rel_error=float("nan"),
            rms_error=float("nan"),
        )

    rms = math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
    return AccuracyComparison(
        task_name=task_name,
        reference_language=language_a,
        comparison_language=language_b,
        max_abs_error=max(abs_errors),
        max_rel_error=max(rel_errors),
        rms_error=rms,
    )


class QuaternionToRotationMatrixTask(BenchmarkTask):
    """Benchmark quaternion to rotation matrix conversion."""

    @property
    def name(self) -> str:
        return "attitude.quaternion_to_rotation_matrix"

    @property
    def module(self) -> str:
        return "attitude"

    @property
    def description(self) -> str:
        return "Convert quaternion [w,x,y,z] to 3x3 rotation matrix"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"quaternions": _generate_random_quaternions(seed)}


class RotationMatrixToQuaternionTask(BenchmarkTask):
    """Benchmark rotation matrix to quaternion conversion."""

    @property
    def name(self) -> str:
        return "attitude.rotation_matrix_to_quaternion"

    @property
    def module(self) -> str:
        return "attitude"

    @property
    def description(self) -> str:
        return "Convert 3x3 rotation matrix to quaternion [w,x,y,z]"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        quats = _generate_random_quaternions(seed)
        matrices = [_quaternion_to_matrix(q) for q in quats]
        return {"matrices": matrices}

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        return _compare_quaternions(
            results_a, results_b, self.name, language_a, language_b
        )


class QuaternionToEulerAngleTask(BenchmarkTask):
    """Benchmark quaternion to Euler angle (ZYX) conversion."""

    @property
    def name(self) -> str:
        return "attitude.quaternion_to_euler_angle"

    @property
    def module(self) -> str:
        return "attitude"

    @property
    def description(self) -> str:
        return "Convert quaternion to Euler angles (ZYX order) [phi, theta, psi] in radians"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        return {"quaternions": _generate_random_quaternions(seed)}

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare Euler angles with angle wrapping normalization."""
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
            ea = results_a[i]
            eb = results_b[i]
            if not isinstance(ea, (list, tuple)) or len(ea) < 3:
                continue

            for j in range(3):
                # Normalize angular difference to [-pi, pi]
                diff = (ea[j] - eb[j]) % (2 * math.pi)
                if diff > math.pi:
                    diff -= 2 * math.pi
                abs_err = abs(diff)
                abs_errors.append(abs_err)
                denom = max(abs(ea[j]), 1e-30)
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


class EulerAngleToQuaternionTask(BenchmarkTask):
    """Benchmark Euler angle (ZYX) to quaternion conversion."""

    @property
    def name(self) -> str:
        return "attitude.euler_angle_to_quaternion"

    @property
    def module(self) -> str:
        return "attitude"

    @property
    def description(self) -> str:
        return "Convert Euler angles (ZYX order) [phi, theta, psi] in radians to quaternion"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        angles = []
        for _ in range(50):
            phi = rng.uniform(-math.pi, math.pi)
            theta = rng.uniform(-math.pi / 2, math.pi / 2)
            psi = rng.uniform(-math.pi, math.pi)
            angles.append([phi, theta, psi])
        return {"angles": angles}

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        return _compare_quaternions(
            results_a, results_b, self.name, language_a, language_b
        )
