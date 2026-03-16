"""
Python (Brahe) attitude conversion benchmarks.
"""

import numpy as np

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def quaternion_to_rotation_matrix(params: dict, iterations: int) -> TaskResult:
    """Benchmark quaternion to rotation matrix conversion using brahe."""
    ensure_eop()
    quaternions = params["quaternions"]

    def run():
        results = []
        for q in quaternions:
            quat = brahe.Quaternion(q[0], q[1], q[2], q[3])
            rm = quat.to_rotation_matrix()
            mat = rm.to_matrix()
            # Flatten 3x3 to row-major list
            results.append([mat[r][c] for r in range(3) for c in range(3)])
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="attitude.quaternion_to_rotation_matrix",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def rotation_matrix_to_quaternion(params: dict, iterations: int) -> TaskResult:
    """Benchmark rotation matrix to quaternion conversion using brahe."""
    ensure_eop()
    matrices = params["matrices"]

    def run():
        results = []
        for mat in matrices:
            np_mat = np.array(mat)
            rm = brahe.RotationMatrix.from_matrix(np_mat)
            q = rm.to_quaternion()
            v = q.to_vector(True)  # scalar_first=True -> [w, x, y, z]
            results.append(v.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="attitude.rotation_matrix_to_quaternion",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def quaternion_to_euler_angle(params: dict, iterations: int) -> TaskResult:
    """Benchmark quaternion to Euler angle (ZYX) conversion using brahe."""
    ensure_eop()
    quaternions = params["quaternions"]

    def run():
        results = []
        for q in quaternions:
            quat = brahe.Quaternion(q[0], q[1], q[2], q[3])
            ea = quat.to_euler_angle(brahe.EulerAngleOrder.ZYX)
            results.append([ea.phi, ea.theta, ea.psi])
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="attitude.quaternion_to_euler_angle",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )


def euler_angle_to_quaternion(params: dict, iterations: int) -> TaskResult:
    """Benchmark Euler angle (ZYX) to quaternion conversion using brahe."""
    ensure_eop()
    angles = params["angles"]

    def run():
        results = []
        for phi, theta, psi in angles:
            ea = brahe.EulerAngle(
                brahe.EulerAngleOrder.ZYX,
                phi,
                theta,
                psi,
                brahe.AngleFormat.RADIANS,
            )
            q = ea.to_quaternion()
            v = q.to_vector(True)  # [w, x, y, z]
            results.append(v.tolist())
        return results

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="attitude.euler_angle_to_quaternion",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )
