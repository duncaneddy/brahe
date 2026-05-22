"""Basilisk attitude conversion benchmarks.

Conventions:
- Quaternion (Euler parameter, EP): scalar-first [w, x, y, z]. Matches brahe
  and Orekit at the level of `quaternion_to_rotation_matrix` /
  `rotation_matrix_to_quaternion`; no reorder needed.
- Rotation matrix: Basilisk's EP2C and C2EP produce passive DCMs that match
  brahe and Hipparchus element-for-element.
- Euler angles: brahe exposes aerospace intrinsic ZYX (yaw-pitch-roll) — phi
  rotates about z first, psi about the body x last. Both the forward
  (quat -> Euler) and reverse (Euler -> quat) paths use Basilisk's native
  Schaub-Junkins primitives: ``EP2Euler321(q)`` and ``euler3212EP(angles)``.
  Schaub-Junkins decomposes the 3-2-1 sequence using the passive DCM, the
  same convention as brahe and Hipparchus' RotationOrder.ZYX +
  FRAME_TRANSFORM, so values match the other adapters at machine precision.
"""

import numpy as np

from Basilisk.utilities import RigidBodyKinematics as rbk

from benchmarks.comparative.implementations.basilisk.base import (
    build_task_result,
    time_iterations,
)


def quaternion_to_rotation_matrix(params: dict, iterations: int):
    """Convert quaternions [w, x, y, z] to 3x3 rotation matrices."""
    quaternions = [np.array(q, dtype=float) for q in params["quaternions"]]

    def run():
        results = []
        for q in quaternions:
            mat = rbk.EP2C(q)
            # Flatten 3x3 row-major to match brahe/Orekit output convention
            results.append([float(mat[r][c]) for r in range(3) for c in range(3)])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.quaternion_to_rotation_matrix", iterations, times, results
    )


def rotation_matrix_to_quaternion(params: dict, iterations: int):
    """Convert 3x3 rotation matrices to quaternions [w, x, y, z]."""
    matrices = [np.array(m, dtype=float) for m in params["matrices"]]

    def run():
        results = []
        for mat in matrices:
            q = rbk.C2EP(mat)
            results.append([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.rotation_matrix_to_quaternion", iterations, times, results
    )


def quaternion_to_euler_angle(params: dict, iterations: int):
    """Convert quaternions [w, x, y, z] to Euler angles ZYX [phi, theta, psi] in radians.

    Uses Basilisk's native ``EP2Euler321`` Schaub-Junkins primitive so the
    timed work is the library's own quaternion-to-Euler decomposition. The
    returned ``[theta1, theta2, theta3]`` correspond directly to the
    aerospace 3-2-1 sequence ``[phi (yaw, z), theta (pitch, y'), psi (roll,
    x'')]`` that brahe and Hipparchus' RotationOrder.ZYX also report.
    """
    quaternions = [np.array(q, dtype=float) for q in params["quaternions"]]

    def run():
        results = []
        for q in quaternions:
            ea = rbk.EP2Euler321(q)
            results.append([float(ea[0]), float(ea[1]), float(ea[2])])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.quaternion_to_euler_angle", iterations, times, results
    )


def euler_angle_to_quaternion(params: dict, iterations: int):
    """Convert Euler angles ZYX [phi, theta, psi] in radians to quaternions [w, x, y, z].

    Basilisk's ``euler3212EP`` returns the active-Hamilton quaternion for the
    aerospace intrinsic ZYX sequence (phi about z first, psi about body x
    last), matching brahe and the updated Java/Orekit adapter element-for-
    element. No reorder or sign change is needed.
    """
    angles = [np.array(a, dtype=float) for a in params["angles"]]

    def run():
        results = []
        for ea in angles:
            q = rbk.euler3212EP(ea)
            results.append([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.euler_angle_to_quaternion", iterations, times, results
    )
