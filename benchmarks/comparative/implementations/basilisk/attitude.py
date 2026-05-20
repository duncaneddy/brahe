"""Basilisk attitude conversion benchmarks.

Conventions:
- Quaternion (Euler parameter, EP): scalar-first [w, x, y, z]. Matches brahe
  and Orekit; no reorder needed.
- Rotation matrix: Basilisk's EP2C and C2EP use the same convention as brahe
  and OreKit (C = EP2C(q) matches brahe's quaternion_to_rotation_matrix
  element-for-element). No transposition required.
- Euler angles: brahe/OreKit decompose C = Rz(phi)@Ry(theta)@Rx(psi) (active
  left-to-right ZYX), giving C[2,0]=sin(theta), C[1,0]=-sin(phi)*cos(theta),
  C[2,1]=-cos(theta)*sin(psi). Basilisk's EP2Euler321 uses the Schaub-Junkins
  passive right-to-left ordering C = Rx(psi)@Ry(theta)@Rz(phi) and produces
  different angle triplets for the same rotation. We bypass EP2Euler321 and
  extract the ZYX angles directly from EP2C using brahe's element formulas.
"""

import math

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
    """Convert quaternions [w, x, y, z] to Euler angles ZYX [phi, theta, psi] in radians."""
    quaternions = [np.array(q, dtype=float) for q in params["quaternions"]]

    def run():
        results = []
        for q in quaternions:
            # EP2Euler321 decomposes C = Rx(psi)@Ry(theta)@Rz(phi) (Schaub-Junkins
            # passive right-to-left). brahe/OreKit use C = Rz(phi)@Ry(theta)@Rx(psi)
            # (active left-to-right). For the same rotation matrix C = EP2C(q) the two
            # conventions yield different angle triplets.
            #
            # Correct approach: extract angles analytically from the rotation matrix C
            # using brahe's ZYX layout where C[2,0] = sin(theta):
            #   theta = asin(C[2,0])
            #   phi   = atan2(-C[1,0], C[0,0])
            #   psi   = atan2(-C[2,1], C[2,2])
            C = rbk.EP2C(q)
            theta = math.asin(float(C[2][0]))
            cos_theta = math.cos(theta)
            phi = math.atan2(-float(C[1][0]) / cos_theta, float(C[0][0]) / cos_theta)
            psi = math.atan2(-float(C[2][1]) / cos_theta, float(C[2][2]) / cos_theta)
            results.append([phi, theta, psi])
        return results

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.quaternion_to_euler_angle", iterations, times, results
    )


def euler_angle_to_quaternion(params: dict, iterations: int):
    """Convert Euler angles ZYX [phi, theta, psi] in radians to quaternions [w, x, y, z].

    Basilisk's euler3212EP uses the Schaub-Junkins passive-rotation convention,
    which matches Hipparchus's Rotation extraction (the Java/Orekit baseline).
    brahe-Python and brahe-Rust use a different convention on this specific
    task and show a pre-existing ~0.67 max error against Java; Basilisk
    matches Java at machine epsilon.
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
