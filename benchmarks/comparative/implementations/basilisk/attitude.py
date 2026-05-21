"""Basilisk attitude conversion benchmarks.

Conventions:
- Quaternion (Euler parameter, EP): scalar-first [w, x, y, z]. Matches brahe
  and Orekit at the level of `quaternion_to_rotation_matrix` /
  `rotation_matrix_to_quaternion`; no reorder needed.
- Rotation matrix: Basilisk's EP2C and C2EP produce passive DCMs that match
  brahe and Hipparchus element-for-element.
- Euler angles: brahe exposes aerospace intrinsic ZYX (yaw-pitch-roll) — phi
  rotates about z first, psi about the body x last — with the passive DCM
  M = Rx_p(psi) * Ry_p(theta) * Rz_p(phi). We extract those angles directly
  from EP2C(q) using the aerospace ZYX element formulas. Basilisk's
  ``euler3212EP`` already returns the active-Hamilton quaternion for the
  same convention, matching brahe and Orekit at machine precision.
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
            # Extract aerospace intrinsic ZYX angles from the passive DCM
            # C = EP2C(q) = Rx_p(psi) * Ry_p(theta) * Rz_p(phi):
            #   C[0,0] = cos(theta)*cos(phi)   C[0,1] = cos(theta)*sin(phi)
            #   C[0,2] = -sin(theta)
            #   C[1,2] = sin(psi)*cos(theta)   C[2,2] = cos(psi)*cos(theta)
            # Bypasses Basilisk's EP2Euler321 because Schaub-Junkins decomposes
            # the matrix using the same elements in a different role-mapping;
            # this extraction matches brahe and the updated Java/Orekit adapter
            # element-for-element.
            C = rbk.EP2C(q)
            phi = math.atan2(float(C[0][1]), float(C[0][0]))
            theta = -math.asin(float(C[0][2]))
            psi = math.atan2(float(C[1][2]), float(C[2][2]))
            results.append([phi, theta, psi])
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
