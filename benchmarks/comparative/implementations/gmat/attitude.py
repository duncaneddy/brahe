"""GMAT benchmark implementations for attitude conversions.

GMAT exposes attitude conversions via the AttitudeConversionUtility class:
  - ToCosineMatrix(Rvector)          -> Rmatrix33
  - ToQuaternion(Rmatrix33)          -> Rvector (4-element, scalar-LAST)
  - ToEulerAngles(Rvector, s1,s2,s3) -> Rvector3
  - ToQuaternion(Rvector3, s1,s2,s3) -> Rvector (4-element, scalar-LAST)

All methods require SWIG-typed objects (Rvector / Rvector3 / Rmatrix33), not
plain Python lists.  Use SetElement / GetElement for access.

Verified conventions (Step 2 spike):
  - GMAT quaternion order: scalar-LAST [q1, q2, q3, q4] where q4 is the scalar.
    Confirmed by round-trip against brahe scalar-first [w, x, y, z].
  - GMAT's ToCosineMatrix returns the PASSIVE (body-to-inertial) DCM, identical
    to Basilisk EP2C and Hipparchus Rotation.getMatrix() — all three agree
    element-for-element.  This is the TRANSPOSE of brahe's active Python formula.
  - GMAT's ToEulerAngles(rv, 3, 2, 1) does NOT match Java/Basilisk's ZYX
    extraction despite using the same passive matrix.  The correct approach is to
    call ToCosineMatrix and then extract angles analytically using the same
    formula as the Java baseline:
      theta = asin(C[2][0]), phi = atan2(-C[1][0]/ct, C[0][0]/ct),
      psi   = atan2(-C[2][1]/ct, C[2][2]/ct)
    where ct = cos(theta).  This matches Java at machine epsilon.

Convention reconciliation (outside the timed region):
  - Quaternion: GMAT scalar-LAST <-> brahe scalar-FIRST via quat_*_to_* helpers.
"""

import numpy as np

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    quat_brahe_to_gmat,
    quat_gmat_to_brahe,
    time_iterations,
)


def _acu():
    import gmatpy as gmat
    return gmat.AttitudeConversionUtility


def _make_rvector(vals):
    """Build a GMAT Rvector from a Python sequence of floats."""
    import gmatpy as gmat
    rv = gmat.Rvector(len(vals))
    for i, v in enumerate(vals):
        rv.SetElement(i, float(v))
    return rv


def _make_rvector3(vals):
    """Build a GMAT Rvector3 from a 3-element Python sequence of floats."""
    import gmatpy as gmat
    rv = gmat.Rvector3()
    for i, v in enumerate(vals):
        rv.SetElement(i, float(v))
    return rv


def _rvector_to_list(rv, n):
    """Extract n elements from a GMAT Rvector / Rvector3 into a Python list."""
    return [rv.GetElement(i) for i in range(n)]


def _rmatrix33_to_flat(mat):
    """Flatten a GMAT Rmatrix33 to a 9-float row-major list."""
    return [mat.GetElement(r, c) for r in range(3) for c in range(3)]


def quaternion_to_rotation_matrix(params: dict, iterations: int):
    """Quaternions [w, x, y, z] -> 3x3 row-major rotation matrices (9 floats).

    Convention: brahe scalar-first -> GMAT scalar-last via quat_brahe_to_gmat;
    DCM elements are read via Rmatrix33.GetElement(r, c).
    """
    acu = _acu()
    # Pre-convert to GMAT Rvectors outside the timed region.
    gmat_rvecs = [_make_rvector(quat_brahe_to_gmat(q)) for q in params["quaternions"]]

    def run():
        return [_rmatrix33_to_flat(acu.ToCosineMatrix(rv)) for rv in gmat_rvecs]

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.quaternion_to_rotation_matrix", iterations, times, results
    )


def rotation_matrix_to_quaternion(params: dict, iterations: int):
    """3x3 row-major rotation matrices (9 floats) -> quaternions [w, x, y, z].

    Matrices arrive as flat 9-float lists (row-major).  We first convert each to
    a GMAT Rmatrix33 via ToCosineMatrix round-trip (build from numpy, pass as
    Rvector to get a quaternion, then invert — actually GMAT's ToQuaternion
    accepts an Rmatrix33 directly).
    """
    acu = _acu()

    # Build Rmatrix33 objects outside the timed region by converting each flat
    # matrix to a unit quaternion first (via numpy), then calling ToCosineMatrix
    # to obtain a proper Rmatrix33.
    import gmatpy as gmat

    gmat_mats = []
    for m in params["matrices"]:
        arr = np.array(m, dtype=float).reshape(3, 3)
        # Shim: build Rmatrix33 using SetElement
        rmat = gmat.Rmatrix33()
        for r in range(3):
            for c in range(3):
                rmat.SetElement(r, c, arr[r, c])
        gmat_mats.append(rmat)

    def run():
        out = []
        for rmat in gmat_mats:
            q_rv = acu.ToQuaternion(rmat)
            out.append(quat_gmat_to_brahe(_rvector_to_list(q_rv, 4)))
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.rotation_matrix_to_quaternion", iterations, times, results
    )


def quaternion_to_euler_angle(params: dict, iterations: int):
    """Quaternions [w, x, y, z] -> Euler angles ZYX [phi, theta, psi] (radians).

    GMAT's ToCosineMatrix returns the passive (body-to-inertial) DCM.
    Java/OreKit (Hipparchus getMatrix()) and Basilisk (EP2C) also return the
    passive DCM.  All three agree on the matrix element-for-element.

    We extract ZYX Euler angles directly from the passive DCM using the same
    formula as the Java baseline:
      theta = asin(C[2][0])
      phi   = atan2(-C[1][0] / cos(theta), C[0][0] / cos(theta))
      psi   = atan2(-C[2][1] / cos(theta), C[2][2] / cos(theta))

    Using GMAT's ToEulerAngles(rv, 3, 2, 1) was tried but yields different
    values because GMAT's internal ZYX decomposition uses a different formula.
    The direct matrix extraction matches the Java baseline at machine epsilon.
    """
    import math
    acu = _acu()
    gmat_rvecs = [_make_rvector(quat_brahe_to_gmat(q)) for q in params["quaternions"]]

    def _extract_zyx(mat):
        c20 = mat.GetElement(2, 0)
        # Clamp to avoid numerical issues with asin
        c20 = max(-1.0, min(1.0, c20))
        theta = math.asin(c20)
        cos_theta = math.cos(theta)
        if abs(cos_theta) < 1e-10:
            # Gimbal lock: set phi=0, solve for psi
            phi = 0.0
            psi = math.atan2(-mat.GetElement(0, 1), mat.GetElement(1, 1))
        else:
            phi = math.atan2(
                -mat.GetElement(1, 0) / cos_theta,
                mat.GetElement(0, 0) / cos_theta,
            )
            psi = math.atan2(
                -mat.GetElement(2, 1) / cos_theta,
                mat.GetElement(2, 2) / cos_theta,
            )
        return [phi, theta, psi]

    def run():
        return [_extract_zyx(acu.ToCosineMatrix(rv)) for rv in gmat_rvecs]

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.quaternion_to_euler_angle", iterations, times, results
    )


def euler_angle_to_quaternion(params: dict, iterations: int):
    """Euler angles ZYX [phi, theta, psi] (radians) -> quaternions [w, x, y, z].

    GMAT sequence (3, 2, 1) = ZYX.  Input [phi, theta, psi] is passed as-is to
    Rvector3 and returned as GMAT scalar-last quaternion, then converted to brahe
    scalar-first [w, x, y, z].
    """
    acu = _acu()
    gmat_rvec3s = [_make_rvector3(a) for a in params["angles"]]

    def run():
        out = []
        for rv3 in gmat_rvec3s:
            q_rv = acu.ToQuaternion(rv3, 3, 2, 1)
            out.append(quat_gmat_to_brahe(_rvector_to_list(q_rv, 4)))
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "attitude.euler_angle_to_quaternion", iterations, times, results
    )
