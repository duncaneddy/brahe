from math import cos, sin

import numpy as np
import pytest

from brahe import (
    AngleFormat,
    BraheError,
    EulerAngle,
    EulerAngleOrder,
    Quaternion,
    angular_velocity_from_quaternion_derivative,
    angular_velocity_to_euler_rates,
    euler_rates_to_angular_velocity,
    quaternion_derivative,
)


def axis_history(n, w, t):
    """Analytic single-axis quaternion history and its derivative.

    Rotation about unit axis n at constant rate w gives
    q(t) = [cos(w t / 2), sin(w t / 2) * n] with q_dot known in closed form.
    """
    half = 0.5 * w * t
    q = Quaternion(cos(half), sin(half) * n[0], sin(half) * n[1], sin(half) * n[2])
    q_dot_expected = np.array(
        [
            -0.5 * w * sin(half),
            0.5 * w * cos(half) * n[0],
            0.5 * w * cos(half) * n[1],
            0.5 * w * cos(half) * n[2],
        ]
    )
    return q, q_dot_expected


def test_quaternion_derivative_single_axis():
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]) / np.linalg.norm(np.array([1.0, 2.0, 3.0])),
    ]
    w = 0.37
    for n in axes:
        for t in [0.0, 0.4, 1.9, 5.0]:
            q, q_dot_expected = axis_history(n, w, t)
            q_dot = quaternion_derivative(q, w * n)
            for i in range(4):
                assert q_dot[i] == pytest.approx(q_dot_expected[i], abs=1e-12)


def test_angular_velocity_from_quaternion_derivative_roundtrip():
    q = Quaternion.from_euler_angle(
        EulerAngle(EulerAngleOrder.ZYX, 0.3, -0.7, 1.1, AngleFormat.RADIANS)
    )
    omega = np.array([0.05, -0.02, 0.4])
    q_dot = quaternion_derivative(q, omega)
    recovered = angular_velocity_from_quaternion_derivative(q, q_dot)
    for i in range(3):
        assert recovered[i] == pytest.approx(omega[i], abs=1e-12)


def test_euler_rates_to_angular_velocity_zyx_classic():
    # Classic aerospace 3-2-1 body-rate map:
    #   p = psi_dot - phi_dot*sin(theta)
    #   q = theta_dot*cos(psi) + phi_dot*cos(theta)*sin(psi)
    #   r = -theta_dot*sin(psi) + phi_dot*cos(theta)*cos(psi)
    phi, theta, psi = 0.3, -0.4, 0.7
    angles = EulerAngle(EulerAngleOrder.ZYX, phi, theta, psi, AngleFormat.RADIANS)
    rates = np.array([0.11, -0.23, 0.05])  # (phi_dot, theta_dot, psi_dot)
    omega = euler_rates_to_angular_velocity(angles, rates)
    expected = np.array(
        [
            rates[2] - rates[0] * sin(theta),
            rates[1] * cos(psi) + rates[0] * cos(theta) * sin(psi),
            -rates[1] * sin(psi) + rates[0] * cos(theta) * cos(psi),
        ]
    )
    for i in range(3):
        assert omega[i] == pytest.approx(expected[i], abs=1e-12)


def test_angular_velocity_to_euler_rates_roundtrip():
    angles = EulerAngle(EulerAngleOrder.ZXZ, 0.5, 0.8, -1.2, AngleFormat.RADIANS)
    rates = np.array([0.02, 0.13, -0.07])
    omega = euler_rates_to_angular_velocity(angles, rates)
    recovered = angular_velocity_to_euler_rates(angles, omega)
    for i in range(3):
        assert recovered[i] == pytest.approx(rates[i], abs=1e-10)


def test_angular_velocity_to_euler_rates_singularities():
    # Distinct-axis family: singular at theta = +/- 90 deg
    tait = EulerAngle(EulerAngleOrder.ZYX, 0.4, np.pi / 2, 0.1, AngleFormat.RADIANS)
    with pytest.raises(BraheError):
        angular_velocity_to_euler_rates(tait, np.array([0.1, 0.0, 0.0]))

    # Repeated-axis family: singular at theta = 0
    sym = EulerAngle(EulerAngleOrder.ZXZ, 0.4, 0.0, 0.1, AngleFormat.RADIANS)
    with pytest.raises(BraheError):
        angular_velocity_to_euler_rates(sym, np.array([0.1, 0.0, 0.0]))


def test_quaternion_derivative_matches_rotation_matrix_derivative():
    q0 = Quaternion.from_euler_angle(
        EulerAngle(EulerAngleOrder.XYZ, 0.2, 0.5, -0.3, AngleFormat.RADIANS)
    )
    w = 0.9
    omega = np.array([0.0, 0.0, w])
    t = 0.8
    dt = 1e-6

    def q_at(tau):
        half = 0.5 * w * tau
        spin = Quaternion(cos(half), 0.0, 0.0, sin(half))
        return q0 * spin

    # Central difference on raw components with hemisphere continuity enforced
    qc = np.array(q_at(t).to_vector(True))
    qp = np.array(q_at(t + dt).to_vector(True))
    qm = np.array(q_at(t - dt).to_vector(True))
    if np.dot(qp, qc) < 0.0:
        qp = -qp
    if np.dot(qm, qc) < 0.0:
        qm = -qm
    q_dot_numeric = (qp - qm) / (2.0 * dt)

    q_dot = quaternion_derivative(q_at(t), omega)
    for i in range(4):
        assert q_dot[i] == pytest.approx(q_dot_numeric[i], abs=1e-8)

    # Cross-check omega against the matrix route used in src/frames/custom.rs
    def r(tau):
        return np.array(q_at(tau).to_rotation_matrix().to_matrix())

    r_dot = (r(t + dt) - r(t - dt)) / (2.0 * dt)
    s = -r_dot @ r(t).T
    omega_matrix = np.array(
        [
            (s[2, 1] - s[1, 2]) / 2.0,
            (s[0, 2] - s[2, 0]) / 2.0,
            (s[1, 0] - s[0, 1]) / 2.0,
        ]
    )
    omega_recovered = angular_velocity_from_quaternion_derivative(q_at(t), q_dot)
    for i in range(3):
        assert omega_matrix[i] == pytest.approx(omega[i], abs=1e-6)
        assert omega_recovered[i] == pytest.approx(omega[i], abs=1e-10)


@pytest.mark.parametrize(
    "order",
    [
        EulerAngleOrder.XYZ,
        EulerAngleOrder.XZY,
        EulerAngleOrder.YXZ,
        EulerAngleOrder.YZX,
        EulerAngleOrder.ZXY,
        EulerAngleOrder.ZYX,
        EulerAngleOrder.XYX,
        EulerAngleOrder.XZX,
        EulerAngleOrder.YXY,
        EulerAngleOrder.YZY,
        EulerAngleOrder.ZXZ,
        EulerAngleOrder.ZYZ,
    ],
)
def test_euler_rates_consistent_with_quaternion_kinematics(order):
    # Smooth angle trajectories, away from singularities for every family
    def ang(t):
        return (
            0.4 + 0.3 * sin(0.7 * t),
            0.9 + 0.2 * cos(0.5 * t),
            -0.2 + 0.25 * sin(0.9 * t),
        )

    def rate(t):
        return (
            0.3 * 0.7 * cos(0.7 * t),
            -0.2 * 0.5 * sin(0.5 * t),
            0.25 * 0.9 * cos(0.9 * t),
        )

    t = 1.3
    dt = 1e-6
    p, h, s = ang(t)
    pd, hd, sd = rate(t)
    angles = EulerAngle(order, p, h, s, AngleFormat.RADIANS)
    omega = euler_rates_to_angular_velocity(angles, np.array([pd, hd, sd]))

    def q_of(tau):
        a, b, c = ang(tau)
        return Quaternion.from_euler_angle(
            EulerAngle(order, a, b, c, AngleFormat.RADIANS)
        )

    qc = np.array(q_of(t).to_vector(True))
    qp = np.array(q_of(t + dt).to_vector(True))
    qm = np.array(q_of(t - dt).to_vector(True))
    if np.dot(qp, qc) < 0.0:
        qp = -qp
    if np.dot(qm, qc) < 0.0:
        qm = -qm
    q_dot = (qp - qm) / (2.0 * dt)
    omega_ref = angular_velocity_from_quaternion_derivative(q_of(t), q_dot)

    for i in range(3):
        assert omega[i] == pytest.approx(omega_ref[i], abs=1e-6)
