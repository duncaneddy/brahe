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
