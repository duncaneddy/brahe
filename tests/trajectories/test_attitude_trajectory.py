"""Tests for AttitudeTrajectory / AttitudeState / OrientationProvider bindings — parity with
Rust tests in src/trajectories/attitude_trajectory.rs."""

import math

import numpy as np
import pytest

import brahe as bh
from brahe.trajectories import AttitudeState, AttitudeTrajectory


def z_axis_quaternion(theta):
    """Quaternion for a rotation of theta radians about the z-axis."""
    return bh.Quaternion(math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


def body_frames():
    return (
        bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY(None)),
        bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY(None)),
    )


def test_attitude_state_construction():
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    state = AttitudeState(q)
    assert state.quaternion == q
    assert state.angular_velocity is None

    omega = np.array([0.001, 0.002, 0.003])
    state_with_rate = AttitudeState(q, omega)
    np.testing.assert_allclose(state_with_rate.angular_velocity, omega)


def test_attitude_trajectory_add_and_len():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    assert len(traj) == 0

    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    assert len(traj) == 1
    assert not traj.has_rates

    traj.add(t0 + 60.0, z_axis_quaternion(0.1))
    assert len(traj) == 2
    assert traj.start_epoch == t0
    assert traj.end_epoch == t0 + 60.0


def test_attitude_trajectory_add_repeated_epoch_is_discontinuity():
    """Mirror of test_attitude_trajectory_add_repeated_epoch_is_discontinuity in Rust."""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0, z_axis_quaternion(0.1))

    # Both states are kept, so an impulsive slew can hold its pre- and
    # post-maneuver attitude at the same instant.
    assert len(traj) == 2

    # A query at the discontinuity is right-continuous: it returns the most
    # recently added state rather than producing NaN.
    np.testing.assert_allclose(
        traj.quaternion(t0).to_vector(scalar_first=True),
        z_axis_quaternion(0.1).to_vector(scalar_first=True),
        atol=1e-12,
    )


def test_attitude_trajectory_add_rejects_mixed_rate_presence():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(Exception, match="angular velocity"):
        traj.add(t0 + 60.0, z_axis_quaternion(0.1), np.array([0.0, 0.0, 0.01]))


def test_attitude_trajectory_frame_a_frame_b():
    frame_a = bh.ReferenceFrame.celestial(bh.CelestialFrame.GCRF)
    frame_b = bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY("1"))
    traj = AttitudeTrajectory(frame_a, frame_b)
    assert traj.frame_a == frame_a
    assert traj.frame_b == frame_b


def test_attitude_trajectory_interpolate_slerp_constant_rate_exact():
    """Mirror of test_attitude_trajectory_interpolate_slerp_constant_rate_exact in Rust."""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    assert traj.interpolation_method == "SLERP"

    omega = 0.01  # rad/s
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    dt = 100.0  # seconds

    traj.add(t0, z_axis_quaternion(0.0), np.array([0.0, 0.0, omega]))
    traj.add(t0 + dt, z_axis_quaternion(omega * dt), np.array([0.0, 0.0, omega]))

    f = 0.37
    query = t0 + f * dt
    q = traj.quaternion(query)

    analytic = z_axis_quaternion(omega * f * dt)
    np.testing.assert_allclose(
        q.to_vector(scalar_first=True),
        analytic.to_vector(scalar_first=True),
        atol=1e-12,
    )

    omega_interp = traj.angular_velocity(query)
    assert omega_interp[2] == pytest.approx(omega, abs=1e-12)


def test_attitude_trajectory_interpolate_linear_hemisphere_continuity():
    """Mirror of test_attitude_trajectory_interpolate_linear_hemisphere_continuity in Rust."""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LINEAR")
    assert traj.interpolation_method == "LINEAR"

    omega = 0.05  # rad/s
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Store 5 samples 1 second apart, with alternating sign to simulate an
    # arbitrary double-cover representative choice at each sample.
    for i in range(5):
        t = t0 + float(i)
        theta = omega * i
        q = z_axis_quaternion(theta)
        if i % 2 == 1:
            v = -q.to_vector(scalar_first=True)
            q = bh.Quaternion.from_vector(v, scalar_first=True)
        traj.add(t, q)

    # Query at the midpoint between index 2 and 3 (opposite stored signs).
    query = t0 + 2.5
    q = traj.quaternion(query)

    analytic = z_axis_quaternion(omega * 2.5)
    dot = np.dot(q.to_vector(scalar_first=True), analytic.to_vector(scalar_first=True))

    # A correctly hemisphere-aligned interpolation stays close to the
    # analytic attitude (dot near +1); a sign-flip bug would land near the
    # negative analytic quaternion or the degenerate near-zero vector.
    assert dot > 0.999, f"dot = {dot}"


def test_attitude_trajectory_set_interpolation_method_lagrange():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("lagrange", degree=3)
    assert traj.interpolation_method == "LAGRANGE"
    assert traj.interpolation_degree == 3

    traj.set_interpolation_method("Slerp")
    assert traj.interpolation_method == "SLERP"
    assert traj.interpolation_degree is None


def test_attitude_trajectory_set_interpolation_method_lagrange_requires_degree():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    with pytest.raises(ValueError, match="degree"):
        traj.set_interpolation_method("LAGRANGE")


def test_attitude_trajectory_set_interpolation_method_unknown_errors():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        traj.set_interpolation_method("CUBIC_SPLINE")


def test_attitude_trajectory_interpolate_lagrange_degree_zero_errors():
    """Mirror of test_attitude_trajectory_interpolate_lagrange_degree_zero_errors in Rust."""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=0)

    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 60.0, z_axis_quaternion(0.2))

    with pytest.raises(Exception, match="degree"):
        traj.quaternion(t0 + 30.0)


def test_attitude_provider_angular_velocity_none_without_rates():
    """Mirror of test_attitude_provider_angular_velocity_none_without_rates in Rust."""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    assert not traj.has_rates

    # The merged OrientationProvider contract reports a provider carrying no
    # rate data as None rather than raising.
    assert traj.angular_velocity(t0) is None


def test_attitude_provider_euler_angle_euler_axis_rotation_matrix():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    q = z_axis_quaternion(math.radians(30.0))
    traj.add(t0, q)

    euler = traj.euler_angle(t0, bh.EulerAngleOrder.ZYX)
    assert euler.phi == pytest.approx(math.radians(30.0), abs=1e-10)

    axis = traj.euler_axis(t0)
    np.testing.assert_allclose(axis.axis, np.array([0.0, 0.0, 1.0]), atol=1e-10)

    rot = traj.rotation_matrix(t0)
    np.testing.assert_allclose(
        rot.to_matrix(), q.to_rotation_matrix().to_matrix(), atol=1e-10
    )
