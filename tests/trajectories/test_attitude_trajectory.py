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


def small_attitude_trajectory():
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 60.0, z_axis_quaternion(0.2))
    return traj


def test_attitude_state_new():
    """Rust: test_attitude_state_new"""
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    state = AttitudeState(q)
    assert state.quaternion == q
    assert state.angular_velocity is None


def test_attitude_state_with_angular_velocity():
    """Rust: test_attitude_state_with_angular_velocity"""
    omega = np.array([0.1, 0.2, 0.3])
    state = AttitudeState(bh.Quaternion(1.0, 0.0, 0.0, 0.0), omega)
    np.testing.assert_array_equal(state.angular_velocity, omega)


def test_attitude_interpolation_method_default():
    """Rust: test_attitude_interpolation_method_default"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    assert traj.interpolation_method == "SLERP"


def test_attitude_trajectory_new():
    """Rust: test_attitude_trajectory_new"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    assert len(traj) == 0
    assert traj.interpolation_method == "SLERP"
    assert not traj.has_rates


def test_attitude_trajectory_add_sorts_out_of_order_epochs():
    """Rust: test_attitude_trajectory_add_sorts_out_of_order_epochs"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    traj.add(t0 + 60.0, z_axis_quaternion(0.1))
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 30.0, z_axis_quaternion(0.05))

    assert len(traj) == 3
    assert traj.start_epoch == t0
    assert traj.end_epoch == t0 + 60.0
    np.testing.assert_array_equal(
        traj.quaternion(t0).to_vector(scalar_first=True),
        z_axis_quaternion(0.0).to_vector(scalar_first=True),
    )
    np.testing.assert_array_equal(
        traj.quaternion(t0 + 30.0).to_vector(scalar_first=True),
        z_axis_quaternion(0.05).to_vector(scalar_first=True),
    )
    np.testing.assert_array_equal(
        traj.quaternion(t0 + 60.0).to_vector(scalar_first=True),
        z_axis_quaternion(0.1).to_vector(scalar_first=True),
    )


def test_attitude_trajectory_add_rate_mixing_error():
    """Rust: test_attitude_trajectory_add_rate_mixing_error"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))

    with pytest.raises(Exception, match="angular velocity"):
        traj.add(t0 + 60.0, z_axis_quaternion(0.1), np.array([0.0, 0.0, 0.01]))


def test_attitude_trajectory_add_rate_mixing_error_reverse_direction():
    """Rust: test_attitude_trajectory_add_rate_mixing_error_reverse_direction"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0), np.array([0.0, 0.0, 0.01]))

    with pytest.raises(Exception, match="does not carry angular velocity"):
        traj.add(t0 + 60.0, z_axis_quaternion(0.1))


def test_attitude_trajectory_has_rates():
    """Rust: test_attitude_trajectory_has_rates"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    assert not traj.has_rates

    traj.add(t0, z_axis_quaternion(0.0), np.array([0.0, 0.0, 0.01]))

    assert traj.has_rates


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


def test_attitude_trajectory_interpolate_exact_node_returns_stored_state():
    """Rust: test_attitude_trajectory_interpolate_exact_node_returns_stored_state"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 60.0, z_axis_quaternion(0.2))

    q = traj.quaternion(t0)
    np.testing.assert_array_equal(
        q.to_vector(scalar_first=True),
        z_axis_quaternion(0.0).to_vector(scalar_first=True),
    )

    q = traj.quaternion(t0 + 60.0)
    np.testing.assert_array_equal(
        q.to_vector(scalar_first=True),
        z_axis_quaternion(0.2).to_vector(scalar_first=True),
    )


def test_attitude_trajectory_interpolate_out_of_range():
    """Rust: test_attitude_trajectory_interpolate_out_of_range"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 60.0, z_axis_quaternion(0.2))

    with pytest.raises(Exception, match="before trajectory start"):
        traj.quaternion(t0 - 10.0)
    with pytest.raises(Exception, match="after trajectory end"):
        traj.quaternion(t0 + 70.0)


def test_attitude_trajectory_interpolate_lagrange_min_points_error():
    """Rust: test_attitude_trajectory_interpolate_lagrange_min_points_error"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, z_axis_quaternion(0.0))
    traj.add(t0 + 60.0, z_axis_quaternion(0.2))

    # Only 2 points but degree 3 requires 4
    with pytest.raises(Exception, match="requires"):
        traj.quaternion(t0 + 30.0)


def test_attitude_trajectory_interpolate_lagrange_tolerance():
    """Rust: test_attitude_trajectory_interpolate_lagrange_tolerance"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Smooth, non-constant-rate rotation profile about the z-axis.
    def theta(t):
        return 0.3 * t + 0.05 * math.sin(t)

    for i in range(6):
        t = float(i)
        traj.add(t0 + t, z_axis_quaternion(theta(t)))

    query_t = 2.5
    q = traj.quaternion(t0 + query_t)
    analytic = z_axis_quaternion(theta(query_t))

    dot = np.dot(q.to_vector(scalar_first=True), analytic.to_vector(scalar_first=True))
    angular_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    assert angular_error < 5e-3, f"angular_error = {angular_error}"


def test_attitude_trajectory_interpolate_lagrange_centered_window_tight_tolerance():
    """Rust: test_attitude_trajectory_interpolate_lagrange_centered_window_tight_tolerance"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Same smooth, non-constant-rate rotation profile as the edge-window case
    # above, but sampled ten times finer (0.1s spacing instead of 1s).
    def theta(t):
        return 0.3 * t + 0.05 * math.sin(t)

    for i in range(6):
        t = 0.1 * i
        traj.add(t0 + t, z_axis_quaternion(theta(t)))

    # At query_t = 0.15, the selected Lagrange window is exactly centered on
    # the query, unlike the t = 2.5 case above.
    query_t = 0.15
    q = traj.quaternion(t0 + query_t)
    analytic = z_axis_quaternion(theta(query_t))

    dot = np.dot(q.to_vector(scalar_first=True), analytic.to_vector(scalar_first=True))
    angular_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    assert angular_error < 1e-6, f"angular_error = {angular_error}"


def test_attitude_trajectory_interpolate_lagrange_wide_span_sequential_alignment():
    """Rust: test_attitude_trajectory_interpolate_lagrange_wide_span_sequential_alignment"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)

    omega = 1.5  # rad/s
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(4):
        t = float(i)
        traj.add(t0 + t, z_axis_quaternion(omega * t))

    # Query at the midpoint of the (only, entire-trajectory) window.
    query_t = 1.5
    q = traj.quaternion(t0 + query_t)
    analytic = z_axis_quaternion(omega * query_t)

    dot = np.dot(q.to_vector(scalar_first=True), analytic.to_vector(scalar_first=True))
    assert dot > 0.99, f"dot = {dot}"


def test_lagrange_window_does_not_span_a_discontinuity():
    """Rust: test_lagrange_window_does_not_span_a_discontinuity"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)

    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    # Continuous run, then an impulsive slew at t0 + 20 whose post-state
    # jumps, then a second continuous run.
    for offset, angle in [(0.0, 0.0), (10.0, 0.2), (20.0, 0.4)]:
        traj.add(t0 + offset, z_axis_quaternion(angle))
    for offset, angle in [(20.0, 2.0), (30.0, 2.2), (40.0, 2.4)]:
        traj.add(t0 + offset, z_axis_quaternion(angle))

    # A query inside the first run fits only that run's samples. Fitting
    # across the repeated epoch would divide by a zero-length abscissa
    # difference and return NaN.
    v = traj.quaternion(t0 + 5.0).to_vector(scalar_first=True)
    assert np.all(np.isfinite(v)), v
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)

    # The same holds for a query inside the second run.
    v = traj.quaternion(t0 + 35.0).to_vector(scalar_first=True)
    assert np.all(np.isfinite(v)), v
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)


def test_attitude_trajectory_interpolate_lagrange_with_angular_velocity():
    """Rust: test_attitude_trajectory_interpolate_lagrange_with_angular_velocity"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)

    omega = np.array([0.0, 0.0, 0.02])
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(4):
        t = float(i)
        traj.add(t0 + t, z_axis_quaternion(omega[2] * t), omega)

    interpolated_omega = traj.angular_velocity(t0 + 1.5)
    assert interpolated_omega[2] == pytest.approx(omega[2], abs=1e-12)


def test_attitude_trajectory_interpolate_lagrange_in_window_hemisphere_flip():
    """Rust: test_attitude_trajectory_interpolate_lagrange_in_window_hemisphere_flip"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    traj.set_interpolation_method("LAGRANGE", degree=3)

    omega = 4.0  # rad/s
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(4):
        t = float(i)
        traj.add(t0 + t, z_axis_quaternion(omega * t))

    q = traj.quaternion(t0 + 1.5)
    norm = np.linalg.norm(q.to_vector(scalar_first=True))
    assert norm == pytest.approx(1.0, abs=1e-9)


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


def test_attitude_provider_quaternion_and_defaults_consistent():
    """Rust: test_attitude_provider_quaternion_and_defaults_consistent"""
    traj = small_attitude_trajectory()
    epoch = traj.start_epoch + 30.0

    q = traj.quaternion(epoch)

    # euler_angle default: EulerAngle::from_quaternion(quaternion, order)
    euler = traj.euler_angle(epoch, bh.EulerAngleOrder.ZYX)
    expected_euler = q.to_euler_angle(bh.EulerAngleOrder.ZYX)
    assert euler.phi == expected_euler.phi
    assert euler.theta == expected_euler.theta
    assert euler.psi == expected_euler.psi

    # euler_axis default: ToAttitude::to_euler_axis on the same quaternion
    axis = traj.euler_axis(epoch)
    expected_axis = q.to_euler_axis()
    assert axis.angle == expected_axis.angle

    # rotation_matrix default: ToAttitude::to_rotation_matrix on the same quaternion
    r = traj.rotation_matrix(epoch)
    expected_r = q.to_rotation_matrix()
    np.testing.assert_array_equal(r.to_matrix(), expected_r.to_matrix())


def test_attitude_provider_angular_velocity_errors_out_of_coverage_without_rates():
    """Rust: test_attitude_provider_angular_velocity_errors_out_of_coverage_without_rates"""
    traj = small_attitude_trajectory()
    before_start = traj.start_epoch - 10.0
    after_end = traj.end_epoch + 10.0

    with pytest.raises(Exception, match="before trajectory start"):
        traj.angular_velocity(before_start)
    with pytest.raises(Exception, match="after trajectory end"):
        traj.angular_velocity(after_end)


def test_attitude_provider_angular_velocity_errors_for_empty_trajectory():
    """Rust: test_attitude_provider_angular_velocity_errors_for_empty_trajectory"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    epoch = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    with pytest.raises(Exception, match="empty trajectory"):
        traj.angular_velocity(epoch)


def test_attitude_provider_angular_velocity_with_rates():
    """Rust: test_attitude_provider_angular_velocity_with_rates"""
    frame_a, frame_b = body_frames()
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omega = np.array([0.0, 0.0, 0.01])
    traj.add(t0, z_axis_quaternion(0.0), omega)
    traj.add(t0 + 60.0, z_axis_quaternion(0.6), omega)

    result = traj.angular_velocity(t0 + 30.0)
    np.testing.assert_array_equal(result, omega)
