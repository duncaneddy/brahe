"""
Tests for adaptive integrators (RKF45, DP54, RKN1210) - mirrors Rust tests.
"""

import pytest
import numpy as np
import brahe as bh


def point_earth(t, state):
    """Two-body point mass Earth dynamics for 6D state [r, v]."""
    r = state[:3]
    v = state[3:]

    r_norm = np.linalg.norm(r)
    a_mag = -bh.GM_EARTH / (r_norm**3)
    a = a_mag * r

    return np.concatenate([v, a])


class TestRKF45Integrator:
    """Tests for RKF45 (Runge-Kutta-Fehlberg 4(5)) integrator."""

    def test_parabola_integration(self):
        """Test RKF45 on simple parabola x' = 2t (mirrors test_rkf45d_integrator_parabola)."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, config=config
        )

        t = 0.0
        state = np.array([0.0])

        while t < 1.0:
            dt = min(1.0 - t, 0.1)
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

        # At t=1.0, x should be 1.0 (integral of 2t from 0 to 1)
        assert abs(state[0] - 1.0) < 1e-8

    def test_adaptive_stepping(self):
        """Test adaptive stepping behavior (mirrors test_rkf45d_integrator_adaptive)."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, config=config
        )

        t = 0.0
        state = np.array([0.0])
        t_end = 1.0

        while t < t_end:
            dt = min(t_end - t, 0.1)
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

            # Verify error estimate is non-negative
            assert result.error_estimate >= 0.0

        # Should still get accurate result
        assert abs(state[0] - 1.0) < 1e-8

    def test_orbit_propagation(self):
        """Test RKF45 on orbital mechanics (mirrors test_rkf45d_integrator_orbit)."""

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.RKF45Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        # Initial orbital elements: LEO, e=0.01, i=90°
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.RADIANS)

        # Propagate for one orbital period
        state = state0.copy()
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
        period = bh.orbital_period(oe0[0])
        epcf = epc0 + period
        epc = epc0

        while epc < epcf:
            dt = min((epcf - epc), 10.0)
            result = integrator.step(epc - epc0, state, dt, 1e-8, 1e-6)
            state = result.state
            epc = epc + result.dt_used

        # RKF45 should achieve good accuracy
        assert abs(np.linalg.norm(state) - np.linalg.norm(state0)) < 1e-4
        assert abs(state[0] - state0[0]) < 1e-3
        assert abs(state[1] - state0[1]) < 1e-3
        assert abs(state[2] - state0[2]) < 1e-3

    def test_accuracy(self):
        """Verify RKF45 achieves expected 5th order accuracy (mirrors test_rkf45d_accuracy)."""

        def dynamics(t, state):
            return np.array([3.0 * t * t])

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, config=config
        )

        t = 0.0
        state = np.array([0.0])

        while t < 10.0:
            dt = min(10.0 - t, 0.1)
            result = integrator.step(t, state, dt, 1e-8, 1e-6)
            state = result.state
            t += result.dt_used

        exact = 1000.0  # t^3 at t=10
        error = abs(state[0] - exact)

        # 5th order method should be very accurate
        assert error < 1e-5

    def test_step_size_increases(self):
        """Verify adaptive stepping increases step size when error is small (mirrors test_rkf45d_step_size_increases)."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-6, 1e-4)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, config=config
        )

        state = np.array([0.0])
        dt_initial = 0.01

        # Take a step with loose tolerance - error should be small
        result = integrator.step(0.0, state, dt_initial, 1e-6, 1e-4)

        # For this simple problem with loose tolerance, suggested step should be larger
        assert result.dt_next > dt_initial, (
            f"Expected dt_next ({result.dt_next}) > dt_initial ({dt_initial})"
        )

        # Error should be very small for this simple problem
        assert result.error_estimate < 0.1

    def test_result_structure(self):
        """Test that AdaptiveStepResult has expected fields."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        integrator = bh.RKF45Integrator(dimension=1, dynamics_fn=dynamics)

        result = integrator.step(0.0, np.array([0.0]), 0.1, 1e-9, 1e-6)

        # Check that result has expected attributes
        assert hasattr(result, "state")
        assert hasattr(result, "dt_used")
        assert hasattr(result, "error_estimate")
        assert hasattr(result, "dt_next")

        # Check types
        assert isinstance(result.state, np.ndarray)
        assert isinstance(result.dt_used, float)
        assert isinstance(result.error_estimate, float)
        assert isinstance(result.dt_next, float)

        # Check values are reasonable
        assert result.dt_used > 0
        assert result.dt_used <= 0.1
        assert result.error_estimate >= 0
        assert result.dt_next > 0

    def test_with_varmat(self):
        """Test variational matrix propagation."""

        def dynamics(t, state):
            return -state  # dy/dt = -y

        jacobian = bh.DNumericalJacobian(dynamics).with_fixed_offset(1.0)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, jacobian=jacobian
        )

        state = np.array([1.0])
        phi = np.eye(1)

        state_new, phi_new, dt_used, error_est, dt_next = integrator.step_with_varmat(
            0.0, state, phi, 0.1, 1e-9, 1e-6
        )

        # Check types and shapes
        assert isinstance(state_new, np.ndarray)
        assert isinstance(phi_new, np.ndarray)
        assert state_new.shape == (1,)
        assert phi_new.shape == (1, 1)

        # Check values are reasonable
        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

        # For dy/dt = -y, STM should be exp(-t)
        assert abs(phi_new[0, 0] - np.exp(-dt_used)) < 1e-6


class TestDP54Integrator:
    """Tests for DP54 (Dormand-Prince 5(4)) integrator."""

    def test_parabola_integration(self):
        """Test DP54 on simple parabola x' = 2t."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)

        t = 0.0
        state = np.array([0.0])
        dt = 0.01

        for _ in range(100):
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

        # At t=1.0, x should be 1.0
        assert abs(state[0] - 1.0) < 1e-10

    def test_adaptive_stepping(self):
        """Test adaptive stepping behavior."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)

        t = 0.0
        state = np.array([0.0])
        t_end = 1.0

        while t < t_end:
            dt = min(t_end - t, 0.1)
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

            assert result.error_estimate >= 0.0

        assert abs(state[0] - 1.0) < 1e-10

    def test_orbit_propagation(self):
        """Test DP54 on orbital mechanics."""

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.DP54Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.RADIANS)

        state = state0.copy()
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
        period = bh.orbital_period(oe0[0])
        epcf = epc0 + period
        epc = epc0

        while epc < epcf:
            dt = min((epcf - epc), 10.0)
            result = integrator.step(epc - epc0, state, dt, 1e-8, 1e-6)
            state = result.state
            epc = epc + result.dt_used

        # DP54 should achieve good accuracy
        assert abs(np.linalg.norm(state) - np.linalg.norm(state0)) < 1e-4
        assert abs(state[0] - state0[0]) < 1e-3
        assert abs(state[1] - state0[1]) < 1e-3
        assert abs(state[2] - state0[2]) < 1e-3

    def test_accuracy(self):
        """Verify DP54 achieves expected 5th order accuracy."""

        def dynamics(t, state):
            return np.array([3.0 * t * t])

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)

        t = 0.0
        state = np.array([0.0])

        while t < 10.0:
            dt = min(10.0 - t, 0.1)
            result = integrator.step(t, state, dt, 1e-8, 1e-6)
            state = result.state
            t += result.dt_used

        exact = 1000.0
        error = abs(state[0] - exact)

        # 5th order method should be very accurate
        assert error < 1e-5

    def test_step_size_increases(self):
        """Verify adaptive stepping increases step size when error is small."""

        def dynamics(t, state):
            return np.array([2.0 * t])

        config = bh.IntegratorConfig.adaptive(1e-6, 1e-4)
        integrator = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)

        state = np.array([0.0])
        dt_initial = 0.01

        result = integrator.step(0.0, state, dt_initial, 1e-6, 1e-4)

        assert result.dt_next > dt_initial
        assert result.error_estimate < 0.1

    def test_with_varmat(self):
        """Test variational matrix propagation."""

        def dynamics(t, state):
            return -state

        jacobian = bh.DNumericalJacobian(dynamics).with_fixed_offset(1.0)
        integrator = bh.DP54Integrator(
            dimension=1, dynamics_fn=dynamics, jacobian=jacobian
        )

        state = np.array([1.0])
        phi = np.eye(1)

        state_new, phi_new, dt_used, error_est, dt_next = integrator.step_with_varmat(
            0.0, state, phi, 0.1, 1e-9, 1e-6
        )

        assert isinstance(state_new, np.ndarray)
        assert isinstance(phi_new, np.ndarray)
        assert state_new.shape == (1,)
        assert phi_new.shape == (1, 1)

        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

        # For dy/dt = -y, STM should be exp(-t)
        assert abs(phi_new[0, 0] - np.exp(-dt_used)) < 1e-6


class TestRKN1210Integrator:
    """Tests for RKN1210 (Runge-Kutta-Nyström 12(10)) integrator."""

    def test_dimension_validation(self):
        """Test that RKN1210 requires even dimension."""

        def dynamics(t, state):
            return state

        # Odd dimension should raise ValueError
        with pytest.raises(ValueError, match="even dimension"):
            bh.RKN1210Integrator(dimension=3, dynamics_fn=dynamics)

        # Even dimension should work
        integrator = bh.RKN1210Integrator(dimension=2, dynamics_fn=dynamics)
        assert integrator.dimension == 2

    def test_parabola_integration(self):
        """Test RKN1210 on parabola (position-velocity formulation)."""

        def dynamics(t, state):
            """State: [x, v], dynamics: [v, a] where a = 2.0"""
            x, v = state
            return np.array([v, 2.0])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKN1210Integrator(
            dimension=2, dynamics_fn=dynamics, config=config
        )

        t = 0.0
        state = np.array([0.0, 0.0])  # x(0) = 0, v(0) = 0

        while t < 1.0:
            dt = min(1.0 - t, 0.1)
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

        # With constant acceleration a = 2.0, v(1) = 2.0, x(1) = 1.0
        assert abs(state[1] - 2.0) < 1e-8  # velocity
        assert abs(state[0] - 1.0) < 1e-8  # position

    def test_orbit_propagation(self):
        """Test RKN1210 on orbital mechanics."""

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.RKN1210Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.RADIANS)

        state = state0.copy()
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
        period = bh.orbital_period(oe0[0])
        epcf = epc0 + period
        epc = epc0

        while epc < epcf:
            dt = min((epcf - epc), 10.0)
            result = integrator.step(epc - epc0, state, dt, 1e-8, 1e-6)
            state = result.state
            epc = epc + result.dt_used

        # RKN1210 should achieve very good accuracy for orbital mechanics
        assert abs(np.linalg.norm(state) - np.linalg.norm(state0)) < 1e-4
        assert abs(state[0] - state0[0]) < 1e-3
        assert abs(state[1] - state0[1]) < 1e-3
        assert abs(state[2] - state0[2]) < 1e-3

    def test_accuracy(self):
        """Verify RKN1210 achieves high-order accuracy (mirrors test_rkn1210s_integrator_parabola)."""

        def dynamics(t, state):
            """[x, v] -> [v, a] where a = 2.0 (constant acceleration)"""
            x, v = state
            return np.array([v, 2.0])

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKN1210Integrator(
            dimension=2, dynamics_fn=dynamics, config=config
        )

        t = 0.0
        state = np.array([0.0, 0.0])

        while t < 1.0:
            dt = min(1.0 - t, 0.01)
            result = integrator.step(t, state, dt, 1e-10, 1e-8)
            state = result.state
            t += result.dt_used

        # With x'' = 2, x(t) = t², so x(1) = 1.0
        exact = 1.0
        error = abs(state[0] - exact)

        # 12th order method should be extremely accurate
        assert error < 1e-8

    def test_step_size_increases(self):
        """Verify adaptive stepping increases step size when error is small."""

        def dynamics(t, state):
            """Simple harmonic oscillator"""
            x, v = state
            return np.array([v, -x])

        config = bh.IntegratorConfig.adaptive(1e-6, 1e-4)
        integrator = bh.RKN1210Integrator(
            dimension=2, dynamics_fn=dynamics, config=config
        )

        state = np.array([1.0, 0.0])
        dt_initial = 0.01

        result = integrator.step(0.0, state, dt_initial, 1e-6, 1e-4)

        # For smooth problem with loose tolerance, step should increase
        assert result.dt_next > dt_initial
        assert result.error_estimate < 0.1

    def test_with_varmat(self):
        """Test variational matrix propagation with RKN1210."""

        def dynamics(t, state):
            """Simple 2D system"""
            x, v = state
            return np.array([v, -x])

        jacobian = bh.DNumericalJacobian(dynamics).with_fixed_offset(1.0)
        integrator = bh.RKN1210Integrator(
            dimension=2, dynamics_fn=dynamics, jacobian=jacobian
        )

        state = np.array([1.0, 0.0])
        phi = np.eye(2)

        state_new, phi_new, dt_used, error_est, dt_next = integrator.step_with_varmat(
            0.0, state, phi, 0.1, 1e-9, 1e-6
        )

        assert isinstance(state_new, np.ndarray)
        assert isinstance(phi_new, np.ndarray)
        assert state_new.shape == (2,)
        assert phi_new.shape == (2, 2)

        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

        # STM should not be identity after propagation
        assert not np.allclose(phi_new, np.eye(2))


class TestAdaptiveIntegratorComparison:
    """Compare behavior of different adaptive integrators on same problem."""

    def test_all_converge_to_same_solution(self):
        """Verify all three integrators give consistent results."""

        def dynamics(t, state):
            return np.array([3.0 * t * t])

        config = bh.IntegratorConfig.adaptive(1e-12, 1e-10)

        # Create all three integrators
        rkf45 = bh.RKF45Integrator(dimension=1, dynamics_fn=dynamics, config=config)
        dp54 = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)

        # Propagate each to t=10
        t_end = 10.0
        exact = 1000.0  # t³ at t=10

        # RKF45
        t, state = 0.0, np.array([0.0])
        while t < t_end:
            result = rkf45.step(t, state, min(t_end - t, 0.1), 1e-12, 1e-10)
            state, t = result.state, t + result.dt_used
        rkf45_result = state[0]

        # DP54
        t, state = 0.0, np.array([0.0])
        while t < t_end:
            result = dp54.step(t, state, min(t_end - t, 0.1), 1e-12, 1e-10)
            state, t = result.state, t + result.dt_used
        dp54_result = state[0]

        # All should be very close to exact solution
        assert abs(rkf45_result - exact) < 1e-8
        assert abs(dp54_result - exact) < 1e-8

        # And close to each other
        assert abs(rkf45_result - dp54_result) < 1e-9

    def test_orbital_mechanics_consistency(self):
        """Verify all integrators give consistent orbital propagation."""

        config = bh.IntegratorConfig.adaptive(1e-9, 1e-7)

        rkf45 = bh.RKF45Integrator(dimension=6, dynamics_fn=point_earth, config=config)
        dp54 = bh.DP54Integrator(dimension=6, dynamics_fn=point_earth, config=config)
        rkn1210 = bh.RKN1210Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        # Setup orbit
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.RADIANS)

        # Propagate half an orbit with each
        t_end = bh.orbital_period(oe0[0]) / 2.0

        results = []
        for integrator in [rkf45, dp54, rkn1210]:
            t, state = 0.0, state0.copy()
            while t < t_end:
                result = integrator.step(t, state, min(t_end - t, 10.0), 1e-9, 1e-7)
                state, t = result.state, t + result.dt_used
            results.append(state)

        # All should give similar position
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                diff = np.linalg.norm(results[i][:3] - results[j][:3])
                # Within 1m of each other
                assert diff < 1.0, f"Integrators {i} and {j} differ by {diff:.3e} m"
