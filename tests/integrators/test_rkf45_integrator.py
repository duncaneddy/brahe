"""
Tests for RKF45 (Runge-Kutta-Fehlberg 4(5)) integrator - mirrors Rust tests.
"""

import numpy as np
import brahe as bh


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
            result = integrator.step(t, state, dt)
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
            result = integrator.step(t, state, dt)
            state = result.state
            t += result.dt_used

            # Verify error estimate is non-negative
            assert result.error_estimate >= 0.0

        # Should still get accurate result
        assert abs(state[0] - 1.0) < 1e-8

    def test_orbit_propagation(self, point_earth):
        """Test RKF45 on orbital mechanics (mirrors test_rkf45d_integrator_orbit)."""

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.RKF45Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        # Initial orbital elements: LEO, e=0.01, i=90°
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_koe_to_eci(oe0, bh.AngleFormat.RADIANS)

        # Propagate for one orbital period
        state = state0.copy()
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
        period = bh.orbital_period(oe0[0])
        epcf = epc0 + period
        epc = epc0

        while epc < epcf:
            dt = min((epcf - epc), 10.0)
            result = integrator.step(epc - epc0, state, dt)
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
            result = integrator.step(t, state, dt)
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
        result = integrator.step(0.0, state, dt_initial)

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

        result = integrator.step(0.0, np.array([0.0]), 0.1)

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

        jacobian = bh.NumericalJacobian(dynamics).with_fixed_offset(1.0)
        integrator = bh.RKF45Integrator(
            dimension=1, dynamics_fn=dynamics, jacobian=jacobian
        )

        state = np.array([1.0])
        phi = np.eye(1)

        state_new, phi_new, dt_used, error_est, dt_next = integrator.step_with_varmat(
            0.0, state, phi, 0.1
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

    def test_with_sensmat(self):
        """Test sensitivity matrix propagation."""

        def dynamics_with_params(t, state, params):
            """Exponential decay: dx/dt = -k*x where k = params[0]"""
            k = params[0]
            return np.array([-k * state[0]])

        def analytical_sensitivity(t, state, params):
            """Analytical sensitivity: d/dp(-k*x) = -x"""
            return np.array([[-state[0]]])

        jacobian = bh.NumericalJacobian(
            lambda t, s: dynamics_with_params(t, s, np.array([1.0]))
        ).with_fixed_offset(1e-6)
        sensitivity = bh.AnalyticSensitivity(analytical_sensitivity)

        integrator = bh.RKF45Integrator(
            dimension=1,
            dynamics_fn=lambda t, s: dynamics_with_params(t, s, np.array([1.0])),
            jacobian=jacobian,
            sensitivity=sensitivity,
        )

        state = np.array([1.0])
        sens = np.zeros((1, 1))
        params = np.array([1.0])

        new_state, new_sens, dt_used, error_est, dt_next = integrator.step_with_sensmat(
            0.0, state, sens, params, 0.1
        )

        assert isinstance(new_state, np.ndarray)
        assert isinstance(new_sens, np.ndarray)
        assert new_state.shape == (1,)
        assert new_sens.shape == (1, 1)
        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

        # State should have decayed
        assert new_state[0] < 1.0
        assert new_state[0] > 0.0

        # Sensitivity should have evolved
        assert new_sens[0, 0] != 0.0

    def test_with_varmat_sensmat(self):
        """Test combined STM and sensitivity matrix propagation."""

        def dynamics_with_params(t, state, params):
            k = params[0]
            return np.array([-k * state[0]])

        def analytical_sensitivity(t, state, params):
            return np.array([[-state[0]]])

        jacobian = bh.NumericalJacobian(
            lambda t, s: dynamics_with_params(t, s, np.array([1.0]))
        ).with_fixed_offset(1e-6)
        sensitivity = bh.AnalyticSensitivity(analytical_sensitivity)

        integrator = bh.RKF45Integrator(
            dimension=1,
            dynamics_fn=lambda t, s: dynamics_with_params(t, s, np.array([1.0])),
            jacobian=jacobian,
            sensitivity=sensitivity,
        )

        state = np.array([1.0])
        phi = np.eye(1)
        sens = np.zeros((1, 1))
        params = np.array([1.0])

        result = integrator.step_with_varmat_sensmat(0.0, state, phi, sens, params, 0.1)
        new_state, new_phi, new_sens, dt_used, error_est, dt_next = result

        assert new_state.shape == (1,)
        assert new_phi.shape == (1, 1)
        assert new_sens.shape == (1, 1)
        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

    def test_backward_integration(self, point_earth):
        """Test backward propagation with orbital mechanics (mirrors test_rkf45d_backward_integration)."""

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKF45Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        # Setup initial state
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_koe_to_eci(oe0, bh.AngleFormat.RADIANS)

        # Propagate forward for 100 seconds
        dt_forward = 10.0
        state_fwd = state0.copy()
        t = 0.0
        while t < 100.0:
            result = integrator.step(t, state_fwd, dt_forward)
            state_fwd = result.state
            t += result.dt_used
        t_max = t

        # Now propagate backward from the final state
        state_back = state_fwd.copy()
        t = t_max
        dt_back = -10.0  # Initial negative timestep for backward integration
        while t > 0.0:
            # Ensure we don't step past t=0
            dt_back = max(dt_back, -t)
            result = integrator.step(t, state_back, dt_back)
            state_back = result.state
            t += result.dt_used
            dt_back = result.dt_next  # Use adaptive timestep suggestion

        # Should return close to initial state
        for i in range(6):
            assert abs(state_back[i] - state0[i]) < 1e-3
