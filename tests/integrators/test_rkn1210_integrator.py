"""
Tests for RKN1210 (Runge-Kutta-Nyström 12(10)) integrator - mirrors Rust tests.
"""

import pytest
import numpy as np
import brahe as bh


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
            result = integrator.step(t, state, dt)
            state = result.state
            t += result.dt_used

        # With constant acceleration a = 2.0, v(1) = 2.0, x(1) = 1.0
        assert abs(state[1] - 2.0) < 1e-8  # velocity
        assert abs(state[0] - 1.0) < 1e-8  # position

    def test_orbit_propagation(self, point_earth):
        """Test RKN1210 on orbital mechanics."""

        config = bh.IntegratorConfig.adaptive(1e-8, 1e-6)
        integrator = bh.RKN1210Integrator(
            dimension=6, dynamics_fn=point_earth, config=config
        )

        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_koe_to_eci(oe0, bh.AngleFormat.RADIANS)

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
            result = integrator.step(t, state, dt)
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

        result = integrator.step(0.0, state, dt_initial)

        # For smooth problem with loose tolerance, step should increase
        assert result.dt_next > dt_initial
        assert result.error_estimate < 0.1

    def test_with_varmat(self):
        """Test variational matrix propagation with RKN1210."""

        def dynamics(t, state):
            """Simple 2D system"""
            x, v = state
            return np.array([v, -x])

        jacobian = bh.NumericalJacobian(dynamics).with_fixed_offset(1.0)
        integrator = bh.RKN1210Integrator(
            dimension=2, dynamics_fn=dynamics, jacobian=jacobian
        )

        state = np.array([1.0, 0.0])
        phi = np.eye(2)

        state_new, phi_new, dt_used, error_est, dt_next = integrator.step_with_varmat(
            0.0, state, phi, 0.1
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

    def test_with_sensmat(self):
        """Test sensitivity matrix propagation with RKN1210."""

        def dynamics_with_params(t, state, params):
            """Harmonic oscillator with parameter k: x' = v, v' = -k*x"""
            x, v = state
            k = params[0]
            return np.array([v, -k * x])

        def analytical_sensitivity(t, state, params):
            """Sensitivity: d/dk(-k*x) = -x for the acceleration component"""
            x, v = state
            return np.array([[0.0], [-x]])

        jacobian = bh.NumericalJacobian(
            lambda t, s: dynamics_with_params(t, s, np.array([1.0]))
        ).with_fixed_offset(1e-6)
        sensitivity = bh.AnalyticSensitivity(analytical_sensitivity)

        integrator = bh.RKN1210Integrator(
            dimension=2,
            dynamics_fn=lambda t, s: dynamics_with_params(t, s, np.array([1.0])),
            jacobian=jacobian,
            sensitivity=sensitivity,
        )

        state = np.array([1.0, 0.0])
        sens = np.zeros((2, 1))
        params = np.array([1.0])

        new_state, new_sens, dt_used, error_est, dt_next = integrator.step_with_sensmat(
            0.0, state, sens, params, 0.1
        )

        assert new_state.shape == (2,)
        assert new_sens.shape == (2, 1)
        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

    def test_with_varmat_sensmat(self):
        """Test combined STM and sensitivity matrix propagation with RKN1210."""

        def dynamics_with_params(t, state, params):
            x, v = state
            k = params[0]
            return np.array([v, -k * x])

        def analytical_sensitivity(t, state, params):
            x, v = state
            return np.array([[0.0], [-x]])

        jacobian = bh.NumericalJacobian(
            lambda t, s: dynamics_with_params(t, s, np.array([1.0]))
        ).with_fixed_offset(1e-6)
        sensitivity = bh.AnalyticSensitivity(analytical_sensitivity)

        integrator = bh.RKN1210Integrator(
            dimension=2,
            dynamics_fn=lambda t, s: dynamics_with_params(t, s, np.array([1.0])),
            jacobian=jacobian,
            sensitivity=sensitivity,
        )

        state = np.array([1.0, 0.0])
        phi = np.eye(2)
        sens = np.zeros((2, 1))
        params = np.array([1.0])

        result = integrator.step_with_varmat_sensmat(0.0, state, phi, sens, params, 0.1)
        new_state, new_phi, new_sens, dt_used, error_est, dt_next = result

        assert new_state.shape == (2,)
        assert new_phi.shape == (2, 2)
        assert new_sens.shape == (2, 1)
        assert dt_used > 0
        assert error_est >= 0
        assert dt_next > 0

    def test_backward_integration(self, point_earth):
        """Test backward propagation with orbital mechanics."""

        config = bh.IntegratorConfig.adaptive(1e-10, 1e-8)
        integrator = bh.RKN1210Integrator(
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
