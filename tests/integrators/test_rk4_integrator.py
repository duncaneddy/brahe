"""
Tests for RK4 (4th order Runge-Kutta) integrator Python bindings.
"""

import numpy as np
import brahe as bh


class TestIntegratorConfig:
    """Tests for IntegratorConfig class."""

    def test_fixed_step_config(self):
        """Test creating fixed-step configuration."""
        config = bh.IntegratorConfig.fixed_step(0.1)
        assert config is not None

    def test_adaptive_config(self):
        """Test creating adaptive-step configuration."""
        config = bh.IntegratorConfig.adaptive(1e-9, 1e-6)
        assert config is not None
        assert config.abs_tol == 1e-9
        assert config.rel_tol == 1e-6

    def test_new_with_all_parameters(self):
        """Test creating configuration with all parameters."""
        config = bh.IntegratorConfig(
            abs_tol=1e-10,
            rel_tol=1e-7,
            initial_step=0.5,
            min_step=1e-6,
            max_step=10.0,
            step_safety_factor=0.9,
            min_step_scale_factor=0.2,
            max_step_scale_factor=5.0,
            max_step_attempts=10,
        )
        assert config is not None

        # Test getters (properties, not methods)
        assert config.abs_tol == 1e-10
        assert config.rel_tol == 1e-7
        assert config.initial_step == 0.5
        assert config.min_step == 1e-6
        assert config.max_step == 10.0
        assert config.step_safety_factor == 0.9
        assert config.min_step_scale_factor == 0.2
        assert config.max_step_scale_factor == 5.0
        assert config.max_step_attempts == 10


class TestRK4Integrator:
    """Tests for RK4Integrator class - mirrors Rust tests."""

    def test_cubic_integration(self):
        """Test integration of x' = 3t² (mirrors test_rk4d_integrator_cubic)."""

        def dynamics(t, state):
            """x' = 3t²"""
            return np.array([3.0 * t * t])

        integrator = bh.RK4Integrator(dimension=1, dynamics_fn=dynamics)

        state = np.array([0.0])
        dt = 1.0

        # Integrate from t=0 to t=10 in steps of 1.0
        for i in range(10):
            state = integrator.step(float(i), state, dt)

        # Exact solution: x(t) = t³
        # At t=10: x = 1000
        assert abs(state[0] - 1000.0) < 1e-12

    def test_parabola_integration(self):
        """Test integration of x' = 2t (mirrors test_rk4d_integrator_parabola)."""

        def dynamics(t, state):
            """x' = 2t"""
            return np.array([2.0 * t])

        integrator = bh.RK4Integrator(dimension=1, dynamics_fn=dynamics)

        t = 0.0
        state = np.array([0.0])
        dt = 0.01

        # Integrate from t=0 to t=1.0 in steps of 0.01
        for _ in range(100):
            state = integrator.step(t, state, dt)
            t += dt

        # Exact solution: x(t) = t²
        # At t=1: x = 1.0
        assert abs(state[0] - 1.0) < 1e-12

    def test_orbit_propagation(self):
        """Test orbital propagation for one period (mirrors test_rk4d_integrator_orbit)."""

        def point_earth(t, state):
            """Two-body point mass Earth dynamics."""
            r = state[:3]
            v = state[3:]

            r_norm = np.linalg.norm(r)
            a_mag = -bh.GM_EARTH / (r_norm**3)
            a = a_mag * r

            return np.concatenate([v, a])

        integrator = bh.RK4Integrator(dimension=6, dynamics_fn=point_earth)

        # Initial orbital elements: LEO, e=0.01, i=90°
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.DEGREES)

        # Propagate for one orbital period
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
        period = bh.orbital_period(oe0[0])
        epcf = epc0 + period

        state = state0.copy()
        epc = epc0

        while epc < epcf:
            dt = min((epcf - epc), 1.0)
            state = integrator.step(epc - epc0, state, dt)
            epc = epc + dt

        # After one orbit, position should return to initial state
        # Check norm (energy conservation proxy)
        assert abs(np.linalg.norm(state) - np.linalg.norm(state0)) < 1e-7

        # Check individual components
        assert abs(state[0] - state0[0]) < 1e-5
        assert abs(state[1] - state0[1]) < 1e-5
        assert abs(state[2] - state0[2]) < 1e-5
        assert abs(state[3] - state0[3]) < 1e-5
        assert abs(state[4] - state0[4]) < 1e-5
        assert abs(state[5] - state0[5]) < 1e-5

    def test_varmat_propagation(self):
        """Test variational matrix propagation (mirrors test_rk4d_integrator_varmat)."""

        def point_earth(t, state):
            """Two-body point mass Earth dynamics."""
            r = state[:3]
            v = state[3:]

            r_norm = np.linalg.norm(r)
            a_mag = -bh.GM_EARTH / (r_norm**3)
            a = a_mag * r

            return np.concatenate([v, a])

        # Create numerical Jacobian provider
        jacobian = bh.NumericalJacobian(point_earth).with_fixed_offset(1.0)

        integrator = bh.RK4Integrator(
            dimension=6, dynamics_fn=point_earth, jacobian=jacobian
        )

        # Initial orbital elements
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.DEGREES)
        phi0 = np.eye(6)

        # Test 1: Zero step should return identity matrix
        _, phi1 = integrator.step_with_varmat(0.0, state0, phi0, 0.0)
        np.testing.assert_allclose(phi1, np.eye(6), atol=1e-12)

        # Test 2: One-second step should show varmat evolution
        _, phi2 = integrator.step_with_varmat(0.0, state0, phi0, 1.0)

        # Diagonal elements should be close to 1 but not exactly 1
        for i in range(6):
            assert phi2[i, i] != 1.0
            assert phi2[i, i] != 0.0
            assert abs(phi2[i, i] - 1.0) < 1e-5

        # Off-diagonal elements should be populated (coupling between states)
        off_diag_populated = False
        for i in range(6):
            for j in range(6):
                if i != j and abs(phi2[i, j]) > 1e-10:
                    off_diag_populated = True
                    break
        assert off_diag_populated, (
            "Variational matrix should have off-diagonal elements"
        )

    def test_backward_integration(self):
        """Test backward propagation with orbital mechanics (mirrors test_rk4d_backward_integration)."""

        def point_earth(t, state):
            """Two-body point mass Earth dynamics."""
            r = state[:3]
            v = state[3:]

            r_norm = np.linalg.norm(r)
            a_mag = -bh.GM_EARTH / (r_norm**3)
            a = a_mag * r

            return np.concatenate([v, a])

        integrator = bh.RK4Integrator(dimension=6, dynamics_fn=point_earth)

        # Setup initial state
        oe0 = np.array([bh.R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0])
        state0 = bh.state_osculating_to_cartesian(oe0, bh.AngleFormat.DEGREES)

        # Propagate forward for 100 seconds with timestep 1 second
        dt_forward = 1.0
        state_fwd = state0.copy()
        for _ in range(100):
            state_fwd = integrator.step(0.0, state_fwd, dt_forward)

        # Now propagate backward from the final state
        dt_back = -1.0  # Negative timestep for backward integration
        state_back = state_fwd.copy()
        for _ in range(100):
            state_back = integrator.step(0.0, state_back, dt_back)

        # Should return close to initial state
        for i in range(6):
            assert abs(state_back[i] - state0[i]) < 1e-9
