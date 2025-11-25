"""
Tests comparing behavior of different adaptive integrators on same problems.
"""

import numpy as np
import brahe as bh


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
            result = rkf45.step(t, state, min(t_end - t, 0.1))
            state, t = result.state, t + result.dt_used
        rkf45_result = state[0]

        # DP54
        t, state = 0.0, np.array([0.0])
        while t < t_end:
            result = dp54.step(t, state, min(t_end - t, 0.1))
            state, t = result.state, t + result.dt_used
        dp54_result = state[0]

        # All should be very close to exact solution
        assert abs(rkf45_result - exact) < 1e-8
        assert abs(dp54_result - exact) < 1e-8

        # And close to each other
        assert abs(rkf45_result - dp54_result) < 1e-9

    def test_orbital_mechanics_consistency(self, point_earth):
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
                result = integrator.step(t, state, min(t_end - t, 10.0))
                state, t = result.state, t + result.dt_used
            results.append(state)

        # All should give similar position
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                diff = np.linalg.norm(results[i][:3] - results[j][:3])
                # Within 1m of each other
                assert diff < 1.0, f"Integrators {i} and {j} differ by {diff:.3e} m"
