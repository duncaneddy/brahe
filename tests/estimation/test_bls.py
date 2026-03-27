"""Tests for BatchLeastSquares Python bindings."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture
def two_body_leo():
    """Two-body LEO orbit: circular, equatorial, ~500km alt."""
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    r = 6878.0e3
    v = (bh.GM_EARTH / r) ** 0.5
    state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    return epoch, state


@pytest.fixture
def position_observations(two_body_leo):
    """Generate noise-free inertial position observations."""
    epoch, true_state = two_body_leo
    prop = bh.NumericalOrbitPropagator(
        epoch,
        true_state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )
    observations = []
    for i in range(1, 21):
        t = epoch + i * 30.0
        prop.propagate_to(t)
        pos = prop.current_state()[:3]
        observations.append(bh.Observation(t, pos, 0))
    return observations


@pytest.fixture
def bls_setup(two_body_leo, position_observations):
    """Create a basic BLS with position measurements for testing."""
    epoch, true_state = two_body_leo
    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    initial_state[1] += 500.0

    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    bls = bh.BatchLeastSquares(
        epoch,
        initial_state,
        p0,
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    )

    return bls, epoch, true_state, initial_state, position_observations


class TestBatchLeastSquares:
    def test_construction(self, two_body_leo):
        epoch, state = two_body_leo
        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

        bls = bh.BatchLeastSquares(
            epoch,
            state,
            p0,
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        )

        assert len(bls.current_state()) == 6
        assert bls.current_covariance().shape == (6, 6)
        assert not bls.converged()
        assert bls.iterations_completed() == 0

    def test_repr(self, bls_setup):
        bls, _, _, _, _ = bls_setup
        r = repr(bls)
        assert "BatchLeastSquares" in r

    def test_convergence_normal_equations(self, bls_setup):
        bls, epoch, true_state, initial_state, observations = bls_setup
        bls.solve(observations)

        assert bls.converged()
        assert bls.iterations_completed() > 0

        pos_error = np.linalg.norm(bls.current_state()[:3] - true_state[:3])
        initial_error = np.linalg.norm(initial_state[:3] - true_state[:3])

        assert pos_error < initial_error * 0.001

    def test_convergence_stacked(self, two_body_leo, position_observations):
        epoch, true_state = two_body_leo
        initial_state = true_state.copy()
        initial_state[0] += 1000.0

        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
        config = bh.BLSConfig(
            solver_method=bh.BLSSolverMethod.STACKED_OBSERVATION_MATRIX,
        )

        bls = bh.BatchLeastSquares(
            epoch,
            initial_state,
            p0,
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
            config=config,
        )
        bls.solve(position_observations)

        assert bls.converged()
        pos_error = np.linalg.norm(bls.current_state()[:3] - true_state[:3])
        assert pos_error < 1.0  # < 1m

    def test_iteration_records(self, bls_setup):
        bls, _, _, _, observations = bls_setup
        bls.solve(observations)

        records = bls.iteration_records()
        assert len(records) > 0

        for i, rec in enumerate(records):
            assert rec.iteration == i
            assert len(rec.state) == 6
            assert rec.covariance.shape == (6, 6)
            assert rec.cost >= 0

    def test_observation_residuals(self, bls_setup):
        bls, _, _, _, observations = bls_setup
        bls.solve(observations)

        obs_res = bls.observation_residuals()
        assert len(obs_res) > 0

        for iter_res in obs_res:
            assert len(iter_res) == 20
            for r in iter_res:
                assert len(r.prefit_residual) == 3
                assert len(r.postfit_residual) == 3
                assert r.model_name == "InertialPosition"

    def test_custom_measurement_model(self, two_body_leo, position_observations):
        """BLS should work with Python-defined custom measurement models."""
        epoch, true_state = two_body_leo

        class PositionModel(bh.MeasurementModel):
            def predict(self, epoch, state):
                return state[:3]

            def noise_covariance(self):
                return np.eye(3) * 100.0

            def measurement_dim(self):
                return 3

            def name(self):
                return "CustomPosition"

        initial_state = true_state.copy()
        initial_state[0] += 500.0

        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

        bls = bh.BatchLeastSquares(
            epoch,
            initial_state,
            p0,
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            measurement_models=[PositionModel()],
        )
        bls.solve(position_observations)

        assert bls.converged()
