"""Tests for UnscentedKalmanFilter Python bindings."""

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


def test_ukf_construction(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    assert len(ukf.current_state()) == 6
    assert ukf.current_covariance().shape == (6, 6)


def test_ukf_repr(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )
    assert "UnscentedKalmanFilter" in repr(ukf)


def test_ukf_converges_from_position_offset(two_body_leo):
    """UKF should converge toward truth with position measurements."""
    epoch, true_state = two_body_leo

    truth_prop = bh.NumericalOrbitPropagator(
        epoch,
        true_state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )

    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        initial_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    dt = 60.0
    for i in range(1, 21):
        obs_epoch = epoch + dt * i
        truth_prop.propagate_to(obs_epoch)
        truth_pos = truth_prop.current_state()[:3]
        obs = bh.Observation(obs_epoch, truth_pos, model_index=0)
        ukf.process_observation(obs)

    truth_prop.propagate_to(ukf.current_epoch())
    final_pos_error = np.linalg.norm(
        ukf.current_state()[:3] - truth_prop.current_state()[:3]
    )
    assert final_pos_error < 1000.0  # Less than initial 1km offset


def test_ukf_records_stored(two_body_leo):
    epoch, true_state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        true_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    obs = bh.Observation(epoch + 60.0, true_state[:3], model_index=0)
    record = ukf.process_observation(obs)

    assert len(record.state_predicted) == 6
    assert record.covariance_predicted.shape == (6, 6)
    assert len(record.prefit_residual) == 3
    assert len(record.postfit_residual) == 3
    assert record.measurement_name == "InertialPosition"
    assert len(ukf.records()) == 1


def test_ukf_process_observations_batch(two_body_leo):
    epoch, true_state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    truth_prop = bh.NumericalOrbitPropagator(
        epoch,
        true_state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        true_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    observations = []
    for i in range(1, 6):
        obs_epoch = epoch + 60.0 * i
        truth_prop.propagate_to(obs_epoch)
        observations.append(
            bh.Observation(obs_epoch, truth_prop.current_state()[:3], model_index=0)
        )

    ukf.process_observations(observations)
    assert len(ukf.records()) == 5


def test_ukf_with_config(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    config = bh.UKFConfig(alpha=1e-2, beta=2.0, kappa=0.0)

    ukf = bh.UnscentedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        config=config,
    )

    assert len(ukf.current_state()) == 6


def test_ukf_config_defaults():
    config = bh.UKFConfig.default()
    assert config.state_dim == 6
    assert config.alpha == pytest.approx(1e-3)
    assert config.beta == pytest.approx(2.0)
    assert config.kappa == pytest.approx(0.0)
    assert config.store_records is True
