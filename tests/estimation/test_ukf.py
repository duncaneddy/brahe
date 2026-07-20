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
    assert config.alpha == pytest.approx(1e-3)
    assert config.beta == pytest.approx(2.0)
    assert config.kappa == pytest.approx(0.0)
    assert config.store_records is True


def test_ukf_config_custom_all_params():
    """UKFConfig with all custom parameters."""
    q = np.diag([1e-6] * 3 + [1e-8] * 3)
    pn = bh.ProcessNoiseConfig(q)
    config = bh.UKFConfig(
        alpha=0.5,
        beta=3.0,
        kappa=1.0,
        process_noise=pn,
        store_records=False,
    )

    assert config.alpha == pytest.approx(0.5)
    assert config.beta == pytest.approx(3.0)
    assert config.kappa == pytest.approx(1.0)
    assert config.store_records is False


def test_ukf_with_process_noise(two_body_leo):
    """UKF with process noise configuration."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    q = np.diag([1e-4] * 3 + [1e-6] * 3)
    pn = bh.ProcessNoiseConfig(q, scale_with_dt=True)
    config = bh.UKFConfig(process_noise=pn, store_records=True)

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


def test_ukf_invalid_sigma_parameters_raise(two_body_leo):
    """Non-finite or degenerate sigma-point parameters must fail at
    construction rather than silently producing NaN estimates."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    for bad_config in [
        bh.UKFConfig(alpha=float("nan")),
        bh.UKFConfig(kappa=float("nan")),
        bh.UKFConfig(beta=float("nan")),
        bh.UKFConfig(alpha=0.0),
        bh.UKFConfig(kappa=-6.0),  # state_dim + kappa == 0 for a 6D state
    ]:
        with pytest.raises(RuntimeError):
            bh.UnscentedKalmanFilter(
                epoch,
                state,
                p0,
                measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
                propagation_config=bh.NumericalPropagationConfig.default(),
                force_config=bh.ForceModelConfig.two_body(),
                config=bad_config,
            )


def test_ukf_propagate_to(two_body_leo):
    """propagate_to advances the filter without a measurement update."""
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

    p0_trace = np.trace(ukf.current_covariance())
    record = ukf.propagate_to(epoch + 600.0)

    assert ukf.current_epoch() == epoch + 600.0
    assert record.measurement_name == "Propagation"
    assert len(record.prefit_residual) == 0
    assert len(record.postfit_residual) == 0
    assert record.kalman_gain.shape == (0, 0)
    np.testing.assert_array_equal(record.state_updated, record.state_predicted)
    np.testing.assert_array_equal(
        record.covariance_updated, record.covariance_predicted
    )

    # Covariance grows without measurements (Keplerian shear stretches the
    # along-track uncertainty even with no explicit process noise).
    assert np.trace(ukf.current_covariance()) > p0_trace

    # Backwards propagation is supported (e.g. for smoothing): the filter
    # epoch moves back to the target.
    ukf.propagate_to(epoch)
    assert ukf.current_epoch() == epoch


def test_ukf_propagate_to_store_records_and_process_noise(two_body_leo):
    """propagate_to gates records on store_records and applies Q*dt (and no
    process noise on a zero-duration step)."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    dt = 600.0
    q_diag = [1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]
    q = np.diag(q_diag)
    q_trace = sum(q_diag)

    def make(config):
        return bh.UnscentedKalmanFilter(
            epoch,
            state.copy(),
            p0,
            measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            config=config,
        )

    # store_records gating: records() grows by one per call when enabled.
    rec = make(bh.UKFConfig(store_records=True))
    assert len(rec.records()) == 0
    rec.propagate_to(epoch + 300.0)
    assert len(rec.records()) == 1
    rec.propagate_to(epoch + 600.0)
    assert len(rec.records()) == 2

    norec = make(bh.UKFConfig(store_records=False))
    norec.propagate_to(epoch + 300.0)
    assert len(norec.records()) == 0

    # process noise: Q*dt increment over the no-noise baseline.
    plain = make(bh.UKFConfig())
    noisy = make(
        bh.UKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))
    )
    plain.propagate_to(epoch + dt)
    noisy.propagate_to(epoch + dt)
    increment = np.trace(noisy.current_covariance()) - np.trace(
        plain.current_covariance()
    )
    assert increment == pytest.approx(q_trace * dt, rel=0.05)

    # A zero-duration propagate_to adds no process noise.
    before = np.trace(noisy.current_covariance())
    noisy.propagate_to(noisy.current_epoch())
    after = np.trace(noisy.current_covariance())
    assert after == pytest.approx(before, rel=1e-9)


class NegativeNoiseModel(bh.MeasurementModel):
    """Position model with a strongly negative-definite noise covariance R,
    driving the unscented innovation covariance S non-positive-definite."""

    def predict(self, epoch, state, params=None):
        return np.array(state[:3])

    def noise_covariance(self):
        return -1e9 * np.eye(3)

    def measurement_dim(self):
        return 3

    def name(self):
        return "NegativeNoise"


def _make_ukf(epoch, state, p0, models, config=None):
    return bh.UnscentedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=models,
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        config=config if config is not None else bh.UKFConfig.default(),
    )


def test_ukf_model_index_out_of_bounds(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    ukf = _make_ukf(epoch, state, p0, [bh.InertialPositionMeasurementModel(10.0)])
    obs = bh.Observation(epoch + 60.0, state[:3], model_index=5)
    with pytest.raises(RuntimeError, match="out of bounds"):
        ukf.process_observation(obs)


def test_ukf_non_positive_definite_covariance(two_body_leo):
    """The constructor validates only dimensions; a negative-diagonal initial
    covariance surfaces at the first sigma-point Cholesky as an error."""
    epoch, state = two_body_leo
    bad_p0 = np.diag([-1.0, 1e6, 1e6, 1e2, 1e2, 1e2])
    ukf = _make_ukf(epoch, state, bad_p0, [bh.InertialPositionMeasurementModel(10.0)])
    obs = bh.Observation(epoch + 60.0, state[:3], model_index=0)
    with pytest.raises(RuntimeError, match="positive-definite"):
        ukf.process_observation(obs)


def test_ukf_innovation_covariance_not_positive_definite(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    ukf = _make_ukf(epoch, state, p0, [NegativeNoiseModel()])
    obs = bh.Observation(epoch + 60.0, state[:3], model_index=0)
    with pytest.raises(RuntimeError, match="positive-definite"):
        ukf.process_observation(obs)
    assert ukf.current_epoch() == epoch


def test_ukf_observation_process_noise_both_branches(two_body_leo):
    """The sigma-point predict step adds Q for both scale_with_dt True (Q*dt)
    and False (Q); the predicted-covariance trace difference vs a no-noise twin
    equals the Q contribution."""
    epoch, true_state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    dt = 60.0
    q_diag = [1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]
    q = np.diag(q_diag)
    q_trace = sum(q_diag)

    def make(pn):
        config = (
            bh.UKFConfig(process_noise=pn) if pn is not None else bh.UKFConfig.default()
        )
        return _make_ukf(
            epoch,
            true_state.copy(),
            p0,
            [bh.InertialPositionMeasurementModel(10.0)],
            config,
        )

    obs = bh.Observation(epoch + dt, true_state[:3], model_index=0)

    trace_none = np.trace(make(None).process_observation(obs).covariance_predicted)

    rec_scaled = make(bh.ProcessNoiseConfig(q, scale_with_dt=True)).process_observation(
        obs
    )
    assert np.trace(rec_scaled.covariance_predicted) - trace_none == pytest.approx(
        q_trace * dt, rel=1e-6
    )

    rec_fixed = make(bh.ProcessNoiseConfig(q, scale_with_dt=False)).process_observation(
        obs
    )
    assert np.trace(rec_fixed.covariance_predicted) - trace_none == pytest.approx(
        q_trace, rel=1e-6
    )


def test_ukf_builder_equivalence(two_body_leo):
    """Builder-constructed UKF should behave identically to the flat constructor."""
    epoch, true_state = two_body_leo
    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    via_builder = (
        bh.UnscentedKalmanFilter.builder(
            epoch,
            initial_state,
            p0,
            bh.ForceModelConfig.two_body(),
            bh.UKFConfig.default(),
        )
        .propagation_config(bh.NumericalPropagationConfig.default())
        .measurement_model(bh.InertialPositionMeasurementModel(10.0))
        .build()
    )

    via_constructor = bh.UnscentedKalmanFilter(
        epoch,
        initial_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    obs = bh.Observation(epoch + 60.0, true_state[:3], model_index=0)
    via_builder.process_observation(obs)
    via_constructor.process_observation(obs)

    np.testing.assert_allclose(
        via_builder.current_state(), via_constructor.current_state()
    )
    np.testing.assert_allclose(
        via_builder.current_covariance(), via_constructor.current_covariance()
    )


def test_ukf_builder_unchained_setter(two_body_leo):
    """Calling a setter without reassigning its return value must not orphan
    the original builder variable -- build() on the original must succeed."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    builder = bh.UnscentedKalmanFilter.builder(
        epoch,
        state,
        p0,
        bh.ForceModelConfig.two_body(),
        bh.UKFConfig.default(),
    ).measurement_model(bh.InertialPositionMeasurementModel(10.0))
    builder.propagation_config(
        bh.NumericalPropagationConfig.default()
    )  # not reassigned
    ukf = builder.build()

    assert len(ukf.current_state()) == 6


def test_ukf_builder_double_build_raises(two_body_leo):
    """Calling build() twice on the same builder should raise RuntimeError."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    builder = bh.UnscentedKalmanFilter.builder(
        epoch,
        state,
        p0,
        bh.ForceModelConfig.two_body(),
        bh.UKFConfig.default(),
    ).measurement_model(bh.InertialPositionMeasurementModel(10.0))
    builder.build()

    with pytest.raises(RuntimeError, match="builder already consumed"):
        builder.build()


def test_ukf_builder_build_failure_raises(two_body_leo):
    """Degenerate sigma-point parameters must fail at build() with RuntimeError."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    builder = bh.UnscentedKalmanFilter.builder(
        epoch,
        state,
        p0,
        bh.ForceModelConfig.two_body(),
        bh.UKFConfig(alpha=0.0),
    ).measurement_model(bh.InertialPositionMeasurementModel(10.0))

    with pytest.raises(RuntimeError):
        builder.build()
