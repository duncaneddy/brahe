"""Tests for ExtendedKalmanFilter Python bindings."""

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
def ekf_setup(two_body_leo):
    """Create a basic EKF with position measurements for testing."""
    epoch, true_state = two_body_leo

    # Perturbed initial state (1km position offset)
    initial_state = true_state.copy()
    initial_state[0] += 1000.0

    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        initial_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    return ekf, epoch, true_state


def test_ekf_construction(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    assert len(ekf.current_state()) == 6
    assert ekf.current_covariance() is not None
    assert ekf.current_covariance().shape == (6, 6)


def test_ekf_repr(ekf_setup):
    ekf, _, _ = ekf_setup
    r = repr(ekf)
    assert "ExtendedKalmanFilter" in r


def test_ekf_converges_from_position_offset(two_body_leo):
    """EKF should converge toward truth with perfect position measurements."""
    epoch, true_state = two_body_leo

    # Create a truth propagator to generate observations
    truth_prop = bh.NumericalOrbitPropagator(
        epoch,
        true_state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )

    # Perturbed initial state (1km position offset)
    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        initial_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    # Generate observations at 60s intervals using truth propagator
    dt = 60.0
    n_obs = 20

    for i in range(1, n_obs + 1):
        obs_epoch = epoch + dt * i
        truth_prop.propagate_to(obs_epoch)
        truth_pos = truth_prop.current_state()[:3]
        obs = bh.Observation(obs_epoch, truth_pos, model_index=0)
        ekf.process_observation(obs)

    # Position error should decrease from initial 1km offset
    final_state = ekf.current_state()
    truth_prop.propagate_to(ekf.current_epoch())
    truth_final = truth_prop.current_state()
    final_pos_error = np.linalg.norm(final_state[:3] - truth_final[:3])
    assert final_pos_error < 1000.0  # Less than initial 1km offset


def test_ekf_records_stored(ekf_setup):
    ekf, epoch, true_state = ekf_setup

    obs = bh.Observation(epoch + 60.0, true_state[:3], model_index=0)
    record = ekf.process_observation(obs)

    # Check record fields
    assert len(record.state_predicted) == 6
    assert record.covariance_predicted.shape == (6, 6)
    assert len(record.state_updated) == 6
    assert record.covariance_updated.shape == (6, 6)
    assert len(record.prefit_residual) == 3
    assert len(record.postfit_residual) == 3
    assert record.kalman_gain.shape[0] == 6
    assert record.kalman_gain.shape[1] == 3
    assert record.measurement_name == "InertialPosition"

    # Records stored
    assert len(ekf.records()) == 1


def test_ekf_process_observations_batch(ekf_setup):
    ekf, epoch, true_state = ekf_setup

    observations = [
        bh.Observation(epoch + 60.0 * i, true_state[:3], model_index=0)
        for i in range(1, 6)
    ]

    ekf.process_observations(observations)
    assert len(ekf.records()) == 5


def test_ekf_with_config(two_body_leo):
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    q = np.diag([1e-6] * 3 + [1e-8] * 3)
    pn = bh.ProcessNoiseConfig(q, scale_with_dt=True)
    config = bh.EKFConfig(process_noise=pn, store_records=True)

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        config=config,
    )

    assert len(ekf.current_state()) == 6


def test_ekf_with_process_noise_no_scale(two_body_leo):
    """EKF construction with process noise that doesn't scale with dt."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    q = np.diag([1e-6] * 3 + [1e-8] * 3)
    pn = bh.ProcessNoiseConfig(q, scale_with_dt=False)
    config = bh.EKFConfig(process_noise=pn, store_records=False)

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        config=config,
    )

    assert len(ekf.current_state()) == 6
    # Records should be empty since store_records=False
    assert len(ekf.records()) == 0


def test_ekf_with_multiple_measurement_models(two_body_leo):
    """EKF with both position and velocity measurement models."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[
            bh.InertialPositionMeasurementModel(10.0),
            bh.InertialVelocityMeasurementModel(0.1),
        ],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    assert len(ekf.current_state()) == 6

    # Process observations with different model indices
    obs_pos = bh.Observation(epoch + 60.0, state[:3], model_index=0)
    record = ekf.process_observation(obs_pos)
    assert record.measurement_name == "InertialPosition"

    obs_vel = bh.Observation(epoch + 120.0, state[3:6], model_index=1)
    record = ekf.process_observation(obs_vel)
    assert record.measurement_name == "InertialVelocity"


def test_ekf_propagate_to(two_body_leo):
    """propagate_to advances the filter without a measurement update."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    p0_trace = np.trace(ekf.current_covariance())
    record = ekf.propagate_to(epoch + 600.0)

    assert ekf.current_epoch() == epoch + 600.0
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
    assert np.trace(ekf.current_covariance()) > p0_trace

    # Backwards propagation is supported (e.g. for smoothing): the filter
    # epoch moves back to the target.
    ekf.propagate_to(epoch)
    assert ekf.current_epoch() == epoch


def test_ekf_propagate_to_store_records_and_process_noise(two_body_leo):
    """propagate_to gates records on store_records and applies Q*dt (and no
    process noise on a zero-duration step)."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    dt = 600.0
    q_diag = [1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]
    q = np.diag(q_diag)
    q_trace = sum(q_diag)

    def make(config):
        return bh.ExtendedKalmanFilter(
            epoch,
            state.copy(),
            p0,
            measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            config=config,
        )

    # store_records gating: records() grows by one per call when enabled.
    rec = make(bh.EKFConfig(store_records=True))
    assert len(rec.records()) == 0
    rec.propagate_to(epoch + 300.0)
    assert len(rec.records()) == 1
    rec.propagate_to(epoch + 600.0)
    assert len(rec.records()) == 2

    norec = make(bh.EKFConfig(store_records=False))
    norec.propagate_to(epoch + 300.0)
    assert len(norec.records()) == 0

    # process noise: Q*dt increment over the no-noise baseline.
    plain = make(bh.EKFConfig())
    noisy = make(
        bh.EKFConfig(process_noise=bh.ProcessNoiseConfig(q, scale_with_dt=True))
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
    driving the innovation covariance S non-positive-definite."""

    def predict(self, epoch, state, params=None):
        return np.array(state[:3])

    def noise_covariance(self):
        return -1e9 * np.eye(3)

    def measurement_dim(self):
        return 3

    def name(self):
        return "NegativeNoise"


def test_ekf_model_index_out_of_bounds(ekf_setup):
    """An observation tagged with an out-of-range model_index must error."""
    ekf, epoch, true_state = ekf_setup
    obs = bh.Observation(epoch + 60.0, true_state[:3], model_index=5)
    with pytest.raises(bh.BraheError, match="out of bounds"):
        ekf.process_observation(obs)


def test_ekf_innovation_covariance_not_positive_definite(two_body_leo):
    """A negative-definite R makes S non-positive-definite; the update must
    surface a structured error and roll the filter back."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    ekf = bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[NegativeNoiseModel()],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )
    obs = bh.Observation(epoch + 60.0, state[:3], model_index=0)
    with pytest.raises(bh.BraheError, match="positive-definite"):
        ekf.process_observation(obs)
    # Rolled back to the entry epoch.
    assert ekf.current_epoch() == epoch


def test_ekf_observation_process_noise_both_branches(two_body_leo):
    """The measurement predict step adds Q for both scale_with_dt True (Q*dt)
    and False (Q). The propagated covariance is identical across the three
    filters, so the predicted-covariance trace difference vs a no-noise twin is
    exactly the Q contribution."""
    epoch, true_state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    dt = 60.0
    q_diag = [1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]
    q = np.diag(q_diag)
    q_trace = sum(q_diag)

    def make(pn):
        config = bh.EKFConfig(process_noise=pn) if pn is not None else bh.EKFConfig()
        return bh.ExtendedKalmanFilter(
            epoch,
            true_state.copy(),
            p0,
            measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
            config=config,
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


def _substep_ekf(epoch, state, p0, q, max_noise_dt):
    pn = bh.ProcessNoiseConfig(q, scale_with_dt=True, max_noise_dt=max_noise_dt)
    return bh.ExtendedKalmanFilter(
        epoch,
        state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        config=bh.EKFConfig(process_noise=pn),
    )


def test_process_noise_config_max_noise_dt_roundtrip():
    q = np.diag([1e-6] * 3 + [1e-8] * 3)
    # Default is None.
    assert bh.ProcessNoiseConfig(q, scale_with_dt=True).max_noise_dt is None
    # Round-trips a set value.
    assert bh.ProcessNoiseConfig(
        q, scale_with_dt=True, max_noise_dt=60.0
    ).max_noise_dt == pytest.approx(60.0)
    # Rejects non-positive / non-finite.
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            bh.ProcessNoiseConfig(q, scale_with_dt=True, max_noise_dt=bad)


def test_ekf_substep_single_chunk_equals_none(two_body_leo):
    # max_noise_dt >= interval matches the single-shot (None) behavior exactly.
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    q = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])
    dt = 300.0

    f_none = _substep_ekf(epoch, state, p0, q, None)
    f_none.propagate_to(epoch + dt)
    f_big = _substep_ekf(epoch, state, p0, q, dt * 2.0)
    f_big.propagate_to(epoch + dt)

    np.testing.assert_allclose(f_none.current_state(), f_big.current_state(), atol=1e-9)
    np.testing.assert_allclose(
        f_none.current_covariance(), f_big.current_covariance(), atol=1e-6
    )


def test_ekf_substep_definitional_equivalence(two_body_leo):
    # One sub-stepped propagate_to equals chained single-shot calls at cadence h.
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    q = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])
    h = 60.0

    f_sub = _substep_ekf(epoch, state, p0, q, h)
    f_sub.propagate_to(epoch + 300.0)

    f_chain = _substep_ekf(epoch, state, p0, q, None)
    for step in range(1, 6):
        f_chain.propagate_to(epoch + step * h)

    np.testing.assert_allclose(
        f_sub.current_state(), f_chain.current_state(), atol=1e-6
    )
    np.testing.assert_allclose(
        f_sub.current_covariance(), f_chain.current_covariance(), atol=1e-3
    )


def test_ekf_builder_equivalence(two_body_leo):
    """Builder-constructed EKF should behave identically to the flat constructor."""
    epoch, true_state = two_body_leo
    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    via_builder = (
        bh.ExtendedKalmanFilter.builder(
            epoch,
            initial_state,
            p0,
            bh.ForceModelConfig.two_body(),
            bh.EKFConfig.default(),
        )
        .propagation_config(bh.NumericalPropagationConfig.default())
        .measurement_model(bh.InertialPositionMeasurementModel(10.0))
        .build()
    )

    via_constructor = bh.ExtendedKalmanFilter(
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


def test_ekf_builder_optionals(two_body_leo):
    """Every optional setter: propagation_config, params, additional dynamics,
    control input, and wholesale measurement_models."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

    def zero_dynamics(t, state, p):
        return np.zeros(len(state))

    ekf = (
        bh.ExtendedKalmanFilter.builder(
            epoch,
            state,
            p0,
            bh.ForceModelConfig.two_body(),
            bh.EKFConfig.default(),
        )
        .propagation_config(bh.NumericalPropagationConfig.default())
        .params(params)
        .additional_dynamics(zero_dynamics)
        .control_input(zero_dynamics)
        .measurement_models([bh.InertialPositionMeasurementModel(10.0)])
        .build()
    )

    obs = bh.Observation(epoch + 60.0, state[:3], model_index=0)
    ekf.process_observation(obs)

    assert ekf.current_epoch() == epoch + 60.0


def test_ekf_builder_unchained_setter(two_body_leo):
    """Calling a setter without reassigning its return value must not orphan
    the original builder variable -- build() on the original must succeed."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    builder = bh.ExtendedKalmanFilter.builder(
        epoch,
        state,
        p0,
        bh.ForceModelConfig.two_body(),
        bh.EKFConfig.default(),
    ).measurement_model(bh.InertialPositionMeasurementModel(10.0))
    builder.propagation_config(
        bh.NumericalPropagationConfig.default()
    )  # not reassigned
    ekf = builder.build()

    assert len(ekf.current_state()) == 6


def test_ekf_builder_double_build_raises(two_body_leo):
    """Calling build() twice on the same builder should raise RuntimeError."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    builder = bh.ExtendedKalmanFilter.builder(
        epoch,
        state,
        p0,
        bh.ForceModelConfig.two_body(),
        bh.EKFConfig.default(),
    ).measurement_model(bh.InertialPositionMeasurementModel(10.0))
    builder.build()

    with pytest.raises(RuntimeError, match="builder already consumed"):
        builder.build()


def test_ekf_propagate_to_raising_callback_reraised(two_body_leo):
    """A raising control-input callback inside propagate_to re-raises the
    original Python exception (not a wrapped BraheError), and the stashed-error
    slot is drained so a second failing call raises a fresh exception."""
    epoch, state = two_body_leo
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    class CallbackBoom(Exception):
        pass

    def raising_control(t, state, p):
        raise CallbackBoom("callback exploded")

    ekf = (
        bh.ExtendedKalmanFilter.builder(
            epoch,
            state,
            p0,
            bh.ForceModelConfig.two_body(),
            bh.EKFConfig.default(),
        )
        .control_input(raising_control)
        .measurement_model(bh.InertialPositionMeasurementModel(10.0))
        .build()
    )

    with pytest.raises(CallbackBoom, match="callback exploded") as first:
        ekf.propagate_to(epoch + 60.0)

    with pytest.raises(CallbackBoom, match="callback exploded") as second:
        ekf.propagate_to(epoch + 120.0)

    assert first.value is not second.value
