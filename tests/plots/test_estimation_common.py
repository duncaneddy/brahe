"""Tests for estimation_common shared helpers."""

import numpy as np
import pytest

import brahe as bh
from brahe.plots.estimation_common import (
    DEFAULT_COLORS,
    compute_grid_layout,
    compute_time_axis,
    extract_covariance_sigmas,
    extract_residuals,
    extract_state_errors,
    extract_state_history,
    extract_sub_covariance,
    resolve_colors,
    resolve_labels,
)


# =============================================================================
# Fixtures
# =============================================================================


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
    """Generate noise-free inertial position observations every 30s."""
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
def solved_bls(two_body_leo, position_observations):
    """Solved BLS solver with iteration records and residuals stored."""
    epoch, true_state = two_body_leo
    initial_state = true_state.copy()
    initial_state[0] += 1000.0

    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    bls = bh.BatchLeastSquares(
        epoch,
        initial_state,
        p0,
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    )
    bls.solve(position_observations)
    return bls, epoch, true_state


@pytest.fixture
def solved_ekf(two_body_leo):
    """Solved EKF with 20 position measurements."""
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

    ekf = bh.ExtendedKalmanFilter(
        epoch,
        initial_state,
        p0,
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )

    for i in range(1, 21):
        t = epoch + i * 30.0
        truth_prop.propagate_to(t)
        pos = truth_prop.current_state()[:3]
        ekf.process_observation(bh.Observation(t, pos, 0))

    return ekf, epoch, true_state


# =============================================================================
# compute_time_axis
# =============================================================================


class TestComputeTimeAxis:
    def _make_epochs(self, n=10, dt=60.0):
        """Create a list of epochs spaced dt seconds apart."""
        epoch0 = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        return [epoch0 + i * dt for i in range(n)]

    def test_seconds_preset(self):
        epochs = self._make_epochs(5, dt=60.0)
        times, label = compute_time_axis(epochs, time_units="seconds")
        assert len(times) == 5
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(60.0)
        assert "second" in label.lower()

    def test_minutes_preset(self):
        epochs = self._make_epochs(5, dt=60.0)
        times, label = compute_time_axis(epochs, time_units="minutes")
        assert times[1] == pytest.approx(1.0)
        assert "minute" in label.lower()

    def test_hours_preset(self):
        epochs = self._make_epochs(5, dt=3600.0)
        times, label = compute_time_axis(epochs, time_units="hours")
        assert times[1] == pytest.approx(1.0)
        assert "hour" in label.lower()

    def test_orbits_preset(self):
        period = 5600.0
        epochs = self._make_epochs(5, dt=period)
        times, label = compute_time_axis(
            epochs, time_units="orbits", orbital_period=period
        )
        assert times[1] == pytest.approx(1.0)
        assert "orbit" in label.lower()

    def test_orbits_raises_without_period(self):
        epochs = self._make_epochs(5)
        with pytest.raises(ValueError, match="orbital_period"):
            compute_time_axis(epochs, time_units="orbits")

    def test_callable_time_units(self):
        epochs = self._make_epochs(5, dt=60.0)

        def to_milliseconds(seconds):
            return seconds * 1000.0

        times, label = compute_time_axis(epochs, time_units=to_milliseconds)
        assert times[1] == pytest.approx(60000.0)

    def test_epoch_mode(self):
        epochs = self._make_epochs(5, dt=60.0)
        times, label = compute_time_axis(epochs, time_units="epoch")
        # Should return the epochs themselves (or numeric representation)
        assert len(times) == 5
        assert "epoch" in label.lower()

    def test_unknown_string_raises(self):
        epochs = self._make_epochs(3)
        with pytest.raises(ValueError):
            compute_time_axis(epochs, time_units="parsecs")

    def test_single_epoch(self):
        epochs = self._make_epochs(1)
        times, label = compute_time_axis(epochs, time_units="seconds")
        assert len(times) == 1
        assert times[0] == pytest.approx(0.0)

    def test_returns_numpy_array(self):
        epochs = self._make_epochs(5)
        times, _ = compute_time_axis(epochs, time_units="seconds")
        assert isinstance(times, np.ndarray)


# =============================================================================
# resolve_colors
# =============================================================================


class TestResolveColors:
    def test_default_colors_returned_when_none(self):
        colors = resolve_colors(3)
        assert len(colors) == 3
        assert colors[0] == DEFAULT_COLORS[0]

    def test_provided_colors_used(self):
        my_colors = ["red", "green", "blue"]
        colors = resolve_colors(3, colors=my_colors)
        assert colors == ["red", "green", "blue"]

    def test_cycles_when_n_exceeds_default(self):
        n = len(DEFAULT_COLORS) + 2
        colors = resolve_colors(n)
        assert len(colors) == n
        # After cycling through, should wrap
        assert colors[len(DEFAULT_COLORS)] == DEFAULT_COLORS[0]

    def test_cycles_when_n_exceeds_provided(self):
        my_colors = ["red", "blue"]
        colors = resolve_colors(4, colors=my_colors)
        assert len(colors) == 4
        assert colors[2] == "red"

    def test_zero_n_returns_empty(self):
        colors = resolve_colors(0)
        assert colors == []

    def test_single_color(self):
        colors = resolve_colors(1)
        assert len(colors) == 1


# =============================================================================
# resolve_labels
# =============================================================================


class TestResolveLabels:
    def test_default_labels_generated(self):
        labels = resolve_labels(3)
        assert labels == ["Series 0", "Series 1", "Series 2"]

    def test_provided_labels_used(self):
        my_labels = ["Position X", "Position Y", "Position Z"]
        labels = resolve_labels(3, labels=my_labels)
        assert labels == my_labels

    def test_zero_n_returns_empty(self):
        labels = resolve_labels(0)
        assert labels == []

    def test_single_label(self):
        labels = resolve_labels(1)
        assert labels == ["Series 0"]

    def test_labels_numbered_correctly(self):
        labels = resolve_labels(5)
        assert labels[4] == "Series 4"


# =============================================================================
# compute_grid_layout
# =============================================================================


class TestComputeGridLayout:
    def test_three_states_three_cols(self):
        nrows, ncols = compute_grid_layout(3, ncols=3)
        assert ncols == 3
        assert nrows == 1

    def test_six_states_three_cols(self):
        nrows, ncols = compute_grid_layout(6, ncols=3)
        assert ncols == 3
        assert nrows == 2

    def test_seven_states_three_cols(self):
        nrows, ncols = compute_grid_layout(7, ncols=3)
        assert ncols == 3
        assert nrows == 3  # ceil(7/3) = 3

    def test_one_state(self):
        nrows, ncols = compute_grid_layout(1, ncols=3)
        assert nrows == 1
        assert ncols == 3

    def test_two_cols(self):
        nrows, ncols = compute_grid_layout(4, ncols=2)
        assert ncols == 2
        assert nrows == 2

    def test_ncols_larger_than_states(self):
        nrows, ncols = compute_grid_layout(2, ncols=3)
        assert ncols == 3
        assert nrows == 1


# =============================================================================
# extract_state_history (BLS)
# =============================================================================


class TestExtractStateHistoryBLS:
    def test_returns_epochs_and_states(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, states = extract_state_history(bls)
        assert len(epochs) > 0
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2

    def test_state_dim_is_six(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, states = extract_state_history(bls)
        assert states.shape[1] == 6

    def test_epochs_count_matches_state_rows(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, states = extract_state_history(bls)
        assert len(epochs) == states.shape[0]


# =============================================================================
# extract_state_history (EKF)
# =============================================================================


class TestExtractStateHistoryEKF:
    def test_returns_epochs_and_states(self, solved_ekf):
        ekf, _, _ = solved_ekf
        epochs, states = extract_state_history(ekf)
        assert len(epochs) == 20
        assert states.shape == (20, 6)

    def test_epochs_match_observation_count(self, solved_ekf):
        ekf, _, _ = solved_ekf
        epochs, states = extract_state_history(ekf)
        assert len(epochs) == len(ekf.records())


# =============================================================================
# extract_state_errors
# =============================================================================


class TestExtractStateErrors:
    def _make_trajectory(self, epoch, state):
        """Create a truth trajectory using numerical propagation."""
        prop = bh.NumericalOrbitPropagator(
            epoch,
            state,
            bh.NumericalPropagationConfig.default(),
            bh.ForceModelConfig.two_body(),
        )
        # Propagate for enough time to cover EKF records
        prop.propagate_to(epoch + 20 * 30.0)
        return prop.trajectory

    def test_errors_shape_matches_state_history(self, solved_ekf):
        ekf, epoch, true_state = solved_ekf
        true_traj = self._make_trajectory(epoch, true_state)
        epochs, errors = extract_state_errors(ekf, true_traj)
        assert errors.ndim == 2
        assert errors.shape[1] == 6
        assert len(epochs) == errors.shape[0]

    def test_ekf_errors_reduce_over_time(self, solved_ekf):
        """Error norm should generally decrease as filter converges."""
        ekf, epoch, true_state = solved_ekf
        true_traj = self._make_trajectory(epoch, true_state)
        epochs, errors = extract_state_errors(ekf, true_traj)
        pos_errors = np.linalg.norm(errors[:, :3], axis=1)
        # Final errors should be less than initial (filter converges)
        assert pos_errors[-1] < pos_errors[0]


# =============================================================================
# extract_covariance_sigmas
# =============================================================================


class TestExtractCovarianceSigmas:
    def test_bls_returns_sigmas(self, solved_bls):
        bls, _, _ = solved_bls
        sigmas = extract_covariance_sigmas(bls, sigma=3)
        assert isinstance(sigmas, np.ndarray)
        assert sigmas.ndim == 2
        assert sigmas.shape[1] == 6

    def test_ekf_returns_sigmas(self, solved_ekf):
        ekf, _, _ = solved_ekf
        sigmas = extract_covariance_sigmas(ekf, sigma=3)
        assert sigmas.ndim == 2
        assert sigmas.shape == (20, 6)

    def test_sigma_scale_is_applied(self, solved_ekf):
        ekf, _, _ = solved_ekf
        sigmas_1 = extract_covariance_sigmas(ekf, sigma=1)
        sigmas_3 = extract_covariance_sigmas(ekf, sigma=3)
        np.testing.assert_allclose(sigmas_3, sigmas_1 * 3.0)

    def test_sigmas_are_non_negative(self, solved_ekf):
        ekf, _, _ = solved_ekf
        sigmas = extract_covariance_sigmas(ekf, sigma=1)
        assert np.all(sigmas >= 0)


# =============================================================================
# extract_residuals
# =============================================================================


class TestExtractResiduals:
    def test_bls_postfit_residuals(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, residuals, n_components = extract_residuals(
            bls, iteration=-1, residual_type="postfit"
        )
        assert len(epochs) > 0
        assert residuals.ndim == 2
        assert n_components == 3  # InertialPosition = 3D

    def test_bls_prefit_residuals(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, residuals, n_components = extract_residuals(
            bls, iteration=0, residual_type="prefit"
        )
        assert len(epochs) > 0
        assert residuals.shape[1] == n_components

    def test_ekf_postfit_residuals(self, solved_ekf):
        ekf, _, _ = solved_ekf
        epochs, residuals, n_components = extract_residuals(
            ekf, residual_type="postfit"
        )
        assert len(epochs) == 20
        assert residuals.shape == (20, n_components)

    def test_ekf_prefit_residuals(self, solved_ekf):
        ekf, _, _ = solved_ekf
        epochs, residuals, n_components = extract_residuals(ekf, residual_type="prefit")
        assert residuals.shape[0] == 20

    def test_bls_model_name_filter(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, residuals, n = extract_residuals(bls, model_name="InertialPosition")
        assert len(epochs) > 0

    def test_bls_unknown_model_name_returns_empty(self, solved_bls):
        bls, _, _ = solved_bls
        epochs, residuals, n = extract_residuals(bls, model_name="NonExistentModel")
        assert len(epochs) == 0

    def test_invalid_residual_type_raises(self, solved_ekf):
        ekf, _, _ = solved_ekf
        with pytest.raises(ValueError, match="residual_type"):
            extract_residuals(ekf, residual_type="invalid")


# =============================================================================
# extract_sub_covariance
# =============================================================================


class TestExtractSubCovariance:
    def test_ekf_default_indices(self, solved_ekf):
        ekf, _, _ = solved_ekf
        mean_2d, cov_2x2 = extract_sub_covariance(ekf, state_indices=(0, 1))
        assert mean_2d.shape == (2,)
        assert cov_2x2.shape == (2, 2)

    def test_bls_default_indices(self, solved_bls):
        bls, _, _ = solved_bls
        mean_2d, cov_2x2 = extract_sub_covariance(bls, state_indices=(0, 1))
        assert mean_2d.shape == (2,)
        assert cov_2x2.shape == (2, 2)

    def test_cov_is_symmetric(self, solved_ekf):
        ekf, _, _ = solved_ekf
        _, cov = extract_sub_covariance(ekf, state_indices=(0, 1))
        np.testing.assert_allclose(cov, cov.T)

    def test_different_indices(self, solved_ekf):
        ekf, _, _ = solved_ekf
        mean, cov = extract_sub_covariance(ekf, state_indices=(3, 4))
        assert mean.shape == (2,)
        assert cov.shape == (2, 2)
