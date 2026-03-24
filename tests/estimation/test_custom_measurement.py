"""Tests for custom Python-defined measurement models with the EKF."""

import numpy as np
import pytest

import brahe as bh


class RangeModel(bh.MeasurementModel):
    """Custom range measurement: distance from a station to the satellite."""

    def __init__(self, station_eci, sigma):
        super().__init__()
        self.station_eci = np.array(station_eci)
        self.sigma = sigma

    def predict(self, epoch, state):
        pos = state[:3]
        return np.array([np.linalg.norm(pos - self.station_eci)])

    def noise_covariance(self):
        return np.array([[self.sigma**2]])

    def measurement_dim(self):
        return 1

    def name(self):
        return "Range"

    # jacobian() NOT overridden - uses default finite differences


class RangeModelWithJacobian(bh.MeasurementModel):
    """Range model with analytical Jacobian."""

    def __init__(self, station_eci, sigma):
        super().__init__()
        self.station_eci = np.array(station_eci)
        self.sigma = sigma

    def predict(self, epoch, state):
        pos = state[:3]
        return np.array([np.linalg.norm(pos - self.station_eci)])

    def jacobian(self, epoch, state):
        pos = state[:3]
        diff = pos - self.station_eci
        r = np.linalg.norm(diff)
        n = len(state)
        h = np.zeros((1, n))
        h[0, 0] = diff[0] / r
        h[0, 1] = diff[1] / r
        h[0, 2] = diff[2] / r
        return h

    def noise_covariance(self):
        return np.array([[self.sigma**2]])

    def measurement_dim(self):
        return 1

    def name(self):
        return "RangeAnalytical"


# =============================================================================
# Custom model standalone tests
# =============================================================================


class TestCustomMeasurementModel:
    def test_construction(self):
        model = RangeModel([0.0, 0.0, 0.0], 100.0)
        assert model.measurement_dim() == 1
        assert model.name() == "Range"

    def test_predict(self):
        station = np.array([6378e3, 0.0, 0.0])
        model = RangeModel(station, 100.0)
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        state = np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])

        z = model.predict(epoch, state)
        expected_range = 500e3  # 6878km - 6378km
        assert z[0] == pytest.approx(expected_range)

    def test_noise_covariance(self):
        model = RangeModel([0.0, 0.0, 0.0], 100.0)
        r = model.noise_covariance()
        assert r.shape == (1, 1)
        assert r[0, 0] == pytest.approx(10000.0)


class TestCustomMeasurementModelWithJacobian:
    def test_jacobian_returns_array(self):
        station = np.array([6378e3, 0.0, 0.0])
        model = RangeModelWithJacobian(station, 100.0)
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        state = np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])

        h = model.jacobian(epoch, state)
        assert h.shape == (1, 6)
        # For position along x-axis, Jacobian should be [1, 0, 0, 0, 0, 0]
        assert h[0, 0] == pytest.approx(1.0)
        assert h[0, 1] == pytest.approx(0.0)
        assert h[0, 3] == pytest.approx(0.0)


# =============================================================================
# Integration with EKF
# =============================================================================


class TestCustomModelWithEKF:
    @pytest.fixture
    def two_body_setup(self):
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        r = 6878.0e3
        v = (bh.GM_EARTH / r) ** 0.5
        true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        return epoch, true_state

    def test_ekf_with_custom_range_model(self, two_body_setup):
        """EKF runs with a Python-defined range model."""
        epoch, true_state = two_body_setup

        station = np.array([6378e3, 0.0, 0.0])
        model = RangeModel(station, 100.0)

        initial_state = true_state.copy()
        initial_state[0] += 500.0  # 500m offset
        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

        ekf = bh.ExtendedKalmanFilter(
            epoch,
            initial_state,
            p0,
            measurement_models=[model],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
        )

        # Generate a range observation
        true_range = np.linalg.norm(true_state[:3] - station)
        obs = bh.Observation(epoch + 60.0, np.array([true_range]), model_index=0)

        record = ekf.process_observation(obs)
        assert record.measurement_name == "Range"
        assert len(record.prefit_residual) == 1
        assert len(record.postfit_residual) == 1

    def test_ekf_with_analytical_jacobian_model(self, two_body_setup):
        """EKF runs with a Python model that provides analytical Jacobian."""
        epoch, true_state = two_body_setup

        station = np.array([6378e3, 0.0, 0.0])
        model = RangeModelWithJacobian(station, 100.0)

        initial_state = true_state.copy()
        initial_state[0] += 500.0
        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

        ekf = bh.ExtendedKalmanFilter(
            epoch,
            initial_state,
            p0,
            measurement_models=[model],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
        )

        true_range = np.linalg.norm(true_state[:3] - station)
        obs = bh.Observation(epoch + 60.0, np.array([true_range]), model_index=0)

        record = ekf.process_observation(obs)
        assert record.measurement_name == "RangeAnalytical"

    def test_ekf_mixed_models(self, two_body_setup):
        """EKF with both a built-in model and a custom Python model."""
        epoch, true_state = two_body_setup

        builtin_model = bh.InertialPositionMeasurementModel(10.0)
        station = np.array([6378e3, 0.0, 0.0])
        custom_model = RangeModel(station, 100.0)

        initial_state = true_state.copy()
        initial_state[0] += 500.0
        p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

        ekf = bh.ExtendedKalmanFilter(
            epoch,
            initial_state,
            p0,
            measurement_models=[builtin_model, custom_model],
            propagation_config=bh.NumericalPropagationConfig.default(),
            force_config=bh.ForceModelConfig.two_body(),
        )

        # Process a position observation (model_index=0 -> builtin)
        obs1 = bh.Observation(epoch + 60.0, true_state[:3], model_index=0)
        record1 = ekf.process_observation(obs1)
        assert record1.measurement_name == "InertialPosition"

        # Process a range observation (model_index=1 -> custom)
        true_range = np.linalg.norm(true_state[:3] - station)
        obs2 = bh.Observation(epoch + 120.0, np.array([true_range]), model_index=1)
        record2 = ekf.process_observation(obs2)
        assert record2.measurement_name == "Range"

        assert len(ekf.records()) == 2
