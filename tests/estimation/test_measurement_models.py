"""Tests for built-in measurement model Python bindings."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture
def test_state():
    """Standard test state: LEO with slight off-axis position and velocity."""
    return np.array([6878.0e3, 100.0e3, 50.0e3, 0.0, 7612.0, 100.0])


@pytest.fixture
def test_epoch():
    return bh.Epoch(2024, 1, 1, 0, 0, 0.0)


# =============================================================================
# InertialPositionMeasurementModel
# =============================================================================


class TestInertialPositionMeasurementModel:
    def test_construction(self):
        model = bh.InertialPositionMeasurementModel(10.0)
        assert model.measurement_dim() == 3
        assert model.name() == "InertialPosition"

    def test_predict(self, test_state, test_epoch):
        model = bh.InertialPositionMeasurementModel(10.0)
        z = model.predict(test_epoch, test_state)
        assert len(z) == 3
        assert z[0] == pytest.approx(6878.0e3)
        assert z[1] == pytest.approx(100.0e3)
        assert z[2] == pytest.approx(50.0e3)

    def test_jacobian(self, test_state, test_epoch):
        model = bh.InertialPositionMeasurementModel(10.0)
        h = model.jacobian(test_epoch, test_state)
        assert h.shape == (3, 6)
        # Should be [I_3 | 0_3x3]
        assert h[0, 0] == pytest.approx(1.0)
        assert h[1, 1] == pytest.approx(1.0)
        assert h[2, 2] == pytest.approx(1.0)
        assert h[0, 3] == pytest.approx(0.0)

    def test_noise_covariance(self):
        model = bh.InertialPositionMeasurementModel(10.0)
        r = model.noise_covariance()
        assert r.shape == (3, 3)
        assert r[0, 0] == pytest.approx(100.0)
        assert r[0, 1] == pytest.approx(0.0)

    def test_per_axis(self):
        model = bh.InertialPositionMeasurementModel.per_axis(5.0, 10.0, 15.0)
        r = model.noise_covariance()
        assert r[0, 0] == pytest.approx(25.0)
        assert r[1, 1] == pytest.approx(100.0)
        assert r[2, 2] == pytest.approx(225.0)

    def test_repr(self):
        model = bh.InertialPositionMeasurementModel(10.0)
        assert "InertialPosition" in repr(model)


# =============================================================================
# InertialVelocityMeasurementModel
# =============================================================================


class TestInertialVelocityMeasurementModel:
    def test_construction(self):
        model = bh.InertialVelocityMeasurementModel(0.1)
        assert model.measurement_dim() == 3
        assert model.name() == "InertialVelocity"

    def test_predict(self, test_state, test_epoch):
        model = bh.InertialVelocityMeasurementModel(0.1)
        z = model.predict(test_epoch, test_state)
        assert len(z) == 3
        assert z[0] == pytest.approx(0.0)
        assert z[1] == pytest.approx(7612.0)
        assert z[2] == pytest.approx(100.0)

    def test_jacobian(self, test_state, test_epoch):
        model = bh.InertialVelocityMeasurementModel(0.1)
        h = model.jacobian(test_epoch, test_state)
        assert h.shape == (3, 6)
        assert h[0, 3] == pytest.approx(1.0)
        assert h[1, 4] == pytest.approx(1.0)
        assert h[2, 5] == pytest.approx(1.0)
        assert h[0, 0] == pytest.approx(0.0)


# =============================================================================
# InertialStateMeasurementModel
# =============================================================================


class TestInertialStateMeasurementModel:
    def test_construction(self):
        model = bh.InertialStateMeasurementModel(10.0, 0.1)
        assert model.measurement_dim() == 6
        assert model.name() == "InertialState"

    def test_predict(self, test_state, test_epoch):
        model = bh.InertialStateMeasurementModel(10.0, 0.1)
        z = model.predict(test_epoch, test_state)
        assert len(z) == 6
        np.testing.assert_allclose(z, test_state, atol=1e-10)

    def test_noise_covariance(self):
        model = bh.InertialStateMeasurementModel(10.0, 0.1)
        r = model.noise_covariance()
        assert r.shape == (6, 6)
        assert r[0, 0] == pytest.approx(100.0)
        assert r[3, 3] == pytest.approx(0.01)

    def test_per_axis(self):
        model = bh.InertialStateMeasurementModel.per_axis(
            5.0, 10.0, 15.0, 0.05, 0.1, 0.15
        )
        r = model.noise_covariance()
        assert r[0, 0] == pytest.approx(25.0)
        assert r[5, 5] == pytest.approx(0.0225)


# =============================================================================
# ECEF measurement models (require EOP)
# =============================================================================


class TestECEFPositionMeasurementModel:
    def test_construction(self):
        model = bh.ECEFPositionMeasurementModel(5.0)
        assert model.measurement_dim() == 3
        assert model.name() == "EcefPosition"

    def test_predict(self, test_state, test_epoch):
        model = bh.ECEFPositionMeasurementModel(5.0)
        z = model.predict(test_epoch, test_state)
        assert len(z) == 3
        # ECEF position magnitude should match ECI magnitude
        eci_mag = np.linalg.norm(test_state[:3])
        ecef_mag = np.linalg.norm(z)
        assert ecef_mag == pytest.approx(eci_mag, abs=1.0)

    def test_per_axis_noise(self):
        model = bh.ECEFPositionMeasurementModel.per_axis(3.0, 5.0, 8.0)
        r = model.noise_covariance()
        assert r[0, 0] == pytest.approx(9.0)
        assert r[1, 1] == pytest.approx(25.0)
        assert r[2, 2] == pytest.approx(64.0)


class TestECEFVelocityMeasurementModel:
    def test_construction(self):
        model = bh.ECEFVelocityMeasurementModel(0.05)
        assert model.measurement_dim() == 3
        assert model.name() == "EcefVelocity"


class TestECEFStateMeasurementModel:
    def test_construction(self):
        model = bh.ECEFStateMeasurementModel(5.0, 0.05)
        assert model.measurement_dim() == 6
        assert model.name() == "EcefState"

    def test_predict(self, test_state, test_epoch):
        model = bh.ECEFStateMeasurementModel(5.0, 0.05)
        z = model.predict(test_epoch, test_state)
        assert len(z) == 6


# =============================================================================
# Data type tests
# =============================================================================


class TestObservation:
    def test_construction(self):
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        meas = np.array([1.0, 2.0, 3.0])
        obs = bh.Observation(epoch, meas, model_index=0)
        assert obs.model_index == 0
        np.testing.assert_allclose(obs.measurement, meas)

    def test_repr(self):
        obs = bh.Observation(bh.Epoch(2024, 1, 1, 0, 0, 0.0), np.array([1.0, 2.0, 3.0]))
        assert "Observation" in repr(obs)


class TestProcessNoiseConfig:
    def test_construction(self):
        q = np.eye(6) * 1e-6
        pn = bh.ProcessNoiseConfig(q)
        assert pn.scale_with_dt is False
        np.testing.assert_allclose(pn.q_matrix, q)

    def test_scale_with_dt(self):
        q = np.eye(6) * 1e-6
        pn = bh.ProcessNoiseConfig(q, scale_with_dt=True)
        assert pn.scale_with_dt is True


class TestEKFConfig:
    def test_default(self):
        config = bh.EKFConfig.default()
        assert config.store_records is True

    def test_custom(self):
        config = bh.EKFConfig(store_records=False)
        assert config.store_records is False
