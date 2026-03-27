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


# =============================================================================
# from_covariance / from_upper_triangular constructors
# =============================================================================


class TestFromCovariance:
    def test_inertial_position_from_covariance(self):
        cov = np.array([[100.0, 5.0, 0.0], [5.0, 225.0, 10.0], [0.0, 10.0, 400.0]])
        model = bh.InertialPositionMeasurementModel.from_covariance(cov)
        r = model.noise_covariance()
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r, cov)
        assert model.measurement_dim() == 3

    def test_inertial_position_from_covariance_wrong_dim(self):
        cov = np.eye(6) * 100.0
        with pytest.raises(ValueError):
            bh.InertialPositionMeasurementModel.from_covariance(cov)

    def test_inertial_state_from_covariance(self):
        cov = np.eye(6) * 50.0
        model = bh.InertialStateMeasurementModel.from_covariance(cov)
        r = model.noise_covariance()
        assert r.shape == (6, 6)
        np.testing.assert_allclose(r, cov)

    def test_inertial_state_from_covariance_wrong_dim(self):
        cov = np.eye(3) * 100.0
        with pytest.raises(ValueError):
            bh.InertialStateMeasurementModel.from_covariance(cov)

    def test_ecef_position_from_covariance(self):
        cov = np.diag([25.0, 25.0, 100.0])
        model = bh.ECEFPositionMeasurementModel.from_covariance(cov)
        r = model.noise_covariance()
        np.testing.assert_allclose(r[(2, 2)], 100.0)

    def test_ecef_state_from_covariance(self):
        cov = np.eye(6) * 10.0
        model = bh.ECEFStateMeasurementModel.from_covariance(cov)
        assert model.measurement_dim() == 6

    def test_inertial_velocity_from_covariance(self):
        cov = np.diag([0.01, 0.04, 0.09])
        model = bh.InertialVelocityMeasurementModel.from_covariance(cov)
        r = model.noise_covariance()
        np.testing.assert_allclose(r, cov)

    def test_ecef_velocity_from_covariance(self):
        cov = np.diag([0.01, 0.04, 0.09])
        model = bh.ECEFVelocityMeasurementModel.from_covariance(cov)
        r = model.noise_covariance()
        np.testing.assert_allclose(r, cov)


class TestFromUpperTriangular:
    def test_inertial_position_from_upper_triangular(self):
        # [c00, c01, c02, c11, c12, c22]
        upper = np.array([100.0, 5.0, 0.0, 225.0, 10.0, 400.0])
        model = bh.InertialPositionMeasurementModel.from_upper_triangular(upper)
        r = model.noise_covariance()
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r[0, 0], 100.0)
        np.testing.assert_allclose(r[0, 1], 5.0)
        np.testing.assert_allclose(r[1, 0], 5.0)  # symmetric

    def test_ecef_state_from_upper_triangular(self):
        # 6x6 -> 21 elements, diagonal only
        upper = np.zeros(21)
        upper[0] = 100.0  # (0,0)
        upper[6] = 100.0  # (1,1)
        upper[11] = 400.0  # (2,2)
        upper[15] = 0.01  # (3,3)
        upper[18] = 0.01  # (4,4)
        upper[20] = 0.04  # (5,5)
        model = bh.ECEFStateMeasurementModel.from_upper_triangular(upper)
        r = model.noise_covariance()
        assert r.shape == (6, 6)
        np.testing.assert_allclose(r[0, 0], 100.0)
        np.testing.assert_allclose(r[5, 5], 0.04)

    def test_wrong_element_count(self):
        with pytest.raises(ValueError):
            bh.InertialPositionMeasurementModel.from_upper_triangular(
                np.array([1.0, 2.0])
            )


# =============================================================================
# Standalone covariance helper functions
# =============================================================================


class TestCovarianceHelpers:
    def test_isotropic_covariance(self):
        r = bh.isotropic_covariance(3, 10.0)
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r[0, 0], 100.0)
        np.testing.assert_allclose(r[1, 1], 100.0)
        np.testing.assert_allclose(r[0, 1], 0.0)

    def test_diagonal_covariance(self):
        r = bh.diagonal_covariance(np.array([5.0, 10.0, 15.0]))
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r[0, 0], 25.0)
        np.testing.assert_allclose(r[1, 1], 100.0)
        np.testing.assert_allclose(r[2, 2], 225.0)
        np.testing.assert_allclose(r[0, 1], 0.0)
