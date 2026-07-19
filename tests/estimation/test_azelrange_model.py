"""Tests for AzElRangeMeasurementModel Python bindings."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture
def test_epoch():
    return bh.Epoch(2024, 1, 1, 0, 0, 0.0)


def make_state_above(epoch, lon, lat, alt):
    """ECI state directly above a geodetic point (zero velocity)."""
    ecef = bh.position_geodetic_to_ecef(
        np.array([lon, lat, alt]), bh.AngleFormat.DEGREES
    )
    eci = bh.position_ecef_to_eci(epoch, ecef)
    return np.array([eci[0], eci[1], eci[2], 0.0, 0.0, 0.0])


class TestAzElRangeMeasurementModel:
    def test_construction(self):
        model = bh.AzElRangeMeasurementModel(-71.49, 42.62, 123.1, 0.01, 0.01, 10.0)
        assert model.measurement_dim() == 3
        assert model.name() == "AzElRange"

    def test_constructor_rejects_invalid_latitude(self):
        # Latitude out of [-90, 90] must raise ValueError (the Rust constructor
        # now returns a Result mapped to a Python exception).
        with pytest.raises(ValueError, match="latitude"):
            bh.AzElRangeMeasurementModel(-71.49, 100.0, 123.1, 0.01, 0.01, 10.0)
        # Non-finite coordinates are rejected with a finite-value error.
        with pytest.raises(ValueError, match="finite"):
            bh.AzElRangeMeasurementModel(float("nan"), 42.62, 123.1, 0.01, 0.01, 10.0)

    def test_from_covariance_rejects_invalid_latitude(self):
        cov = np.diag([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="latitude"):
            bh.AzElRangeMeasurementModel.from_covariance(0.0, 100.0, 100.0, cov)

    def test_residual_wraps_azimuth(self):
        model = bh.AzElRangeMeasurementModel(0.0, 0.0, 0.0, 0.01, 0.01, 10.0)
        r = model.residual(np.array([359.9, 45.0, 1e6]), np.array([0.1, 45.0, 1e6]))
        assert r[0] == pytest.approx(-0.2, abs=1e-9)
        assert r[1] == pytest.approx(0.0, abs=1e-12)
        assert r[2] == pytest.approx(0.0, abs=1e-6)

    def test_residual_rejects_wrong_length(self):
        model = bh.AzElRangeMeasurementModel(0.0, 0.0, 0.0, 0.01, 0.01, 10.0)
        # Mismatched / empty arrays must raise, not panic.
        with pytest.raises(ValueError, match="measurement_dim"):
            model.residual(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        # Mixed: valid-length measured, wrong-length predicted.
        with pytest.raises(ValueError, match="measurement_dim"):
            model.residual(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="measurement_dim"):
            model.residual(np.array([]), np.array([]))

    def test_predict_zenith(self, test_epoch):
        lon, lat, alt = -71.49, 42.62, 123.1
        model = bh.AzElRangeMeasurementModel(lon, lat, alt, 0.01, 0.01, 10.0)
        state = make_state_above(test_epoch, lon, lat, alt + 500e3)

        z = model.predict(test_epoch, state)
        assert len(z) == 3
        assert z[1] == pytest.approx(90.0, abs=1e-5)  # elevation
        assert z[2] == pytest.approx(500e3, abs=1.0)  # range

    def test_bias(self, test_epoch):
        base = bh.AzElRangeMeasurementModel(-71.49, 42.62, 123.1, 0.01, 0.01, 10.0)
        biased = bh.AzElRangeMeasurementModel(
            -71.49,
            42.62,
            123.1,
            0.01,
            0.01,
            10.0,
            bias_az=0.5,
            bias_el=-0.25,
            bias_range=100.0,
        )
        state = make_state_above(test_epoch, -70.0, 44.0, 800e3)
        z0 = base.predict(test_epoch, state)
        z1 = biased.predict(test_epoch, state)
        assert z1[0] - z0[0] == pytest.approx(0.5, abs=1e-9)
        assert z1[1] - z0[1] == pytest.approx(-0.25, abs=1e-9)
        assert z1[2] - z0[2] == pytest.approx(100.0, abs=1e-6)

    def test_measurement_dim_and_noise(self):
        model = bh.AzElRangeMeasurementModel(0.0, 45.0, 100.0, 0.02, 0.03, 50.0)
        assert model.measurement_dim() == 3
        r = model.noise_covariance()
        assert r[0, 0] == pytest.approx(0.02**2)
        assert r[1, 1] == pytest.approx(0.03**2)
        assert r[2, 2] == pytest.approx(2500.0)

    def test_from_covariance_and_upper_triangular(self):
        cov = np.diag([1.0, 2.0, 3.0])
        model = bh.AzElRangeMeasurementModel.from_covariance(0.0, 45.0, 100.0, cov)
        r = model.noise_covariance()
        assert r[0, 0] == pytest.approx(1.0)
        assert r[1, 1] == pytest.approx(2.0)
        assert r[2, 2] == pytest.approx(3.0)

        model2 = bh.AzElRangeMeasurementModel.from_upper_triangular(
            0.0, 45.0, 100.0, np.array([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])
        )
        r2 = model2.noise_covariance()
        assert r2[0, 0] == pytest.approx(1.0)
        assert r2[1, 1] == pytest.approx(2.0)
        assert r2[2, 2] == pytest.approx(3.0)

        bad = np.identity(2)
        with pytest.raises(ValueError):
            bh.AzElRangeMeasurementModel.from_covariance(0.0, 45.0, 100.0, bad)

    def test_from_covariance_invalid_latitude_errors(self):
        cov = np.diag([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            bh.AzElRangeMeasurementModel.from_covariance(0.0, 100.0, 100.0, cov)


class TestAzElRangeFilterAzimuthWrap:
    """Verifies azimuth-wrap residual FORWARDING through the binding layer.

    Builds an EKF with an AzElRangeMeasurementModel constructed via the
    normal Python constructor and processes an observation whose measured
    azimuth is ~0.1 deg across the 0/360 boundary from the prediction. The
    test's sole purpose is to prove that ``MeasurementModelHolder`` dispatches
    to the model's wrapping ``residual()`` rather than falling back to plain
    subtraction; it does NOT assert anything about Jacobian correctness (see
    ``test_jacobian_wrap_aware_near_north`` for that). If forwarding were lost,
    the filter's pre-fit azimuth innovation would be the raw ~-359.9 deg
    difference instead of the wrapped ~-0.1 deg residual.
    """

    def test_residual_forwarding_survives_binding_layer(self, test_epoch):
        lon, lat, alt = -71.49, 42.62, 123.1
        # Small noise sigmas + large initial covariance so the Kalman gain
        # is large enough that an unwrapped ~360 deg residual would blow
        # the state estimate up if wrapping were lost.
        model = bh.AzElRangeMeasurementModel(lon, lat, alt, 0.05, 0.05, 50.0)

        # Geometry due north of the station -> predicted azimuth lands
        # extremely close to the 0/360 deg boundary (on either side,
        # depending on floating-point/EOP noise).
        state0 = make_state_above(test_epoch, lon, lat + 2.0, 500e3 + alt)
        z_pred = model.predict(test_epoch, state0)
        pred_az = float(z_pred[0])
        assert pred_az < 1e-3 or pred_az > 360.0 - 1e-3

        # Offset the "measured" azimuth by 0.1 deg across the wrap boundary
        # from whichever side the prediction landed on, so the *raw*
        # difference is ~360 deg while the *true* angular deviation is
        # only 0.1 deg.
        if pred_az < 180.0:
            measured_az = (pred_az - 0.1) % 360.0
        else:
            measured_az = (pred_az + 0.1) % 360.0
        raw_diff = measured_az - pred_az
        assert abs(raw_diff) > 180.0  # sanity: this is the phantom-jump case

        p0 = bh.isotropic_covariance(6, 1000.0**2)
        ekf = bh.ExtendedKalmanFilter(
            test_epoch,
            state0,
            p0,
            bh.NumericalPropagationConfig.default(),
            bh.ForceModelConfig.two_body(),
            [model],
        )

        measured = np.array([measured_az, z_pred[1], z_pred[2]])
        obs = bh.Observation(test_epoch, measured, 0)
        record = ekf.process_observation(obs)

        # The one discriminating assertion: the filter's pre-fit azimuth
        # innovation is the wrapped ~-0.1 deg residual, not the ~-359.9 deg raw
        # difference. If MeasurementModelHolder no longer dispatched to the
        # model's wrapping residual() and fell back to plain subtraction, this
        # value would be ~359.8 deg (raw_diff above is asserted > 180 deg, so
        # the wrap is genuinely in play) and the assertion would fail.
        assert abs(record.prefit_residual[0]) < 1.0


class TestAzElRangeJacobianWrap:
    """Discriminates the wrap-aware finite-difference Jacobian at the Python
    level. Mirrors the Rust ``test_azelrange_jacobian_wrap_aware_near_north``
    geometry so the magnitude bound is grounded, not guessed."""

    def test_jacobian_wrap_aware_near_north(self, test_epoch):
        # Station at (lon, lat) = (0, 0); target ~3 deg due north at 500 km ->
        # predicted azimuth sits on the 0/360 wrap.
        model = bh.AzElRangeMeasurementModel(0.0, 0.0, 0.0, 0.01, 0.01, 10.0)
        state_north = make_state_above(test_epoch, 0.0, 3.0, 500e3)

        z_north = model.predict(test_epoch, state_north)
        az = float(z_north[0])
        assert az < 1.0 or az > 359.0  # geometry genuinely on the wrap

        h_north = model.jacobian(test_epoch, state_north)
        assert h_north.shape == (3, 6)
        assert np.all(np.isfinite(h_north)), "Jacobian must be finite near wrap"

        # Equivalent geometry rotated due east (azimuth ~90 deg) where the wrap
        # cannot occur; the azimuth-row magnitude sets the reference scale.
        state_east = make_state_above(test_epoch, 3.0, 0.0, 500e3)
        h_east = model.jacobian(test_epoch, state_east)

        az_norm_north = np.linalg.norm(h_north[0, :])
        az_norm_east = np.linalg.norm(h_east[0, :])
        ratio = az_norm_north / az_norm_east
        # Broken raw differencing across the wrap inflates the north azimuth row
        # by ~360/h, i.e. many orders of magnitude; the wrap-aware Jacobian keeps
        # the two comparable.
        assert 0.1 < ratio < 10.0, (
            f"azimuth-row magnitudes should be comparable "
            f"(north={az_norm_north:.3e}, east={az_norm_east:.3e}, ratio={ratio:.3f})"
        )


def test_azelrange_predict_rejects_short_state(test_epoch):
    """predict() must reject a state shorter than 3 elements."""
    model = bh.AzElRangeMeasurementModel(-71.49, 42.62, 123.1, 0.01, 0.01, 10.0)
    with pytest.raises(RuntimeError, match="state dimension >= 3"):
        model.predict(test_epoch, np.array([6878e3, 0.0]))
