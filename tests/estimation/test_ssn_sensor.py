"""Tests for SimpleSSNSensor Python bindings."""

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


def test_from_locations_dataset():
    sites = bh.datasets.ssn_sensors.load()
    sensors = bh.SimpleSSNSensor.from_locations(sites, seed=42)
    # All 21 sites: 15 azel_range + 6 optical, incl. HAX + Haystack (zero noise).
    assert len(sensors) == 21
    names = {s.name for s in sensors}
    assert "Eglin" in names
    assert "Millstone" in names
    assert "Socorro" in names  # optical site now constructs
    assert "Haystack" in names  # loaded with zero noise (uncalibrated)


def test_from_locations_calibrated_only():
    sites = bh.datasets.ssn_sensors.load()
    sensors = bh.SimpleSSNSensor.from_locations_calibrated(sites, seed=42)
    # 13 calibrated azel_range + 3 calibrated optical (Socorro, Maui,
    # Diego Garcia) = 16.
    assert len(sensors) == 16
    assert all(s.calibrated for s in sensors)
    names = {s.name for s in sensors}
    assert "Haystack" not in names
    assert "HAX" not in names
    assert "Socorro" in names  # calibrated optical site is included


def test_from_location_dataset_roundtrip():
    sites = bh.datasets.ssn_sensors.load()
    eglin = next(s for s in sites if s.get_name() == "Eglin")
    sensor = bh.SimpleSSNSensor.from_location(eglin)
    assert sensor.name == "Eglin"
    assert sensor.az_min == pytest.approx(145.0)
    assert sensor.az_max == pytest.approx(215.0)
    assert sensor.el_min == pytest.approx(1.0)
    assert sensor.range_max == pytest.approx(13_210_000.0)
    assert sensor.calibrated


def test_from_location_optical_and_uncalibrated_load():
    sites = bh.datasets.ssn_sensors.load()
    # Optical site now constructs as an angles-only (2-dim) sensor whose
    # matching model is the AzElMeasurementModel.
    socorro = next(s for s in sites if s.get_name() == "Socorro")
    sensor = bh.SimpleSSNSensor.from_location(socorro)
    assert sensor.measurement_dim == 2
    assert sensor.calibrated
    model = sensor.measurement_model()
    assert model.name() == "AzEl"
    assert model.measurement_dim() == 2

    # Uncalibrated azel_range site loads with zero noise instead of erroring.
    haystack = next(s for s in sites if s.get_name() == "Haystack")
    sensor = bh.SimpleSSNSensor.from_location(haystack)
    assert not sensor.calibrated
    assert sensor.measurement_dim == 3
    r = sensor.measurement_model().noise_covariance()
    assert r[0, 0] == pytest.approx(0.0)


def test_sensor_type_enum():
    sites = bh.datasets.ssn_sensors.load()
    eglin = next(s for s in sites if s.get_name() == "Eglin")
    socorro = next(s for s in sites if s.get_name() == "Socorro")
    radar = bh.SimpleSSNSensor.from_location(eglin)
    optical = bh.SimpleSSNSensor.from_location(socorro)
    assert radar.sensor_type == bh.SensorType.AZEL_RANGE
    assert optical.sensor_type == bh.SensorType.AZEL
    assert optical.sensor_type != bh.SensorType.AZEL_RANGE
    assert repr(bh.SensorType.AZEL) == "SensorType.AZEL"
    assert str(bh.SensorType.AZEL_RANGE) == "AzElRange"


def test_with_noise_override():
    sites = bh.datasets.ssn_sensors.load()
    hax = next(s for s in sites if s.get_name() == "HAX")
    sensor = bh.SimpleSSNSensor.from_location(hax).with_noise(0.05, 0.06, 120.0)
    r = sensor.measurement_model().noise_covariance()
    assert r[0, 0] == pytest.approx(0.05**2)
    assert r[2, 2] == pytest.approx(120.0**2)
    with pytest.raises(ValueError):
        bh.SimpleSSNSensor.from_location(hax).with_noise(-1.0, 0.0, 0.0)


def test_with_bias_override(test_epoch):
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    base = bh.SimpleSSNSensor(loc, noise=(0.02, 0.02, 50.0), seed=1)
    biased = base.with_bias(0.1, -0.05, 25.0)
    state = make_state_above(test_epoch, -71.49, 42.62, 500e3)
    z_base = base.measurement_model().predict(test_epoch, state)
    z_biased = biased.measurement_model().predict(test_epoch, state)
    # predict() returns geometry + bias, so the difference is the bias delta
    # (the base sensor carries zero bias).
    assert z_biased[0] - z_base[0] == pytest.approx(0.1, abs=1e-9)
    assert z_biased[1] - z_base[1] == pytest.approx(-0.05, abs=1e-9)
    assert z_biased[2] - z_base[2] == pytest.approx(25.0, abs=1e-6)
    with pytest.raises(ValueError, match="finite"):
        base.with_bias(float("nan"), 0.0, 0.0)


def test_visible_accepts_position_only_state(test_epoch):
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(loc, el_min=5.0, noise=(0.02, 0.02, 50.0), seed=1)
    full = make_state_above(test_epoch, -71.49, 42.62, 500e3)
    pos_only = full[:3]
    assert sensor.visible(test_epoch, pos_only)
    assert sensor.visible(test_epoch, pos_only) == sensor.visible(test_epoch, full)


def test_construction_and_repr():
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(
        loc,
        el_min=5.0,
        range_max=5_000_000.0,
        bias=(0.01, 0.005, 100.0),
        noise=(0.02, 0.02, 50.0),
        seed=1,
    )
    assert sensor.name == "Test"
    assert sensor.location.get_name() == "Test"
    assert sensor.el_min == pytest.approx(5.0)
    assert sensor.el_max == pytest.approx(90.0)
    assert sensor.az_min == pytest.approx(0.0)
    assert sensor.az_max == pytest.approx(360.0)
    assert sensor.range_max == pytest.approx(5_000_000.0)
    assert "SimpleSSNSensor" in repr(sensor)


def test_construction_rejects_negative_noise():
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("BadNoise")
    with pytest.raises(ValueError, match="noise"):
        bh.SimpleSSNSensor(loc, noise=(0.02, -0.02, 50.0))


def test_state_length_validation(test_epoch):
    # A state array shorter than 3 elements must raise ValueError on every
    # geometry method, not trip the Rust length assertion.
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(loc, noise=(0.02, 0.02, 50.0), seed=1)
    short = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="at least 3 elements"):
        sensor.visible(test_epoch, short)
    with pytest.raises(ValueError, match="at least 3 elements"):
        sensor.measure(test_epoch, short)
    with pytest.raises(ValueError, match="at least 3 elements"):
        sensor.azelrange(test_epoch, short)


def test_measure_seeded_and_visible(test_epoch):
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    state = make_state_above(test_epoch, -71.49, 42.62, 500e3)

    s1 = bh.SimpleSSNSensor(loc, noise=(0.02, 0.02, 50.0), seed=1)
    s2 = bh.SimpleSSNSensor(loc, noise=(0.02, 0.02, 50.0), seed=1)
    assert s1.visible(test_epoch, state)

    z1 = s1.measure(test_epoch, state)
    z2 = s2.measure(test_epoch, state)
    np.testing.assert_array_equal(z1, z2)
    assert z1[1] > 89.5

    # Not visible from the antipode (below the horizon).
    far = make_state_above(test_epoch, 108.51, -42.62, 500e3)
    assert not s1.visible(test_epoch, far)
    assert s1.measure(test_epoch, far) is None


def test_azelrange_matches_measure_geometry(test_epoch):
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(
        loc, noise=(0.0, 0.0, 0.0), bias=(0.0, 0.0, 0.0), seed=1
    )
    state = make_state_above(test_epoch, -71.0, 43.0, 700e3)
    truth = sensor.azelrange(test_epoch, state)
    z = sensor.measure(test_epoch, state)
    np.testing.assert_allclose(z, truth, atol=1e-6)


def test_simulate_observations_and_model(test_epoch):
    oe = np.array([bh.R_EARTH + 700e3, 0.001, 55.0, 0.0, 0.0, 0.0])
    state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    prop = bh.NumericalOrbitPropagator(
        test_epoch,
        state0,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )
    prop.propagate_to(test_epoch + 3600.0)

    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(loc, el_min=5.0, noise=(0.02, 0.02, 50.0), seed=3)
    obs = sensor.simulate_observations(
        prop.trajectory, test_epoch, test_epoch + 3600.0, 15.0, 0
    )
    assert len(obs) > 0
    for o in obs:
        assert o.model_index == 0

    model = sensor.measurement_model()
    assert model.measurement_dim() == 3
    assert model.name() == "AzElRange"

    with pytest.raises(ValueError):
        sensor.simulate_observations(
            prop.trajectory, test_epoch, test_epoch + 3600.0, 0.0, 0
        )


def test_azimuth_window_wrap(test_epoch):
    # Cape-Cod-style window 347 -> 227 crosses north: azimuths through north
    # are accepted, azimuths in the southern gap (227, 347) are rejected.
    loc = bh.PointLocation(-70.54, 41.75, 80.3).with_name("CapeCod")
    sensor = bh.SimpleSSNSensor(
        loc,
        az_min=347.0,
        az_max=227.0,
        el_min=3.0,
        el_max=85.0,
        noise=(0.01, 0.01, 10.0),
        seed=1,
    )

    # Target due north of the station -> azimuth ~0/360, inside the window.
    north = make_state_above(test_epoch, -70.54, 43.75, 500e3)
    az_north = sensor.azelrange(test_epoch, north)[0]
    assert az_north < 20.0 or az_north > 340.0
    assert sensor.visible(test_epoch, north)
    assert sensor.measure(test_epoch, north) is not None

    # Target due west -> azimuth ~270, in the rejected southern gap.
    west = make_state_above(test_epoch, -72.54, 41.75, 500e3)
    az_west = sensor.azelrange(test_epoch, west)[0]
    assert 227.0 < az_west < 347.0
    assert not sensor.visible(test_epoch, west)
    assert sensor.measure(test_epoch, west) is None


def test_range_cap_rejects_distant_target(test_epoch):
    # Overhead target beyond the range cap -> not visible, measure() is None.
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("ShortRange")
    sensor = bh.SimpleSSNSensor(loc, range_max=400e3, noise=(0.01, 0.01, 10.0), seed=1)
    overhead = make_state_above(test_epoch, -71.49, 42.62, 500e3)
    assert not sensor.visible(test_epoch, overhead)
    assert sensor.measure(test_epoch, overhead) is None


def test_simulate_observations_matches_stepwise(test_epoch):
    # Batched simulate_observations with a given seed must reproduce the
    # step-wise measure() output exactly: same epochs, measurements, and index.
    oe = np.array([bh.R_EARTH + 700e3, 0.001, 55.0, 0.0, 0.0, 0.0])
    state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    prop = bh.NumericalOrbitPropagator(
        test_epoch,
        state0,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )
    prop.propagate_to(test_epoch + 3600.0)
    traj = prop.trajectory

    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    start, end, dt = test_epoch, test_epoch + 3600.0, 15.0

    batched = bh.SimpleSSNSensor(loc, el_min=5.0, noise=(0.02, 0.02, 50.0), seed=99)
    obs = batched.simulate_observations(traj, start, end, dt, 3)

    stepwise = bh.SimpleSSNSensor(loc, el_min=5.0, noise=(0.02, 0.02, 50.0), seed=99)
    manual = []
    t = start
    while (t - end) <= 1e-9:
        z = stepwise.measure(t, traj.interpolate(t))
        if z is not None:
            manual.append((t, z))
        t = t + dt

    assert len(obs) == len(manual) > 0
    for o, (t_manual, z) in zip(obs, manual):
        assert o.epoch == t_manual
        np.testing.assert_array_equal(o.measurement, z)
        assert o.model_index == 3


def test_with_constraint_rejects_everything(test_epoch):
    # A user-supplied constraint that always fails makes the target invisible,
    # so measure returns None even when the standard limits would admit it.
    loc = bh.PointLocation(-71.49, 42.62, 123.1).with_name("Test")
    sensor = bh.SimpleSSNSensor(loc, noise=(0.02, 0.02, 50.0), seed=1)
    state = make_state_above(test_epoch, -71.49, 42.62, 500e3)
    assert sensor.visible(test_epoch, state)

    # An impossible elevation window (>= 91 deg) rejects everything.
    reject_all = bh.ElevationConstraint(min_elevation_deg=91.0, max_elevation_deg=None)
    extended = sensor.with_constraint(reject_all)
    assert not extended.visible(test_epoch, state)
    assert extended.measure(test_epoch, state) is None


def _optical_sensor(lon, lat, alt, name, seed):
    """Build a co-located optical sensor via dataset-style properties."""
    loc = bh.PointLocation(lon, lat, alt).with_name(name)
    loc.properties["sensor_type"] = "azel"
    loc.properties["el_min_deg"] = 5.0
    loc.properties["az_noise_deg"] = 0.005
    loc.properties["el_noise_deg"] = 0.005
    return bh.SimpleSSNSensor.from_location(loc, seed=seed)


def test_optical_sensor_measures_angles_only(test_epoch):
    # An optical sensor produces 2-dim [az, el] measurements and an AzEl model.
    sensor = _optical_sensor(-106.66, 33.82, 1510.2, "Optical", 7)
    assert sensor.measurement_dim == 2
    state = make_state_above(test_epoch, -106.66, 33.82, 500e3)
    z = sensor.measure(test_epoch, state)
    assert z is not None
    assert len(z) == 2
    assert z[1] > 89.0  # near zenith
    model = sensor.measurement_model()
    assert model.name() == "AzEl"
    assert model.measurement_dim() == 2


def test_mixed_network_ekf_processes_both_models(test_epoch):
    # A co-located radar (az/el/range) and optical (az/el) sensor feed an EKF
    # via distinct model indices; the mixed models must process without error
    # and reduce the position error over a short arc.
    oe = np.array([bh.R_EARTH + 700e3, 0.001, 55.0, 0.0, 0.0, 0.0])
    state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    prop = bh.NumericalOrbitPropagator(
        test_epoch,
        state0,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )
    end = test_epoch + 2 * 3600.0
    prop.propagate_to(end)
    traj = prop.trajectory

    lon, lat, alt = -71.49, 42.62, 123.1
    radar_loc = bh.PointLocation(lon, lat, alt).with_name("Radar")
    radar = bh.SimpleSSNSensor(
        radar_loc, el_min=5.0, range_max=5_000_000.0, noise=(0.01, 0.01, 25.0), seed=11
    )
    optical = _optical_sensor(lon, lat, alt, "Optical", 23)

    observations = radar.simulate_observations(traj, test_epoch, end, 30.0, 0)
    observations += optical.simulate_observations(traj, test_epoch, end, 30.0, 1)
    observations.sort(key=lambda o: o.epoch)
    assert any(o.model_index == 0 for o in observations)
    assert any(o.model_index == 1 for o in observations)

    initial = np.array(state0)
    initial[0] += 500.0
    initial[4] += 0.5
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    ekf = bh.ExtendedKalmanFilter(
        test_epoch,
        initial,
        p0,
        measurement_models=[radar.measurement_model(), optical.measurement_model()],
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
    )
    for obs in observations:
        ekf.process_observation(obs)

    truth_final = traj.interpolate(ekf.current_epoch())
    final_err = np.linalg.norm(ekf.current_state()[:3] - truth_final[:3])
    assert final_err < 500.0

    names = {rec.measurement_name for rec in ekf.records()}
    assert "AzElRange" in names
    assert "AzEl" in names
