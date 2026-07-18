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
    assert len(sensors) == 13
    names = {s.name for s in sensors}
    assert "Eglin" in names
    assert "Millstone" in names
    assert "Socorro" not in names  # optical site is skipped
    assert "Haystack" not in names  # no calibration values


def test_from_location_dataset_roundtrip():
    sites = bh.datasets.ssn_sensors.load()
    eglin = next(s for s in sites if s.get_name() == "Eglin")
    sensor = bh.SimpleSSNSensor.from_location(eglin)
    assert sensor.name == "Eglin"
    assert sensor.az_min == pytest.approx(145.0)
    assert sensor.az_max == pytest.approx(215.0)
    assert sensor.el_min == pytest.approx(1.0)
    assert sensor.range_max == pytest.approx(13_210_000.0)


def test_from_location_errors_on_unsupported_site():
    sites = bh.datasets.ssn_sensors.load()
    socorro = next(s for s in sites if s.get_name() == "Socorro")
    with pytest.raises(ValueError):
        bh.SimpleSSNSensor.from_location(socorro)

    haystack = next(s for s in sites if s.get_name() == "Haystack")
    with pytest.raises(ValueError):
        bh.SimpleSSNSensor.from_location(haystack)


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
