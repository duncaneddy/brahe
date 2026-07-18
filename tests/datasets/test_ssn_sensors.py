"""Tests for SSN sensor dataset module"""

import brahe as bh


def test_load_ssn_sensors():
    sensors = bh.datasets.ssn_sensors.load()
    assert len(sensors) == 21
    for s in sensors:
        props = s.properties
        assert "sensor_type" in props
        assert "system" in props
        assert "category" in props
        assert -180.0 <= s.lon <= 180.0
        assert -90.0 <= s.lat <= 90.0


def test_ssn_sensor_values_against_vallado_tables():
    sensors = bh.datasets.ssn_sensors.load()
    eglin = next(s for s in sensors if s.get_name() == "Eglin")
    props = eglin.properties
    assert props["sensor_type"] == "azel_range"
    assert props["range_max_m"] == 13_210_000.0
    assert props["az_min_deg"] == 145.0
    assert props["az_noise_deg"] == 0.0154

    socorro = next(s for s in sensors if s.get_name() == "Socorro")
    assert socorro.properties["sensor_type"] == "radec"
    assert "range_max_m" not in socorro.properties


def test_ssn_sensor_type_counts():
    sensors = bh.datasets.ssn_sensors.load()
    azel = [s for s in sensors if s.properties["sensor_type"] == "azel_range"]
    radec = [s for s in sensors if s.properties["sensor_type"] == "radec"]
    assert len(azel) == 15
    assert len(radec) == 6
