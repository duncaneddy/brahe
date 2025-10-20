"""Tests for groundstation datasets module"""

import pytest
import brahe as bh


def test_list_providers():
    """Test listing available providers"""
    providers = bh.datasets.groundstations.list_providers()
    assert len(providers) > 0
    assert "ksat" in providers
    assert "atlas" in providers
    assert "aws" in providers
    assert "leaf" in providers
    assert "ssc" in providers
    assert "viasat" in providers


def test_load_ksat_groundstations():
    """Test loading KSAT groundstations"""
    stations = bh.datasets.groundstations.load("ksat")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_atlas_groundstations():
    """Test loading Atlas groundstations"""
    stations = bh.datasets.groundstations.load("atlas")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_aws_groundstations():
    """Test loading AWS groundstations"""
    stations = bh.datasets.groundstations.load("aws")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_leaf_groundstations():
    """Test loading Leaf groundstations"""
    stations = bh.datasets.groundstations.load("leaf")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_ssc_groundstations():
    """Test loading SSC groundstations"""
    stations = bh.datasets.groundstations.load("ssc")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_viasat_groundstations():
    """Test loading Viasat groundstations"""
    stations = bh.datasets.groundstations.load("viasat")
    assert len(stations) > 0
    assert all(isinstance(s, bh.PointLocation) for s in stations)


def test_load_invalid_provider():
    """Test error handling for invalid provider"""
    with pytest.raises(RuntimeError, match="Unknown groundstation provider"):
        bh.datasets.groundstations.load("nonexistent")


def test_case_insensitive_provider():
    """Test that provider names are case-insensitive"""
    stations1 = bh.datasets.groundstations.load("KSAT")
    stations2 = bh.datasets.groundstations.load("ksat")
    stations3 = bh.datasets.groundstations.load("KsAt")

    assert len(stations1) == len(stations2)
    assert len(stations1) == len(stations3)


def test_load_all_groundstations():
    """Test loading all groundstations at once"""
    all_stations = bh.datasets.groundstations.load_all()
    assert len(all_stations) > 10  # Should have multiple stations

    # Verify total count matches sum of individual providers
    providers = bh.datasets.groundstations.list_providers()
    total = 0
    for provider in providers:
        stations = bh.datasets.groundstations.load(provider)
        total += len(stations)

    assert len(all_stations) == total


def test_groundstation_has_name():
    """Test that groundstations have names"""
    stations = bh.datasets.groundstations.load("ksat")

    for station in stations:
        name = station.get_name()
        assert name is not None
        assert len(name) > 0


def test_groundstation_coordinates():
    """Test that groundstations have valid coordinates"""
    stations = bh.datasets.groundstations.load("ksat")

    for station in stations:
        # Should have valid longitude
        assert -180.0 <= station.lon <= 180.0

        # Should have valid latitude
        assert -90.0 <= station.lat <= 90.0

        # Altitude should be reasonable (not all zeros or negatives)
        assert station.alt >= -500.0  # Allow some below sea level


def test_groundstation_properties():
    """Test that groundstations have expected properties"""
    stations = bh.datasets.groundstations.load("ksat")

    for station in stations:
        props = station.properties

        # Should have provider property
        assert "provider" in props
        assert props["provider"] == "KSAT"

        # Should have frequency_bands property
        assert "frequency_bands" in props
        assert isinstance(props["frequency_bands"], list)
        assert len(props["frequency_bands"]) > 0


def test_groundstation_properties_for_all_providers():
    """Test that all providers have consistent property structure"""
    providers = bh.datasets.groundstations.list_providers()

    for provider in providers:
        stations = bh.datasets.groundstations.load(provider)

        for station in stations:
            props = station.properties

            # All should have provider
            assert "provider" in props

            # All should have frequency_bands
            assert "frequency_bands" in props
            assert isinstance(props["frequency_bands"], list)


def test_load_all_groups_by_provider():
    """Test that load_all returns stations from all providers"""
    all_stations = bh.datasets.groundstations.load_all()

    # Group by provider
    providers_found = set()
    for station in all_stations:
        props = station.properties
        provider = props.get("provider", "Unknown")
        providers_found.add(provider.lower())

    # Should have multiple providers
    assert len(providers_found) >= 6
