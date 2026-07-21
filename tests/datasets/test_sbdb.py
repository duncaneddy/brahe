"""Tests for the SBDB Lookup Python bindings."""

import pytest

import brahe as bh


class TestSBDBConstruction:
    def test_default(self):
        client = bh.datasets.sbdb.SBDBClient()
        assert client is not None

    def test_with_base_url(self):
        client = bh.datasets.sbdb.SBDBClient(base_url="https://example.test")
        assert client is not None

    def test_with_cache_age(self):
        client = bh.datasets.sbdb.SBDBClient(cache_max_age=0)
        assert client is not None

    def test_with_base_url_and_cache_age(self):
        client = bh.datasets.sbdb.SBDBClient(
            base_url="https://example.test", cache_max_age=0
        )
        assert client is not None

    def test_object_type_present(self):
        assert hasattr(bh.datasets.sbdb, "SBDBObject")


@pytest.mark.integration
class TestSBDBIntegration:
    def test_lookup_ceres(self):
        client = bh.datasets.sbdb.SBDBClient(cache_max_age=0)
        obj = client.lookup("Ceres")
        assert obj.spkid == 2000001
        assert obj.naif_id() == 2000001
        assert "Ceres" in obj.full_name
        assert obj.gm is not None and obj.gm > 6.0e10
        assert obj.radius is not None and obj.radius > 400e3
