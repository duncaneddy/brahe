"""Tests for CelestrakClient Python bindings."""

import pytest

import brahe as bh


class TestCelestrakClientConstruction:
    """Tests for CelestrakClient construction."""

    def test_default_construction(self):
        client = bh.celestrak.CelestrakClient()
        assert client is not None

    def test_with_cache_age(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=3600.0)
        assert client is not None

    def test_with_base_url(self):
        client = bh.celestrak.CelestrakClient(base_url="https://test.celestrak.org")
        assert client is not None

    def test_with_base_url_and_cache_age(self):
        client = bh.celestrak.CelestrakClient(
            base_url="https://test.celestrak.org", cache_max_age=1800.0
        )
        assert client is not None


class TestCelestrakSATCATRecord:
    """Tests for CelestrakSATCATRecord attributes."""

    # Note: We can't directly construct CelestrakSATCATRecord from Python,
    # but we verify the class is importable and repr works
    def test_import(self):
        assert hasattr(bh.celestrak, "CelestrakSATCATRecord")


# -- CI-gated integration tests --


@pytest.mark.ci
class TestCelestrakClientIntegration:
    """Integration tests against live Celestrak API."""

    def test_query_gp_by_group(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp().group("stations")
        records = client.query_gp(query)
        assert len(records) > 0
        # All records should have object_name
        for r in records:
            assert r.object_name is not None

    def test_query_gp_by_catnr(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp().catnr(25544)
        records = client.query_gp(query)
        assert len(records) > 0
        assert records[0].norad_cat_id == "25544"

    def test_query_gp_by_name(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp().name_search("ISS")
        records = client.query_gp(query)
        assert len(records) > 0

    def test_query_gp_returns_gprecord(self):
        """Verify GP queries return GPRecord (same as SpaceTrack)."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp().catnr(25544)
        records = client.query_gp(query)
        assert len(records) > 0
        # Should be GPRecord type
        assert isinstance(records[0], bh.GPRecord)
        # Should have standard GPRecord fields
        assert records[0].norad_cat_id is not None
        assert records[0].inclination is not None

    def test_query_satcat(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.satcat().catnr(25544)
        records = client.query_satcat(query)
        assert len(records) > 0
        assert records[0].norad_cat_id == "25544"
        assert records[0].object_name is not None

    def test_query_raw(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .catnr(25544)
            .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
        )
        result = client.query_raw(query)
        assert "25544" in result

    def test_query_gp_with_client_side_filter(self):
        """Test client-side filtering on live data."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("stations")
            .filter("INCLINATION", ">50")
        )
        records = client.query_gp(query)
        # All returned records should have inclination > 50
        for r in records:
            if r.inclination is not None:
                assert float(r.inclination) > 50.0

    def test_query_gp_with_limit(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp().group("stations").limit(2)
        records = client.query_gp(query)
        assert len(records) <= 2
