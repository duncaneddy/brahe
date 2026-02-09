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


class TestCelestrakQueryClassattrs:
    """Tests for CelestrakQuery class attribute constructors."""

    def test_gp_classattr(self):
        query = bh.celestrak.CelestrakQuery.gp
        assert query is not None
        assert "CelestrakQuery" in repr(query)

    def test_sup_gp_classattr(self):
        query = bh.celestrak.CelestrakQuery.sup_gp
        assert query is not None

    def test_satcat_classattr(self):
        query = bh.celestrak.CelestrakQuery.satcat
        assert query is not None

    def test_gp_chaining(self):
        query = bh.celestrak.CelestrakQuery.gp.group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_satcat_chaining(self):
        query = bh.celestrak.CelestrakQuery.satcat.active(True)
        assert "ACTIVE=Y" in query.build_url()


class TestGetGpValidation:
    """Tests for get_gp() argument validation."""

    def test_get_gp_no_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp()

    def test_get_gp_multiple_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp(catnr=25544, name="ISS")


class TestGetSatcatValidation:
    """Tests for get_satcat() argument validation."""

    def test_get_satcat_no_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="at least one"):
            client.get_satcat()


class TestCelestrakSATCATRecord:
    """Tests for CelestrakSATCATRecord attributes."""

    def test_import(self):
        assert hasattr(bh.celestrak, "CelestrakSATCATRecord")


# -- CI-gated integration tests --


@pytest.mark.ci
class TestCelestrakClientIntegration:
    """Integration tests against live Celestrak API."""

    def test_get_gp_by_group(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(group="stations")
        assert len(records) > 0
        for r in records:
            assert r.object_name is not None

    def test_get_gp_by_catnr(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(catnr=25544)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544

    def test_get_gp_by_name(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(name="ISS")
        assert len(records) > 0

    def test_get_gp_returns_gprecord(self):
        """Verify GP queries return GPRecord (same as SpaceTrack)."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(catnr=25544)
        assert len(records) > 0
        assert isinstance(records[0], bh.GPRecord)
        assert records[0].norad_cat_id is not None
        assert records[0].inclination is not None

    def test_get_satcat(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_satcat(catnr=25544)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544
        assert records[0].object_name is not None

    def test_query_raw(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.catnr(25544).format(
            bh.celestrak.CelestrakOutputFormat.THREE_LE
        )
        result = client.query_raw(query)
        assert "25544" in result

    def test_query_gp_with_filter(self):
        """Test query() with client-side filtering on live data."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.group("stations").filter(
            "INCLINATION", ">50"
        )
        records = client.query(query)
        for r in records:
            if r.inclination is not None:
                assert float(r.inclination) > 50.0

    def test_query_gp_with_limit(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.group("stations").limit(2)
        records = client.query(query)
        assert len(records) <= 2

    def test_query_satcat(self):
        """Test query() dispatch for SATCAT queries."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.satcat.catnr(25544)
        records = client.query(query)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544

    def test_get_sgp_propagator(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)
        assert propagator is not None
        assert isinstance(propagator, bh.SGPPropagator)
