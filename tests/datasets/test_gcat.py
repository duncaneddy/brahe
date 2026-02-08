"""Tests for GCAT Python bindings."""

import pytest
import polars as pl

import brahe.datasets as datasets


# ─── CI tests (live downloads) ──────────────────────────────────────


@pytest.mark.ci
class TestGCATLiveDownload:
    """Tests that require network access to download GCAT data."""

    def test_get_satcat(self):
        """Test downloading and parsing GCAT SATCAT."""
        satcat = datasets.gcat.get_satcat()
        assert len(satcat) > 0
        assert repr(satcat).startswith("GCATSatcat(")

    def test_get_psatcat(self):
        """Test downloading and parsing GCAT PSATCAT."""
        psatcat = datasets.gcat.get_psatcat()
        assert len(psatcat) > 0
        assert repr(psatcat).startswith("GCATPsatcat(")

    def test_satcat_lookup_iss(self):
        """Test looking up ISS by SATCAT number."""
        satcat = datasets.gcat.get_satcat()

        iss = satcat.get_by_satcat("25544")
        assert iss is not None
        assert iss.jcat is not None
        assert "ISS" in iss.name or "Zarya" in iss.name
        assert iss.inc is not None
        assert iss.perigee is not None

    def test_satcat_search_by_name(self):
        """Test name search returns results."""
        satcat = datasets.gcat.get_satcat()

        results = satcat.search_by_name("ISS")
        assert len(results) > 0

    def test_satcat_filter_by_type(self):
        """Test filtering by object type."""
        satcat = datasets.gcat.get_satcat()

        payloads = satcat.filter_by_type("P")
        assert len(payloads) > 0
        # All records should be payload type
        for r in payloads.records()[:10]:
            assert r.object_type == "P"

    def test_satcat_filter_by_status(self):
        """Test filtering by status."""
        satcat = datasets.gcat.get_satcat()

        operational = satcat.filter_by_status("O")
        assert len(operational) > 0

    def test_satcat_filter_chaining(self):
        """Test chaining multiple filters."""
        satcat = datasets.gcat.get_satcat()

        result = satcat.filter_by_type("P").filter_by_status("O").filter_by_state("US")
        assert len(result) > 0
        # Original should be unchanged
        assert len(satcat) > len(result)

    def test_satcat_filter_by_perigee_range(self):
        """Test filtering by perigee range."""
        satcat = datasets.gcat.get_satcat()

        leo = satcat.filter_by_perigee_range(200.0, 2000.0)
        assert len(leo) > 0

    def test_satcat_filter_by_inc_range(self):
        """Test filtering by inclination range."""
        satcat = datasets.gcat.get_satcat()

        polar = satcat.filter_by_inc_range(85.0, 100.0)
        assert len(polar) > 0

    def test_satcat_to_dataframe(self):
        """Test converting SATCAT to Polars DataFrame."""
        satcat = datasets.gcat.get_satcat()

        df = satcat.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(satcat)
        assert "jcat" in df.columns
        assert "satcat" in df.columns
        assert "name" in df.columns
        assert "perigee" in df.columns
        assert "apogee" in df.columns
        assert "inc" in df.columns

    def test_satcat_record_repr(self):
        """Test record repr."""
        satcat = datasets.gcat.get_satcat()
        iss = satcat.get_by_satcat("25544")
        assert iss is not None
        r = repr(iss)
        assert "GCATSatcatRecord" in r
        assert "25544" in r

    def test_psatcat_lookup(self):
        """Test PSATCAT lookup by JCAT."""
        psatcat = datasets.gcat.get_psatcat()
        records = psatcat.records()
        assert len(records) > 0

        # Lookup first record by its jcat
        first_jcat = records[0].jcat
        found = psatcat.get_by_jcat(first_jcat)
        assert found is not None
        assert found.jcat == first_jcat

    def test_psatcat_filter_by_category(self):
        """Test PSATCAT filtering by category."""
        psatcat = datasets.gcat.get_psatcat()

        comms = psatcat.filter_by_category("COM")
        assert len(comms) > 0

    def test_psatcat_filter_active(self):
        """Test PSATCAT active filter."""
        psatcat = datasets.gcat.get_psatcat()

        active = psatcat.filter_active()
        assert len(active) > 0

    def test_psatcat_to_dataframe(self):
        """Test converting PSATCAT to Polars DataFrame."""
        psatcat = datasets.gcat.get_psatcat()

        df = psatcat.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(psatcat)
        assert "jcat" in df.columns
        assert "name" in df.columns
        assert "program" in df.columns
        assert "category" in df.columns

    def test_psatcat_search_by_name(self):
        """Test PSATCAT name search."""
        psatcat = datasets.gcat.get_psatcat()

        results = psatcat.search_by_name("Starlink")
        assert len(results) > 0

    def test_psatcat_filter_by_class(self):
        """Test PSATCAT filter by class."""
        psatcat = datasets.gcat.get_psatcat()

        b_class = psatcat.filter_by_class("B")
        assert len(b_class) > 0

    def test_psatcat_filter_by_result(self):
        """Test PSATCAT filter by result."""
        psatcat = datasets.gcat.get_psatcat()

        success = psatcat.filter_by_result("S")
        assert len(success) > 0

    def test_get_satcat_nonexistent_lookup(self):
        """Test lookup returns None for non-existent records."""
        satcat = datasets.gcat.get_satcat()

        assert satcat.get_by_jcat("NONEXISTENT") is None
        assert satcat.get_by_satcat("99999999") is None

    def test_psatcat_record_fields(self):
        """Test PSATCAT record has expected fields."""
        psatcat = datasets.gcat.get_psatcat()
        records = psatcat.records()
        assert len(records) > 0

        r = records[0]
        # Check all expected attributes exist
        assert hasattr(r, "jcat")
        assert hasattr(r, "piece")
        assert hasattr(r, "name")
        assert hasattr(r, "ldate")
        assert hasattr(r, "program")
        assert hasattr(r, "category")
        assert hasattr(r, "result")
        assert hasattr(r, "un_period")
        assert hasattr(r, "comment")

    def test_satcat_record_fields(self):
        """Test SATCAT record has expected fields."""
        satcat = datasets.gcat.get_satcat()
        iss = satcat.get_by_satcat("25544")
        assert iss is not None

        # Check all expected attributes exist
        assert hasattr(iss, "jcat")
        assert hasattr(iss, "satcat")
        assert hasattr(iss, "object_type")
        assert hasattr(iss, "name")
        assert hasattr(iss, "status")
        assert hasattr(iss, "owner")
        assert hasattr(iss, "state")
        assert hasattr(iss, "mass")
        assert hasattr(iss, "perigee")
        assert hasattr(iss, "apogee")
        assert hasattr(iss, "inc")
        assert hasattr(iss, "alt_names")
