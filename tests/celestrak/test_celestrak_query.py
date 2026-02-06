"""Tests for CelestrakQuery Python bindings."""

import brahe as bh
from brahe.spacetrack import operators as op


class TestCelestrakQueryConstructors:
    """Tests for CelestrakQuery static constructors."""

    def test_gp_constructor(self):
        query = bh.celestrak.CelestrakQuery.gp()
        assert query.build_url() == ""
        assert "GP" in repr(query)

    def test_sup_gp_constructor(self):
        query = bh.celestrak.CelestrakQuery.sup_gp()
        assert query.build_url() == ""
        assert "SupGP" in repr(query)

    def test_satcat_constructor(self):
        query = bh.celestrak.CelestrakQuery.satcat()
        assert query.build_url() == ""
        assert "SATCAT" in repr(query)


class TestCelestrakQueryGP:
    """Tests for GP query builder methods."""

    def test_gp_by_group(self):
        query = bh.celestrak.CelestrakQuery.gp().group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_gp_by_catnr(self):
        query = bh.celestrak.CelestrakQuery.gp().catnr(25544)
        assert "CATNR=25544" in query.build_url()

    def test_gp_by_intdes(self):
        query = bh.celestrak.CelestrakQuery.gp().intdes("1998-067A")
        assert "INTDES=1998-067A" in query.build_url()

    def test_gp_by_name(self):
        query = bh.celestrak.CelestrakQuery.gp().name_search("ISS")
        assert "NAME=ISS" in query.build_url()

    def test_gp_with_format(self):
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("stations")
            .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
        )
        assert "GROUP=stations" in query.build_url()
        assert "FORMAT=3LE" in query.build_url()

    def test_gp_with_special(self):
        query = bh.celestrak.CelestrakQuery.gp().special("test")
        assert "SPECIAL=test" in query.build_url()


class TestCelestrakQuerySupGP:
    """Tests for SupGP query builder methods."""

    def test_sup_gp_by_source(self):
        query = bh.celestrak.CelestrakQuery.sup_gp().source(
            bh.celestrak.SupGPSource.SPACEX
        )
        assert "SOURCE=spacex" in query.build_url()

    def test_sup_gp_by_file(self):
        query = bh.celestrak.CelestrakQuery.sup_gp().file("test.txt")
        assert "FILE=test.txt" in query.build_url()

    def test_sup_gp_by_catnr(self):
        query = bh.celestrak.CelestrakQuery.sup_gp().catnr(25544)
        assert "CATNR=25544" in query.build_url()

    def test_sup_gp_with_format(self):
        query = (
            bh.celestrak.CelestrakQuery.sup_gp()
            .source(bh.celestrak.SupGPSource.STARLINK)
            .format(bh.celestrak.CelestrakOutputFormat.JSON)
        )
        url = query.build_url()
        assert "SOURCE=starlink" in url
        assert "FORMAT=JSON" in url


class TestCelestrakQuerySATCAT:
    """Tests for SATCAT query builder methods."""

    def test_satcat_active(self):
        query = bh.celestrak.CelestrakQuery.satcat().active(True)
        assert "ACTIVE=Y" in query.build_url()

    def test_satcat_payloads(self):
        query = bh.celestrak.CelestrakQuery.satcat().payloads(True)
        assert "PAYLOADS=Y" in query.build_url()

    def test_satcat_on_orbit(self):
        query = bh.celestrak.CelestrakQuery.satcat().on_orbit(True)
        assert "ONORBIT=Y" in query.build_url()

    def test_satcat_max(self):
        query = bh.celestrak.CelestrakQuery.satcat().max(100)
        assert "MAX=100" in query.build_url()

    def test_satcat_by_group(self):
        query = bh.celestrak.CelestrakQuery.satcat().group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_satcat_by_name(self):
        query = bh.celestrak.CelestrakQuery.satcat().name_search("ISS")
        assert "NAME=ISS" in query.build_url()

    def test_satcat_multiple_flags(self):
        query = (
            bh.celestrak.CelestrakQuery.satcat()
            .active(True)
            .payloads(True)
            .on_orbit(True)
        )
        url = query.build_url()
        assert "PAYLOADS=Y" in url
        assert "ONORBIT=Y" in url
        assert "ACTIVE=Y" in url

    def test_satcat_false_flags_not_in_url(self):
        query = bh.celestrak.CelestrakQuery.satcat().active(False)
        assert "ACTIVE" not in query.build_url()


class TestCelestrakQueryClientSide:
    """Tests for client-side filter/order/limit methods."""

    def test_filter(self):
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("active")
            .filter("INCLINATION", ">50")
        )
        # Filters are client-side, not in URL
        assert "GROUP=active" in query.build_url()

    def test_order_by(self):
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("active")
            .order_by("INCLINATION", False)
        )
        assert "GROUP=active" in query.build_url()

    def test_limit(self):
        query = bh.celestrak.CelestrakQuery.gp().group("active").limit(10)
        assert "GROUP=active" in query.build_url()

    def test_filter_with_operators(self):
        """Test that SpaceTrack operators work as filter values."""
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("active")
            .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
            .filter("INCLINATION", op.greater_than("50"))
            .filter("NORAD_CAT_ID", op.inclusive_range("25544", "25600"))
        )
        assert "GROUP=active" in query.build_url()


class TestCelestrakQueryImmutability:
    """Test that builder methods return new instances (Python pattern)."""

    def test_group_does_not_mutate(self):
        base = bh.celestrak.CelestrakQuery.gp()
        with_group = base.group("stations")
        assert base.build_url() == ""
        assert "GROUP=stations" in with_group.build_url()

    def test_filter_does_not_mutate(self):
        base = bh.celestrak.CelestrakQuery.gp().group("active")
        with_filter = base.filter("INCLINATION", ">50")
        # Both have GROUP in URL
        assert "GROUP=active" in base.build_url()
        assert "GROUP=active" in with_filter.build_url()

    def test_chaining(self):
        """Test fluent method chaining."""
        query = (
            bh.celestrak.CelestrakQuery.gp()
            .group("stations")
            .format(bh.celestrak.CelestrakOutputFormat.JSON)
            .filter("INCLINATION", ">50")
            .order_by("INCLINATION", True)
            .limit(5)
        )
        url = query.build_url()
        assert "GROUP=stations" in url
        assert "FORMAT=JSON" in url
