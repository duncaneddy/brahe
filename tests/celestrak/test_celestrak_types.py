"""Tests for Celestrak type enum Python bindings."""

import brahe as bh


class TestCelestrakQueryType:
    """Tests for CelestrakQueryType enum."""

    def test_gp_variant(self):
        qt = bh.celestrak.CelestrakQueryType.GP
        assert str(qt) == "gp"

    def test_sup_gp_variant(self):
        qt = bh.celestrak.CelestrakQueryType.SUP_GP
        assert str(qt) == "sup_gp"

    def test_satcat_variant(self):
        qt = bh.celestrak.CelestrakQueryType.SATCAT
        assert str(qt) == "satcat"

    def test_equality(self):
        assert bh.celestrak.CelestrakQueryType.GP == bh.celestrak.CelestrakQueryType.GP
        assert (
            bh.celestrak.CelestrakQueryType.GP != bh.celestrak.CelestrakQueryType.SATCAT
        )

    def test_repr(self):
        qt = bh.celestrak.CelestrakQueryType.GP
        assert "GP" in repr(qt)


class TestCelestrakOutputFormat:
    """Tests for CelestrakOutputFormat enum."""

    def test_all_variants(self):
        assert str(bh.celestrak.CelestrakOutputFormat.TLE) == "TLE"
        assert str(bh.celestrak.CelestrakOutputFormat.TWO_LE) == "2LE"
        assert str(bh.celestrak.CelestrakOutputFormat.THREE_LE) == "3LE"
        assert str(bh.celestrak.CelestrakOutputFormat.XML) == "XML"
        assert str(bh.celestrak.CelestrakOutputFormat.KVN) == "KVN"
        assert str(bh.celestrak.CelestrakOutputFormat.JSON) == "JSON"
        assert str(bh.celestrak.CelestrakOutputFormat.JSON_PRETTY) == "JSON-PRETTY"
        assert str(bh.celestrak.CelestrakOutputFormat.CSV) == "CSV"

    def test_equality(self):
        assert (
            bh.celestrak.CelestrakOutputFormat.JSON
            == bh.celestrak.CelestrakOutputFormat.JSON
        )
        assert (
            bh.celestrak.CelestrakOutputFormat.JSON
            != bh.celestrak.CelestrakOutputFormat.CSV
        )

    def test_repr(self):
        fmt = bh.celestrak.CelestrakOutputFormat.JSON
        assert "Json" in repr(fmt)


class TestSupGPSource:
    """Tests for SupGPSource enum."""

    def test_key_variants(self):
        assert str(bh.celestrak.SupGPSource.SPACEX) == "spacex"
        assert str(bh.celestrak.SupGPSource.SPACEX_SUP) == "spacex-sup"
        assert str(bh.celestrak.SupGPSource.PLANET) == "planet"
        assert str(bh.celestrak.SupGPSource.ONEWEB) == "oneweb"
        assert str(bh.celestrak.SupGPSource.STARLINK) == "starlink"
        assert str(bh.celestrak.SupGPSource.STARLINK_SUP) == "starlink-sup"
        assert str(bh.celestrak.SupGPSource.IRIDIUM) == "iridium"
        assert str(bh.celestrak.SupGPSource.IRIDIUM_NEXT) == "iridium-next"
        assert str(bh.celestrak.SupGPSource.ORBCOMM) == "orbcomm"
        assert str(bh.celestrak.SupGPSource.GLOBALSTAR) == "globalstar"

    def test_additional_variants(self):
        assert str(bh.celestrak.SupGPSource.GEO) == "geo"
        assert str(bh.celestrak.SupGPSource.GPS) == "gps"
        assert str(bh.celestrak.SupGPSource.GLONASS) == "glonass"
        assert str(bh.celestrak.SupGPSource.METEOSAT) == "meteosat"
        assert str(bh.celestrak.SupGPSource.INTELSAT) == "intelsat"
        assert str(bh.celestrak.SupGPSource.SES) == "ses"
        assert str(bh.celestrak.SupGPSource.SWARM_TECHNOLOGIES) == "swarm"
        assert str(bh.celestrak.SupGPSource.AMATEUR) == "amateur"
        assert str(bh.celestrak.SupGPSource.CELESTRAK) == "celestrak"

    def test_equality(self):
        assert bh.celestrak.SupGPSource.SPACEX == bh.celestrak.SupGPSource.SPACEX
        assert bh.celestrak.SupGPSource.SPACEX != bh.celestrak.SupGPSource.PLANET

    def test_repr(self):
        src = bh.celestrak.SupGPSource.SPACEX
        assert "SpaceX" in repr(src)
