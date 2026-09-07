"""Tests for NAIF body name resolution."""

import pytest

import brahe as bh


def test_naifid_from_name_round_trip():
    for member in [
        bh.NAIFId.SOLAR_SYSTEM_BARYCENTER,
        bh.NAIFId.EARTH_MOON_BARYCENTER,
        bh.NAIFId.EARTH,
        bh.NAIFId.MOON,
        bh.NAIFId.MARS,
        bh.NAIFId.MARS_BARYCENTER,
        bh.NAIFId.TITAN,
    ]:
        assert bh.NAIFId.from_name(member.naif_name) == member

    assert bh.NAIFId.EARTH.naif_name == "EARTH"
    assert bh.NAIFId.EARTH_MOON_BARYCENTER.naif_name == "EARTH MOON BARYCENTER"
    assert bh.NAIFId.from_name(" mars barycenter ") == bh.NAIFId.MARS_BARYCENTER
    assert (
        bh.NAIFId.from_name("EARTH_MOON_BARYCENTER") == bh.NAIFId.EARTH_MOON_BARYCENTER
    )
    assert bh.naif_id_from_name("mars barycenter") == bh.NAIFId.MARS_BARYCENTER
    assert bh.naif_name(bh.NAIFId.EARTH) == "EARTH"


def test_naifid_from_name_integer_and_error():
    assert bh.NAIFId.from_name("2000001") == 2000001
    assert bh.naif_name(2000001) == "2000001"

    with pytest.raises(bh.BraheError, match="PLANET X"):
        bh.NAIFId.from_name("PLANET X")

    with pytest.raises(bh.BraheError, match="PLANET X"):
        bh.naif_id_from_name("PLANET X")


def test_naifid_from_name_aliases():
    """Mirror of test_naifid_from_name_aliases in Rust."""
    assert bh.NAIFId.from_name("SSB") == bh.NAIFId.SOLAR_SYSTEM_BARYCENTER
    assert bh.NAIFId.from_name(" ssb ") == bh.NAIFId.SOLAR_SYSTEM_BARYCENTER
    assert bh.NAIFId.from_name("emb") == bh.NAIFId.EARTH_MOON_BARYCENTER
    assert (
        bh.NAIFId.from_name("EARTH         BARYCENTER")
        == bh.NAIFId.EARTH_MOON_BARYCENTER
    )
    assert bh.NAIFId.from_name("Mars_Barycenter") == bh.NAIFId.MARS_BARYCENTER
    assert (
        bh.NAIFId.from_name("EARTH\tMOON\nBARYCENTER")
        == bh.NAIFId.EARTH_MOON_BARYCENTER
    )

    # Aliases are input spellings only; the canonical name is unchanged.
    assert bh.NAIFId.SOLAR_SYSTEM_BARYCENTER.naif_name == "SOLAR SYSTEM BARYCENTER"
    assert bh.NAIFId.EARTH_MOON_BARYCENTER.naif_name == "EARTH MOON BARYCENTER"
    assert bh.NAIFId.MARS_BARYCENTER.naif_name == "MARS BARYCENTER"
