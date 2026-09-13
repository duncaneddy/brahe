"""Tests for Space-Track ephemeris file names — parity with src/clients/spacetrack/file_name.rs."""

import pytest

import brahe as bh
from brahe.spacetrack import (
    SpaceTrackEphemerisFileCategory,
    SpaceTrackEphemerisFileName,
)


def test_ephemeris_file_name_parse_starlink():
    name = SpaceTrackEphemerisFileName.parse(
        "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
    )
    assert name.data_type == "MEME"
    assert name.norad_cat_id == 100001
    assert name.object_name == "STARLINK-38128"
    assert (name.day_of_year, name.hour, name.minute) == (254, 1, 42)
    assert name.category == SpaceTrackEphemerisFileCategory.OPERATIONAL
    assert name.metadata == "1473385380"
    assert name.classification == "UNCLASSIFIED"
    assert name.extension == "txt"


def test_ephemeris_file_name_parse_handbook_examples():
    a = SpaceTrackEphemerisFileName.parse(
        "MEME_25544_ISS_1651200_oper__unclassified.txt"
    )
    assert (a.norad_cat_id, a.object_name, a.metadata, a.classification) == (
        25544,
        "ISS",
        "",
        "unclassified",
    )
    assert (a.day_of_year, a.hour, a.minute) == (165, 12, 0)
    b = SpaceTrackEphemerisFileName.parse(
        "MEME_25544_ISS(ZARYA)_1651200_operational_nomnvr_UNCLASSIFIED.txt"
    )
    assert (b.object_name, b.metadata) == ("ISS(ZARYA)", "nomnvr")
    c = SpaceTrackEphemerisFileName.parse(
        "MEME_799500234_Sat1_1651200_special_separation_unclassified.txt"
    )
    assert (c.norad_cat_id, c.category, c.metadata) == (
        799500234,
        SpaceTrackEphemerisFileCategory.SPECIAL,
        "separation",
    )


def test_ephemeris_file_name_parse_object_name_with_underscores():
    name = SpaceTrackEphemerisFileName.parse(
        "MEME_12345_MY_SAT_A_0010530_Special_burn02_UNCLASSIFIED.txt"
    )
    assert name.object_name == "MY_SAT_A"
    assert (name.day_of_year, name.hour, name.minute) == (1, 5, 30)


@pytest.mark.parametrize(
    "bad",
    [
        "MEME_25544_ISS_1651200_oper__unclassified",
        "MEME_25544_ISS_1651200_oper.txt",
        "MEME_abc_ISS_1651200_oper__unclassified.txt",
        "MEME_25544_ISS_165120_oper__unclassified.txt",
        "MEME_25544_ISS_1652500_oper__unclassified.txt",
        "MEME_25544_ISS_1651260_oper__unclassified.txt",
        "MEME_25544_ISS_3671200_oper__unclassified.txt",
        "MEME_25544_ISS_1651200_planned__unclassified.txt",
        "MEME_25544__1651200_oper__unclassified.txt",
    ],
)
def test_ephemeris_file_name_parse_errors(bad):
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileName.parse(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "MEME_100001_../../../../tmp/evil_2540142_Operational__UNCLASSIFIED.txt",
        "/tmp/pwn_100001_X_2540142_Operational__UNCLASSIFIED.txt",
        "MEME_100001_..\\..\\tmp\\evil_2540142_Operational__UNCLASSIFIED.txt",
        "MEME_100001_.._2540142_Operational__UNCLASSIFIED.txt",
        "MEME_100001_X_2540142_Operational_../evil_UNCLASSIFIED.txt",
        "MEME_100001_X_2540142_Operational__UNCLASSIFIED.txt/../evil",
        "MEME_100001_X_2540142_Operational__UNCLASS/IFIED.txt",
        "MEME_100001_X_2540142_Operational__UNCLASSIFIED.t\0xt",
        "MEME_100001_X_2540142_oper/../../../evil_meta_UNCLASSIFIED.txt",
    ],
)
def test_ephemeris_file_name_parse_rejects_path_traversal(bad):
    """Rust: test_ephemeris_file_name_parse_rejects_path_traversal"""
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileName.parse(bad)


def test_ephemeris_file_name_builders_reject_path_traversal():
    """Rust: test_ephemeris_file_name_builders_reject_path_traversal"""
    start = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0)
    for object_name in ["..", "../../tmp/evil", "A\\B"]:
        with pytest.raises(bh.BraheError):
            SpaceTrackEphemerisFileName(
                1, object_name, start, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
            )
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileName(
            1, "A", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, "../evil"
        )

    name = SpaceTrackEphemerisFileName(
        1, "A", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
    )
    with pytest.raises(bh.BraheError):
        name.with_data_type("..")
    with pytest.raises(bh.BraheError):
        name.with_classification("A\\B")
    with pytest.raises(bh.BraheError):
        name.with_extension("t\0xt")


def test_ephemeris_file_name_new_and_display():
    start = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0)
    name = SpaceTrackEphemerisFileName(
        100001,
        "STARLINK-38128",
        start,
        SpaceTrackEphemerisFileCategory.OPERATIONAL,
        "1473385380",
    )
    assert (
        str(name)
        == "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
    )
    padded = SpaceTrackEphemerisFileName(
        900, "CALSPHERE 1", start, SpaceTrackEphemerisFileCategory.SPECIAL, ""
    )
    assert str(padded) == "MEME_00900_CALSPHERE 1_2540142_Special__UNCLASSIFIED.txt"
    custom = (
        SpaceTrackEphemerisFileName(
            25544, "ISS", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, "nomnvr"
        )
        .with_data_type("TEME")
        .with_classification("unclassified")
        .with_extension("dat")
    )
    assert str(custom) == "TEME_25544_ISS_2540142_Operational_nomnvr_unclassified.dat"


@pytest.mark.parametrize(
    "original",
    [
        "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt",
        "MEME_69995_STARLINK-38084_2540139_Operational_1473385200_UNCLASSIFIED.txt",
        "MEME_799501571_STARLINK-36331_2540207_Operational_1473386880_UNCLASSIFIED.txt",
    ],
)
def test_ephemeris_file_name_round_trip(original):
    assert str(SpaceTrackEphemerisFileName.parse(original)) == original


def test_ephemeris_file_name_uses_utc_for_day_time_group():
    start = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0)
    in_tai = bh.Epoch(2026, 9, 11, 1, 43, 19.0, 0.0, time_system=bh.TimeSystem.TAI)
    a = SpaceTrackEphemerisFileName(
        1, "A", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
    )
    b = SpaceTrackEphemerisFileName(
        1, "A", in_tai, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
    )
    assert str(a) == str(b)


def test_ephemeris_file_name_rejects_delimiter_in_fields():
    start = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0)
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileName(
            1, "A", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, "burn_02"
        )
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileName(
            1, "", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
        )

    name = SpaceTrackEphemerisFileName(
        1, "A", start, SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
    )
    with pytest.raises(bh.BraheError):
        name.with_extension("txt.bak")
    with pytest.raises(bh.BraheError):
        name.with_classification("UN_CLASS")
    with pytest.raises(bh.BraheError):
        name.with_data_type("MEME/EXTRA")


def test_ephemeris_file_category_parse_and_display():
    assert (
        SpaceTrackEphemerisFileCategory.parse("oper")
        == SpaceTrackEphemerisFileCategory.OPERATIONAL
    )
    assert (
        SpaceTrackEphemerisFileCategory.parse("OPERATIONAL")
        == SpaceTrackEphemerisFileCategory.OPERATIONAL
    )
    assert (
        SpaceTrackEphemerisFileCategory.parse("Special")
        == SpaceTrackEphemerisFileCategory.SPECIAL
    )
    with pytest.raises(bh.BraheError):
        SpaceTrackEphemerisFileCategory.parse("planned")
    assert str(SpaceTrackEphemerisFileCategory.OPERATIONAL) == "Operational"
    assert str(SpaceTrackEphemerisFileCategory.SPECIAL) == "Special"
