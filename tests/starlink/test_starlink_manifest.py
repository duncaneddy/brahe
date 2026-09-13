"""Tests mirroring src/clients/starlink/manifest.rs."""

from pathlib import Path

import polars as pl
import pytest

import brahe as bh

FIXTURE = (
    Path(__file__).resolve().parents[2] / "test_assets" / "starlink" / "MANIFEST.txt"
)


def utc(*args):
    return bh.Epoch(*args, 0, time_system=bh.UTC)


def reference():
    return utc(2026, 9, 11, 6, 30, 0.0)


def test_manifest_parse_fixture():
    """Rust: test_manifest_parse_fixture"""
    manifest = bh.StarlinkManifest.parse(
        FIXTURE.read_text(), reference(), utc(2026, 9, 11, 5, 15, 30.0)
    )
    assert len(manifest) == 5
    assert manifest.retrieved == reference()
    assert manifest.last_modified == utc(2026, 9, 11, 5, 15, 30.0)
    first = manifest.entries()[0]
    assert first.norad_cat_id == 100001
    assert first.object_name == "STARLINK-38128"
    assert first.category == bh.SpaceTrackEphemerisFileCategory.OPERATIONAL
    assert first.ephemeris_start == utc(2026, 9, 11, 1, 42, 0.0)
    assert first.ephemeris_stop == utc(2026, 9, 14, 1, 42, 42.0)
    assert (
        first.file_name_string()
        == "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
    )
    assert first.file_name.norad_cat_id == 100001
    assert manifest.find_by_norad_id(100002).object_name == "STARLINK-37711"
    assert manifest.find_by_norad_id(1) is None
    assert manifest.find_by_object_name("STARLINK-38123").norad_cat_id == 100003
    assert manifest.find_by_object_name("starlink-38123") is None
    assert len(list(manifest)) == 5
    assert manifest[1].norad_cat_id == 100002


def test_manifest_parse_skips_blank_lines_and_rejects_malformed():
    """Rust: test_manifest_parse_skips_blank_lines_and_rejects_malformed"""
    text = "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n\n   \nMEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt\n"
    assert len(bh.StarlinkManifest.parse(text, reference())) == 2
    with pytest.raises(bh.BraheError):
        bh.StarlinkManifest.parse(
            "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\nnot-a-file-name\n",
            reference(),
        )
    assert bh.StarlinkManifest.parse("", reference()).is_empty()


def test_manifest_entry_without_gps_metadata_infers_year_from_reference():
    """Rust: test_manifest_entry_without_gps_metadata_infers_year_from_reference"""
    m = bh.StarlinkManifest.parse(
        "MEME_25544_ISS_2540142_Operational_nomnvr_UNCLASSIFIED.txt\n", reference()
    )
    assert m.entries()[0].ephemeris_stop is None
    assert m.entries()[0].ephemeris_start == utc(2026, 9, 11, 1, 42, 0.0)
    m = bh.StarlinkManifest.parse(
        "MEME_25544_ISS_3652300_Operational_nomnvr_UNCLASSIFIED.txt\n",
        utc(2027, 1, 2, 0, 0, 0.0),
    )
    assert m.entries()[0].ephemeris_start == utc(2026, 12, 31, 23, 0, 0.0)
    m = bh.StarlinkManifest.parse(
        "MEME_25544_ISS_2540142_Operational_42_UNCLASSIFIED.txt\n", reference()
    )
    assert m.entries()[0].ephemeris_stop is None
    assert m.entries()[0].ephemeris_start == utc(2026, 9, 11, 1, 42, 0.0)
    leap = "MEME_25544_ISS_0600000_Operational_nomnvr_UNCLASSIFIED.txt\n"
    assert bh.StarlinkManifest.parse(leap, utc(2024, 3, 1, 0, 0, 0.0)).entries()[
        0
    ].ephemeris_start == utc(2024, 2, 29, 0, 0, 0.0)
    assert bh.StarlinkManifest.parse(leap, utc(2026, 3, 1, 0, 0, 0.0)).entries()[
        0
    ].ephemeris_start == utc(2026, 3, 1, 0, 0, 0.0)
    day_366 = "MEME_25544_ISS_3660000_Operational_nomnvr_UNCLASSIFIED.txt\n"
    m = bh.StarlinkManifest.parse(day_366, utc(2025, 1, 2, 0, 0, 0.0))
    assert m.entries()[0].ephemeris_start == utc(2024, 12, 31, 0, 0, 0.0)
    with pytest.raises(bh.BraheError):
        bh.StarlinkManifest.parse(day_366, utc(2027, 1, 2, 0, 0, 0.0))
    old_listing = "MEME_25544_ISS_0010000_Operational_nomnvr_UNCLASSIFIED.txt\n"
    m = bh.StarlinkManifest.parse(old_listing, utc(2026, 12, 30, 0, 0, 0.0))
    assert m.entries()[0].ephemeris_start == utc(2026, 1, 1, 0, 0, 0.0)


def test_manifest_year_from_gps_stop_across_new_year():
    """Rust: test_manifest_year_from_gps_stop_across_new_year"""
    stop = utc(2027, 1, 2, 1, 42, 42.0)
    gps = round(stop.gps_seconds())
    m = bh.StarlinkManifest.parse(
        f"MEME_100001_STARLINK-38128_3650142_Operational_{gps}_UNCLASSIFIED.txt\n",
        utc(2027, 1, 2, 3, 0, 0.0),
    )
    assert m.entries()[0].ephemeris_stop == stop
    assert m.entries()[0].ephemeris_start == utc(2026, 12, 31, 1, 42, 0.0)


def test_manifest_changed_since():
    """Rust: test_manifest_changed_since"""
    old = FIXTURE.read_text()
    a = bh.StarlinkManifest.parse(old, reference())
    lines = old.splitlines()
    lines[1] = (
        "MEME_100002_STARLINK-37711_2540949_Operational_1473414600_UNCLASSIFIED.txt"
    )
    del lines[4]
    lines.append(
        "MEME_100006_STARLINK-38117_2540137_Operational_1473385080_UNCLASSIFIED.txt"
    )
    b = bh.StarlinkManifest.parse("\n".join(lines), reference() + 3600.0)
    assert [e.norad_cat_id for e in b.changed_since(a)] == [100002, 100006]
    assert [e.norad_cat_id for e in a.changed_since(b)] == [100002, 100005]
    assert a.changed_since(a) == []


def test_manifest_to_dataframe():
    """Rust: test_manifest_to_dataframe"""
    df = bh.StarlinkManifest.parse(FIXTURE.read_text(), reference()).to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 5
    assert df.columns == [
        "norad_cat_id",
        "object_name",
        "category",
        "ephemeris_start",
        "ephemeris_stop",
        "file_name",
    ]
    assert df["norad_cat_id"].dtype == pl.UInt32
    assert df["norad_cat_id"][0] == 100001
    assert df["object_name"][0] == "STARLINK-38128"
    assert df["category"][0] == "Operational"
    assert df["ephemeris_start"].dtype == pl.Datetime("ms")
    assert df["ephemeris_start"][0].isoformat() == "2026-09-11T01:42:00"
    assert df["ephemeris_stop"].null_count() == 0
    df2 = bh.StarlinkManifest.parse(
        "MEME_25544_ISS_2540142_Operational_nomnvr_UNCLASSIFIED.txt\n", reference()
    ).to_dataframe()
    assert df2["ephemeris_stop"].null_count() == 1


def test_manifest_entry_keeps_listing_file_name():
    """Rust: test_manifest_entry_keeps_listing_file_name"""
    line = "MEME_1001_STARLINK-1_2540142_Operational_nomnvr_UNCLASSIFIED.txt"
    m = bh.StarlinkManifest.parse(f"{line}\n", reference())
    entry = m.find_by_norad_id(1001)
    assert entry.file_name_string() == line
    assert str(entry.file_name) == line.replace("_1001_", "_01001_")
    assert m.to_dataframe()["file_name"][0] == line
    respelled = "MEME_1001_STARLINK-1_2540142_oper_nomnvr_UNCLASSIFIED.txt"
    other = bh.StarlinkManifest.parse(f"{respelled}\n", reference())
    assert len(other.changed_since(m)) == 1
    assert m.changed_since(m) == []


def test_manifest_entry_rejects_oversized_gps_metadata():
    """Rust: test_manifest_entry_rejects_oversized_gps_metadata"""
    m = bh.StarlinkManifest.parse(
        "MEME_100001_STARLINK-38128_2540142_Operational_10000000000000000000_UNCLASSIFIED.txt\n",
        reference(),
    )
    entry = m.entries()[0]
    assert entry.ephemeris_stop is None
    assert entry.ephemeris_start == utc(2026, 9, 11, 1, 42, 0.0)


def test_manifest_module_exports():
    assert bh.starlink.StarlinkManifest is bh.StarlinkManifest
    assert bh.clients.starlink.StarlinkClient is bh.StarlinkClient
    assert "StarlinkManifestEntry" in bh.starlink.__all__
    assert bh.starlink.RateLimitConfig is bh.RateLimitConfig
