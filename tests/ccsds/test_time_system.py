"""Tests for CCSDS epoch time-system handling — parity with Rust tests in src/ccsds/common.rs."""

import brahe  # noqa: F401
from brahe.ccsds import OEM, OMM, OPM

FORMATS = ["kvn", "xml", "json"]


def test_write_oem_epochs_follow_the_declared_time_system(eop):
    """Mirror of test_write_oem_epochs_follow_the_declared_time_system in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    assert oem.segments[0].time_system == "UTC"
    assert "START_TIME = 1996-12-18T12:00:00.331" in oem.to_string("kvn")

    # Redeclaring TIME_SYSTEM moves the written epochs onto that scale.
    oem.segments[0].time_system = "TAI"
    assert "START_TIME = 1996-12-18T12:00:30.331" in oem.to_string("kvn")


def test_write_omm_epochs_follow_the_declared_time_system(eop):
    """The OMM EPOCH keyword follows the metadata TIME_SYSTEM."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample2.txt")
    assert omm.time_system == "UTC"
    assert "EPOCH = 2007-03-05T10:34:41.4264" in omm.to_string("kvn")

    omm.time_system = "TAI"
    assert "EPOCH = 2007-03-05T10:35:14.4264" in omm.to_string("kvn")


def test_write_opm_epochs_follow_the_declared_time_system(eop):
    """The OPM EPOCH keyword follows the metadata TIME_SYSTEM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    assert opm.time_system == "UTC"
    utc = opm.to_string("kvn")

    opm.time_system = "TAI"
    assert opm.to_string("kvn") != utc


def test_writers_use_declared_time_system_not_epoch_time_system(eop):
    """Mirror of test_writers_use_declared_time_system_not_epoch_time_system in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    # The declared TIME_SYSTEM, not the stored epoch, decides the written value,
    # so the epochs a message reports agree with the strings it writes.
    for fmt in FORMATS:
        written = oem.to_string(fmt)
        assert OEM.from_str(written).segments[0].time_system == "UTC"
    assert oem.to_dict()["segments"][0]["metadata"]["start_time"].startswith(
        "1996-12-18T12:00:00"
    )

    oem.segments[0].time_system = "TAI"
    assert oem.to_dict()["segments"][0]["metadata"]["start_time"].startswith(
        "1996-12-18T12:00:30"
    )


def test_cdm_writes_time_tags_in_utc(eop):
    """Python half of test_writers_use_declared_time_system_not_epoch_time_system."""
    from brahe.ccsds import CDM

    # CCSDS 508.0-B-1 subsection 6.2.3.4 fixes every CDM time tag to UTC, and
    # PyCDM.__repr__ reports the same instant the writers emit.
    cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")

    # Compared as written time codes: an Epoch carries sub-nanosecond
    # representation noise that a CCSDS time code does not encode.
    for fmt in FORMATS:
        assert "2010-03-13T22:37:52.618" in CDM.from_str(cdm.to_string(fmt)).to_string(
            "kvn"
        )
    assert "2010-03-13T22:37:52" in repr(cdm)


def test_ref_frame_epoch_survives_a_non_utc_time_system(eop):
    """Mirror of test_ref_frame_epoch_survives_a_non_utc_time_system in Rust."""
    # REF_FRAME_EPOCH and START_TIME are read in the declared TIME_SYSTEM, but
    # neither the KVN keyword order nor JSON key order guarantees the
    # declaration comes first.
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    oem.segments[0].time_system = "TAI"
    start = oem.segments[0].start_time

    for fmt in FORMATS:
        reparsed = OEM.from_str(oem.to_string(fmt))
        assert reparsed.segments[0].time_system == "TAI", (
            f"{fmt} did not round-trip the declared time system"
        )
        assert reparsed.segments[0].start_time == start, f"{fmt} shifted START_TIME"
