"""Tests for CCSDS OEM parsing and writing — parity with Rust tests."""

import pytest
from brahe.ccsds import OEM


def test_oem_parse_example1(eop):
    """Mirror of test_parse_oem_example1 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    # Header
    assert oem.format_version() == pytest.approx(3.0, abs=1e-10)
    assert oem.classification() == "public, test-data"
    assert oem.originator() == "NASA/JPL"

    # 3 segments
    assert oem.num_segments() == 3

    # Segment 0 metadata
    assert oem.object_name(0) == "MARS GLOBAL SURVEYOR"
    assert oem.object_id(0) == "1996-062A"
    assert oem.center_name(0) == "MARS BARYCENTER"
    assert oem.ref_frame(0) == "J2000"

    # Segment 0 states
    assert oem.num_states(0) == 4

    # First state: position in km converted to meters
    sv = oem.state(0, 0)
    assert sv["position"][0] == pytest.approx(2789.619 * 1000.0, abs=1.0)
    assert sv["position"][1] == pytest.approx(-280.045 * 1000.0, abs=1.0)
    assert sv["position"][2] == pytest.approx(-1746.755 * 1000.0, abs=1.0)
    assert sv["velocity"][0] == pytest.approx(4.73372 * 1000.0, abs=1.0)
    assert sv["velocity"][1] == pytest.approx(-2.49586 * 1000.0, abs=1.0)
    assert sv["velocity"][2] == pytest.approx(-1.04195 * 1000.0, abs=1.0)

    # Segment 0 no covariance
    assert oem.num_covariances(0) == 0

    # Segment 1 has covariance
    assert oem.num_covariances(1) == 1

    # Segment 2 has 2 covariance blocks
    assert oem.num_covariances(2) == 2


def test_oem_parse_example2_doy_format(eop):
    """Mirror of test_parse_oem_example2_doy_format in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample2.txt")

    assert oem.format_version() == pytest.approx(2.0, abs=1e-10)
    assert oem.num_segments() == 2
    assert oem.ref_frame(0) == "TOD"
    assert oem.num_states(0) == 4


def test_oem_parse_example4(eop):
    """Mirror of test_parse_oem_example4 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")

    assert oem.format_version() == pytest.approx(2.0, abs=1e-10)
    assert oem.num_segments() == 1
    assert oem.object_name(0) == "MARS GLOBAL SURVEYOR"
    assert oem.center_name(0) == "MARS"
    assert oem.ref_frame(0) == "EME2000"
    assert oem.num_states(0) == 3


def test_oem_parse_example5_gcrf(eop):
    """Mirror of test_parse_oem_example5_gcrf in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")

    assert oem.num_segments() == 1
    assert oem.ref_frame(0) == "GCRF"
    assert oem.object_name(0) == "ISS"
    assert oem.object_id(0) == "1998-067A"
    assert oem.num_states(0) == 49


def test_oem_parse_xml_example3(eop):
    """Mirror of test_parse_oem_xml_example3 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample3.xml")

    assert oem.format_version() == pytest.approx(3.0, abs=1e-10)
    assert oem.originator() == "NASA/JPL"
    assert oem.num_segments() == 1
    assert oem.object_name(0) == "MARS GLOBAL SURVEYOR"
    assert oem.num_states(0) == 4


def test_oem_round_trip_kvn(eop):
    """Mirror of test_oem_kvn_round_trip_example1 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    written = oem.to_string("KVN")
    oem2 = OEM.from_str(written)

    assert oem2.num_segments() == oem.num_segments()
    assert oem2.originator() == oem.originator()

    for i in range(oem.num_segments()):
        assert oem2.object_name(i) == oem.object_name(i)
        assert oem2.num_states(i) == oem.num_states(i)


def test_oem_to_dict(eop):
    """Test to_dict() serialization."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    d = oem.to_dict()

    assert d["header"]["originator"] == "NASA/JPL"
    assert d["header"]["format_version"] == pytest.approx(2.0)
    assert len(d["segments"]) == 1

    seg = d["segments"][0]
    assert seg["metadata"]["object_name"] == "MARS GLOBAL SURVEYOR"
    assert seg["metadata"]["ref_frame"] == "EME2000"
    assert len(seg["states"]) == 3
    assert seg["states"][0]["position"][0] == pytest.approx(2789619.0, abs=1.0)


def test_oem_repr(eop):
    """Test OEM repr."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    r = repr(oem)
    assert "segments=3" in r
    assert "NASA/JPL" in r
