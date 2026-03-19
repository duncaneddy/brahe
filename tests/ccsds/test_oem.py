"""Tests for CCSDS OEM parsing and writing — parity with Rust tests."""

import pytest
from brahe.ccsds import OEM


def test_oem_parse_example1(eop):
    """Mirror of test_parse_oem_example1 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    # Header properties
    assert oem.format_version == pytest.approx(3.0, abs=1e-10)
    assert oem.classification == "public, test-data"
    assert oem.originator == "NASA/JPL"

    # 3 segments via .segments property
    assert len(oem.segments) == 3

    # Segment 0 metadata via indexing
    seg0 = oem.segments[0]
    assert seg0.object_name == "MARS GLOBAL SURVEYOR"
    assert seg0.object_id == "1996-062A"
    assert seg0.center_name == "MARS BARYCENTER"
    assert seg0.ref_frame == "J2000"

    # Segment 0 states
    assert seg0.num_states == 4

    # First state via .states indexing
    sv = seg0.states[0]
    assert sv["position"][0] == pytest.approx(2789.619 * 1000.0, abs=1.0)
    assert sv["position"][1] == pytest.approx(-280.045 * 1000.0, abs=1.0)
    assert sv["position"][2] == pytest.approx(-1746.755 * 1000.0, abs=1.0)
    assert sv["velocity"][0] == pytest.approx(4.73372 * 1000.0, abs=1.0)
    assert sv["velocity"][1] == pytest.approx(-2.49586 * 1000.0, abs=1.0)
    assert sv["velocity"][2] == pytest.approx(-1.04195 * 1000.0, abs=1.0)

    # Segment 0 no covariance
    assert seg0.num_covariances == 0

    # Segment 1 has covariance
    assert oem.segments[1].num_covariances == 1

    # Segment 2 has 2 covariance blocks
    assert oem.segments[2].num_covariances == 2


def test_oem_parse_example2_doy_format(eop):
    """Mirror of test_parse_oem_example2_doy_format in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample2.txt")

    assert oem.format_version == pytest.approx(2.0, abs=1e-10)
    assert len(oem.segments) == 2
    assert oem.segments[0].ref_frame == "TOD"
    assert oem.segments[0].num_states == 4


def test_oem_parse_example4(eop):
    """Mirror of test_parse_oem_example4 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")

    assert oem.format_version == pytest.approx(2.0, abs=1e-10)
    assert len(oem.segments) == 1
    assert oem.segments[0].object_name == "MARS GLOBAL SURVEYOR"
    assert oem.segments[0].center_name == "MARS"
    assert oem.segments[0].ref_frame == "EME2000"
    assert oem.segments[0].num_states == 3


def test_oem_parse_example5_gcrf(eop):
    """Mirror of test_parse_oem_example5_gcrf in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")

    assert len(oem.segments) == 1
    assert oem.segments[0].ref_frame == "GCRF"
    assert oem.segments[0].object_name == "ISS"
    assert oem.segments[0].object_id == "1998-067A"
    assert oem.segments[0].num_states == 49


def test_oem_parse_xml_example3(eop):
    """Mirror of test_parse_oem_xml_example3 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample3.xml")

    assert oem.format_version == pytest.approx(3.0, abs=1e-10)
    assert oem.originator == "NASA/JPL"
    assert len(oem.segments) == 1
    assert oem.segments[0].object_name == "MARS GLOBAL SURVEYOR"
    assert oem.segments[0].num_states == 4


def test_oem_round_trip_kvn(eop):
    """Mirror of test_oem_kvn_round_trip_example1 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    written = oem.to_string("KVN")
    oem2 = OEM.from_str(written)

    assert len(oem2.segments) == len(oem.segments)
    assert oem2.originator == oem.originator

    for i in range(len(oem.segments)):
        assert oem2.segments[i].object_name == oem.segments[i].object_name
        assert oem2.segments[i].num_states == oem.segments[i].num_states


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


def test_oem_segments_iteration(eop):
    """Test iterating over segments."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    names = [seg.object_name for seg in oem.segments]
    assert len(names) == 3
    assert names[0] == "MARS GLOBAL SURVEYOR"


def test_oem_states_iteration(eop):
    """Test iterating over state vectors in a segment."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    seg = oem.segments[0]

    epochs = [sv["epoch"] for sv in seg.states]
    assert len(epochs) == 3


def test_oem_negative_indexing(eop):
    """Test negative indexing on segments and states."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    # Negative segment index
    last_seg = oem.segments[-1]
    assert last_seg.num_covariances == 2

    # Negative state index
    seg0 = oem.segments[0]
    last_state = seg0.states[-1]
    assert "position" in last_state


def test_oem_index_out_of_range(eop):
    """Test that out-of-range indexing raises IndexError."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    with pytest.raises(IndexError):
        oem.segments[100]

    with pytest.raises(IndexError):
        oem.segments[0].states[9999]


def test_oem_state_shortcut(eop):
    """Test the state() shortcut method still works."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    sv = oem.state(0, 0)
    assert sv["position"][0] == pytest.approx(2789619.0, abs=1.0)


def test_oem_segment_properties(eop):
    """Test all segment-level property accessors."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    seg = oem.segments[0]

    assert isinstance(seg.object_name, str)
    assert isinstance(seg.object_id, str)
    assert isinstance(seg.center_name, str)
    assert isinstance(seg.ref_frame, str)
    assert isinstance(seg.time_system, str)
    assert isinstance(seg.start_time, str)
    assert isinstance(seg.stop_time, str)
    assert isinstance(seg.num_states, int)
    assert isinstance(seg.num_covariances, int)
    assert isinstance(seg.comments, list)
