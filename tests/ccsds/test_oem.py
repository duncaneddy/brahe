"""Tests for CCSDS OEM parsing, writing, mutation, and construction — parity with Rust tests."""

import numpy as np
import pytest

import brahe
from brahe import Epoch
from brahe.ccsds import OEM, OEMSegment, OEMStateVector


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

    # First state via .states indexing — typed access
    sv = seg0.states[0]
    assert sv.position[0] == pytest.approx(2789.619 * 1000.0, abs=1.0)
    assert sv.position[1] == pytest.approx(-280.045 * 1000.0, abs=1.0)
    assert sv.position[2] == pytest.approx(-1746.755 * 1000.0, abs=1.0)
    assert sv.velocity[0] == pytest.approx(4.73372 * 1000.0, abs=1.0)
    assert sv.velocity[1] == pytest.approx(-2.49586 * 1000.0, abs=1.0)
    assert sv.velocity[2] == pytest.approx(-1.04195 * 1000.0, abs=1.0)

    # Epoch is a brahe Epoch object
    assert isinstance(sv.epoch, Epoch)

    # Segment 0 no covariance
    assert seg0.num_covariances == 0

    # Segment 1 has covariance
    assert oem.segments[1].num_covariances == 1

    # Segment 2 has 2 covariance blocks
    assert oem.segments[2].num_covariances == 2


def test_oem_parse_example2_unsupported_time_system(eop):
    """OEMExample2.txt uses TIME_SYSTEM=MRT which is unsupported for epoch conversion."""
    with pytest.raises(Exception, match="MRT"):
        OEM.from_file("test_assets/ccsds/oem/OEMExample2.txt")


def test_oem_json_round_trip(eop):
    """OEM JSON round-trip: from_file -> to_string(JSON) -> from_str -> compare."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    json_str = oem.to_string("JSON")
    oem2 = OEM.from_str(json_str)
    assert oem2.originator == oem.originator
    assert len(oem2.segments) == len(oem.segments)
    assert len(oem2.segments[0].states) == len(oem.segments[0].states)
    assert oem2.segments[0].object_name == oem.segments[0].object_name


def test_oem_json_uppercase_keys(eop):
    """OEM to_json_string with uppercase keys."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    json_str = oem.to_json_string(uppercase_keys=True)
    assert '"OBJECT_NAME"' in json_str
    assert '"header"' in json_str  # container keys always lowercase


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

    epochs = [sv.epoch for sv in seg.states]
    assert len(epochs) == 3
    assert all(isinstance(e, Epoch) for e in epochs)


def test_oem_negative_indexing(eop):
    """Test negative indexing on segments and states."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    # Negative segment index
    last_seg = oem.segments[-1]
    assert last_seg.num_covariances == 2

    # Negative state index
    seg0 = oem.segments[0]
    last_state = seg0.states[-1]
    assert isinstance(last_state.position, np.ndarray)


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
    assert sv.position[0] == pytest.approx(2789619.0, abs=1.0)
    assert isinstance(sv.epoch, Epoch)


def test_oem_segment_properties(eop):
    """Test all segment-level property accessors."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    seg = oem.segments[0]

    assert isinstance(seg.object_name, str)
    assert isinstance(seg.object_id, str)
    assert isinstance(seg.center_name, str)
    assert isinstance(seg.ref_frame, str)
    assert isinstance(seg.time_system, str)
    assert isinstance(seg.start_time, Epoch)
    assert isinstance(seg.stop_time, Epoch)
    assert isinstance(seg.num_states, int)
    assert isinstance(seg.num_covariances, int)
    assert isinstance(seg.comments, list)


# ─────────────────────────────────────────────
# Mutation tests
# ─────────────────────────────────────────────


def test_oem_header_setters(eop):
    """Test setting header properties."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    oem.originator = "MODIFIED"
    assert oem.originator == "MODIFIED"

    oem.classification = "secret"
    assert oem.classification == "secret"

    oem.classification = None
    assert oem.classification is None

    oem.format_version = 2.0
    assert oem.format_version == pytest.approx(2.0)

    oem.message_id = "MSG-001"
    assert oem.message_id == "MSG-001"


def test_oem_segment_setters(eop):
    """Test setting segment metadata via proxy — mutations reflect back."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    oem.segments[0].object_name = "NEW_SAT"
    assert oem.segments[0].object_name == "NEW_SAT"

    oem.segments[0].object_id = "2024-001A"
    assert oem.segments[0].object_id == "2024-001A"

    oem.segments[0].center_name = "EARTH"
    assert oem.segments[0].center_name == "EARTH"

    oem.segments[0].ref_frame = "GCRF"
    assert oem.segments[0].ref_frame == "GCRF"

    oem.segments[0].time_system = "TAI"
    assert oem.segments[0].time_system == "TAI"

    oem.segments[0].interpolation = "LAGRANGE"
    assert oem.segments[0].interpolation == "LAGRANGE"

    oem.segments[0].interpolation_degree = 5
    assert oem.segments[0].interpolation_degree == 5

    oem.segments[0].comments = ["test comment"]
    assert oem.segments[0].comments == ["test comment"]


def test_oem_state_setters(eop):
    """Test setting state vector fields via proxy — mutations reflect back."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")

    # Set position
    oem.segments[0].states[0].position = [1.0, 2.0, 3.0]
    assert oem.segments[0].states[0].position == pytest.approx([1.0, 2.0, 3.0])

    # Set velocity
    oem.segments[0].states[0].velocity = [4.0, 5.0, 6.0]
    assert oem.segments[0].states[0].velocity == pytest.approx([4.0, 5.0, 6.0])

    # Set acceleration
    oem.segments[0].states[0].acceleration = [0.1, 0.2, 0.3]
    assert oem.segments[0].states[0].acceleration == pytest.approx([0.1, 0.2, 0.3])

    # Clear acceleration
    oem.segments[0].states[0].acceleration = None
    assert oem.segments[0].states[0].acceleration is None

    # Set epoch
    new_epoch = Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.segments[0].states[0].epoch = new_epoch
    assert oem.segments[0].states[0].epoch == new_epoch


def test_oem_segment_start_stop_time_setters(eop):
    """Test setting start/stop times on segments."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    new_start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    new_stop = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    oem.segments[0].start_time = new_start
    oem.segments[0].stop_time = new_stop

    assert oem.segments[0].start_time == new_start
    assert oem.segments[0].stop_time == new_stop


# ─────────────────────────────────────────────
# Construction tests
# ─────────────────────────────────────────────


def test_oem_construct_from_scratch(eop):
    """Test constructing an OEM from scratch."""
    oem = OEM(originator="MY_ORG")
    assert oem.originator == "MY_ORG"
    assert oem.format_version == pytest.approx(3.0)
    assert len(oem.segments) == 0
    assert isinstance(oem.creation_date, Epoch)


def test_oem_construct_with_options(eop):
    """Test constructing an OEM with optional parameters."""
    oem = OEM(originator="MY_ORG", format_version=2.0, classification="public")
    assert oem.format_version == pytest.approx(2.0)
    assert oem.classification == "public"


def test_oem_add_segment(eop):
    """Test adding segments to an OEM."""
    oem = OEM(originator="TEST")
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    idx = oem.add_segment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    assert idx == 0
    assert len(oem.segments) == 1
    assert oem.segments[0].object_name == "SAT1"
    assert oem.segments[0].ref_frame == "GCRF"
    assert oem.segments[0].start_time == start
    assert oem.segments[0].stop_time == stop


def test_oem_add_state(eop):
    """Test adding states to a segment."""
    oem = OEM(originator="TEST")
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.add_segment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    sv_idx = oem.segments[0].add_state(
        epoch=start,
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
    )
    assert sv_idx == 0
    assert oem.segments[0].num_states == 1

    sv = oem.segments[0].states[0]
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])
    assert sv.acceleration is None
    assert sv.epoch == start


def test_oem_add_state_with_acceleration(eop):
    """Test adding a state with acceleration."""
    oem = OEM(originator="TEST")
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.add_segment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem.segments[0].add_state(
        epoch=start,
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
        acceleration=[0.001, 0.002, 0.003],
    )

    sv = oem.segments[0].states[0]
    assert sv.acceleration == pytest.approx([0.001, 0.002, 0.003])


def test_oem_remove_segment(eop):
    """Test removing a segment."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    assert len(oem.segments) == 3

    oem.remove_segment(0)
    assert len(oem.segments) == 2


def test_oem_remove_state(eop):
    """Test removing a state from a segment."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    assert oem.segments[0].num_states == 3

    oem.segments[0].remove_state(0)
    assert oem.segments[0].num_states == 2


def test_oem_construct_serialize_round_trip(eop):
    """Test constructing an OEM, serializing, and re-parsing."""
    oem = OEM(originator="ROUND_TRIP_TEST")
    start = Epoch.from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 6, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    mid = Epoch.from_datetime(2024, 6, 1, 0, 30, 0.0, 0.0, brahe.TimeSystem.UTC)

    oem.add_segment(
        object_name="TEST_SAT",
        object_id="2024-999A",
        center_name="EARTH",
        ref_frame="J2000",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    oem.segments[0].add_state(
        epoch=start,
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
    )
    oem.segments[0].add_state(
        epoch=mid,
        position=[6000e3, 3000e3, 0.0],
        velocity=[-2000.0, 6000.0, 0.0],
    )

    written = oem.to_string("KVN")
    oem2 = OEM.from_str(written)

    assert oem2.originator == "ROUND_TRIP_TEST"
    assert len(oem2.segments) == 1
    assert oem2.segments[0].object_name == "TEST_SAT"
    assert oem2.segments[0].num_states == 2
    assert oem2.segments[0].states[0].position[0] == pytest.approx(7000e3, abs=1.0)
    assert oem2.segments[0].states[1].position[0] == pytest.approx(6000e3, abs=1.0)


def test_oem_state_vector_validation(eop):
    """Test validation of state vector dimensions."""
    oem = OEM(originator="TEST")
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.add_segment(
        object_name="SAT",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    with pytest.raises(ValueError, match="length 3"):
        oem.segments[0].add_state(
            epoch=start,
            position=[1.0, 2.0],  # only 2 elements
            velocity=[0.0, 0.0, 0.0],
        )


# ─────────────────────────────────────────────
# Standalone construction tests
# ─────────────────────────────────────────────


def test_OEMStateVector_standalone_construction(eop):
    """Test creating a standalone OEMStateVector."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(
        epoch=epoch,
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
    )
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])
    assert sv.acceleration is None
    assert sv.epoch == epoch

    # With acceleration
    sv2 = OEMStateVector(
        epoch=epoch,
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
        acceleration=[0.001, 0.002, 0.003],
    )
    assert sv2.acceleration == pytest.approx([0.001, 0.002, 0.003])


def test_OEMStateVector_standalone_setters(eop):
    """Test setting properties on a standalone OEMStateVector."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch=epoch, position=[1, 2, 3], velocity=[4, 5, 6])

    sv.position = [10.0, 20.0, 30.0]
    assert sv.position == pytest.approx([10.0, 20.0, 30.0])

    sv.velocity = [40.0, 50.0, 60.0]
    assert sv.velocity == pytest.approx([40.0, 50.0, 60.0])

    new_epoch = Epoch.from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv.epoch = new_epoch
    assert sv.epoch == new_epoch


def test_OEMSegment_standalone_construction(eop):
    """Test creating a standalone OEMSegment."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    assert seg.object_name == "SAT1"
    assert seg.object_id == "2024-001A"
    assert seg.center_name == "EARTH"
    assert seg.ref_frame == "GCRF"
    assert seg.time_system == "UTC"
    assert seg.start_time == start
    assert seg.stop_time == stop
    assert seg.num_states == 0
    assert seg.interpolation is None


def test_OEMSegment_standalone_add_state(eop):
    """Test adding states to a standalone segment."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    idx = seg.add_state(epoch=start, position=[7000e3, 0, 0], velocity=[0, 7500, 0])
    assert idx == 0
    assert seg.num_states == 1

    # Access states on standalone segment returns a list
    states = seg.states
    assert isinstance(states, list)
    assert len(states) == 1
    assert states[0].position == pytest.approx([7000e3, 0.0, 0.0])


# ─────────────────────────────────────────────
# append/extend tests
# ─────────────────────────────────────────────


def test_OEMSegments_append(eop):
    """Test appending a standalone segment to an OEM."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem = OEM(originator="TEST")
    oem.segments.append(seg)
    assert len(oem.segments) == 1
    assert oem.segments[0].object_name == "SAT1"


def test_OEMSegments_append_with_states(eop):
    """Test appending a segment with states — states carry over."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    seg.add_state(epoch=start, position=[7000e3, 0, 0], velocity=[0, 7500, 0])
    seg.add_state(epoch=stop, position=[6000e3, 3000e3, 0], velocity=[-2000, 6000, 0])

    oem = OEM(originator="TEST")
    oem.segments.append(seg)
    assert oem.segments[0].num_states == 2
    assert oem.segments[0].states[0].position[0] == pytest.approx(7000e3)
    assert oem.segments[0].states[1].position[0] == pytest.approx(6000e3)


def test_OEMSegments_extend(eop):
    """Test extending OEM with multiple segments."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg1 = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    seg2 = OEMSegment(
        object_name="SAT2",
        object_id="2024-002A",
        center_name="EARTH",
        ref_frame="J2000",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem = OEM(originator="TEST")
    oem.segments.extend([seg1, seg2])
    assert len(oem.segments) == 2
    assert oem.segments[0].object_name == "SAT1"
    assert oem.segments[1].object_name == "SAT2"


def test_OEMStates_append(eop):
    """Test appending a standalone state vector to a segment."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch=epoch, position=[7000e3, 0, 0], velocity=[0, 7500, 0])

    oem = OEM(originator="TEST")
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.add_segment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem.segments[0].states.append(sv)
    assert oem.segments[0].num_states == 1
    assert oem.segments[0].states[0].position[0] == pytest.approx(7000e3)


def test_OEMStates_extend(eop):
    """Test extending a segment's states with multiple state vectors."""
    epoch1 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    epoch2 = Epoch.from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv1 = OEMStateVector(epoch=epoch1, position=[7000e3, 0, 0], velocity=[0, 7500, 0])
    sv2 = OEMStateVector(
        epoch=epoch2, position=[6000e3, 3000e3, 0], velocity=[-2000, 6000, 0]
    )

    oem = OEM(originator="TEST")
    start = epoch1
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.add_segment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem.segments[0].states.extend([sv1, sv2])
    assert oem.segments[0].num_states == 2


def test_append_preserves_standalone(eop):
    """Test that appending copies data — original standalone object is unchanged."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="ORIGINAL",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    oem = OEM(originator="TEST")
    oem.segments.append(seg)

    # Modify the OEM copy
    oem.segments[0].object_name = "MODIFIED"

    # Original standalone object is unchanged
    assert seg.object_name == "ORIGINAL"


# ─────────────────────────────────────────────
# __delitem__ tests
# ─────────────────────────────────────────────


def test_OEMSegments_delitem(eop):
    """Test del oem.segments[0]."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    assert len(oem.segments) == 3

    del oem.segments[0]
    assert len(oem.segments) == 2


def test_OEMSegments_delitem_negative(eop):
    """Test del oem.segments[-1]."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    assert len(oem.segments) == 3

    del oem.segments[-1]
    assert len(oem.segments) == 2


def test_OEMStates_delitem(eop):
    """Test del seg.states[0]."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    assert oem.segments[0].num_states == 3

    del oem.segments[0].states[0]
    assert oem.segments[0].num_states == 2


# ─────────────────────────────────────────────
# Flexible input tests (numpy arrays, lists, slices)
# ─────────────────────────────────────────────


def test_OEMStateVector_constructor_accepts_numpy(eop):
    """OEMStateVector constructor accepts numpy arrays."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    pos = np.array([7000e3, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    sv = OEMStateVector(epoch, pos, vel)
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])


def test_OEMStateVector_constructor_accepts_numpy_slices(eop):
    """OEMStateVector constructor accepts numpy array slices."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    sv = OEMStateVector(epoch, state[:3], state[3:6])
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])


def test_OEMStateVector_setters_accept_numpy(eop):
    """OEMStateVector setters accept numpy arrays."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch, [7000e3, 0, 0], [0, 7500, 0])
    sv.position = np.array([8000e3, 0.0, 0.0])
    sv.velocity = np.array([0.0, 8000.0, 0.0])
    assert sv.position == pytest.approx([8000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 8000.0, 0.0])


def test_OEMStateVector_proxy_setters_accept_numpy(eop):
    """OEMStateVector setters accept numpy arrays in proxy mode."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    sv = oem.segments[0].states[0]
    sv.position = np.array([1.0, 2.0, 3.0])
    sv.velocity = np.array([4.0, 5.0, 6.0])
    assert oem.segments[0].states[0].position == pytest.approx([1.0, 2.0, 3.0])
    assert oem.segments[0].states[0].velocity == pytest.approx([4.0, 5.0, 6.0])


def test_OEMSegment_add_state_accepts_numpy(eop):
    """OEMSegment.add_state accepts numpy arrays."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    seg.add_state(
        epoch=epoch, position=np.array([7000e3, 0, 0]), velocity=np.array([0, 7500, 0])
    )
    assert seg.num_states == 1
    states = seg.states
    assert states[0].position == pytest.approx([7000e3, 0.0, 0.0])


# ─────────────────────────────────────────────
# .state property tests
# ─────────────────────────────────────────────


def test_OEMStateVector_state_getter(eop):
    """OEMStateVector.state returns 6-element numpy array."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    sv = oem.segments[0].states[0]
    state = sv.state
    assert isinstance(state, np.ndarray)
    assert len(state) == 6
    assert state[0] == pytest.approx(2789619.0, abs=1.0)
    assert state[3] == pytest.approx(4733.72, abs=0.01)


def test_OEMStateVector_state_setter_owned(eop):
    """OEMStateVector.state setter works on owned state vectors."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch, [0, 0, 0], [0, 0, 0])
    sv.state = [7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0]
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])


def test_OEMStateVector_state_setter_numpy(eop):
    """OEMStateVector.state setter accepts numpy array."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch, [0, 0, 0], [0, 0, 0])
    sv.state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    assert sv.position == pytest.approx([7000e3, 0.0, 0.0])
    assert sv.velocity == pytest.approx([0.0, 7500.0, 0.0])


def test_OEMStateVector_state_setter_proxy(eop):
    """OEMStateVector.state setter works in proxy mode."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    sv = oem.segments[0].states[0]
    sv.state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Verify via fresh proxy access
    assert oem.segments[0].states[0].position == pytest.approx([1.0, 2.0, 3.0])
    assert oem.segments[0].states[0].velocity == pytest.approx([4.0, 5.0, 6.0])


# ─────────────────────────────────────────────
# Trajectory import tests
# ─────────────────────────────────────────────


@pytest.fixture
def keplerian_trajectory(eop):
    """Create a short Keplerian-propagated trajectory for testing."""
    epoch = Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oe = np.array([brahe.R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 0.0])
    prop = brahe.KeplerianPropagator.from_keplerian(
        epoch, oe, brahe.AngleFormat.DEGREES, 60.0
    )
    # Propagate 5 steps of 60s
    for i in range(1, 6):
        prop.state(epoch + i * 60.0)
    return epoch, prop


def test_OEMSegment_add_trajectory(keplerian_trajectory):
    """Test bulk-adding trajectory states to a standalone segment.

    add_trajectory() should auto-convert from the trajectory's internal
    representation (Keplerian elements) to Cartesian states in the segment's
    declared ref_frame (EME2000).
    """
    epoch, prop = keplerian_trajectory
    traj = prop.trajectory
    stop_epoch = epoch + 5 * 60.0

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=stop_epoch,
    )
    seg.add_trajectory(traj)

    assert seg.num_states == len(traj)
    # Verify first state matches trajectory's EME2000 Cartesian output
    epc0, _ = traj.get(0)
    expected = prop.state_eme2000(epc0)
    states = seg.states
    assert states[0].epoch == epc0
    assert states[0].position == pytest.approx(list(expected[:3]), abs=1.0)
    assert states[0].velocity == pytest.approx(list(expected[3:6]), abs=0.001)
    # Sanity check: position magnitude should be ~R_EARTH + 500km
    pos_mag = np.linalg.norm(states[0].position)
    assert pos_mag == pytest.approx(brahe.R_EARTH + 500e3, rel=0.01)


def test_OEM_add_segment_with_trajectory(keplerian_trajectory):
    """Test add_segment with trajectory kwarg populates states automatically.

    States should be auto-converted to Cartesian in the declared ref_frame.
    """
    epoch, prop = keplerian_trajectory
    traj = prop.trajectory
    stop_epoch = epoch + 5 * 60.0

    oem = OEM(originator="TEST")
    seg_idx = oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=stop_epoch,
        trajectory=traj,
    )

    seg = oem.segments[seg_idx]
    assert seg.num_states == len(traj)
    # Verify last state matches trajectory's EME2000 output
    epc_last, _ = traj.get(len(traj) - 1)
    expected = prop.state_eme2000(epc_last)
    sv_last = seg.states[-1]
    assert sv_last.epoch == epc_last
    assert sv_last.position == pytest.approx(list(expected[:3]), abs=1.0)
    assert sv_last.velocity == pytest.approx(list(expected[3:6]), abs=0.001)


def test_OEMSegment_add_trajectory_proxy(keplerian_trajectory):
    """Test add_trajectory on a proxy segment (accessed via OEM).

    Proxy mode should also auto-convert trajectory states to the segment's
    declared ref_frame.
    """
    epoch, prop = keplerian_trajectory
    traj = prop.trajectory
    stop_epoch = epoch + 5 * 60.0

    oem = OEM(originator="TEST")
    seg_idx = oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=stop_epoch,
    )

    # Access via proxy and add trajectory
    oem.segments[seg_idx].add_trajectory(traj)

    assert oem.segments[seg_idx].num_states == len(traj)
    epc0, _ = traj.get(0)
    expected = prop.state_eme2000(epc0)
    sv0 = oem.segments[seg_idx].states[0]
    assert sv0.epoch == epc0
    assert sv0.position == pytest.approx(list(expected[:3]), abs=1.0)
    assert sv0.velocity == pytest.approx(list(expected[3:6]), abs=0.001)


def test_OEMSegment_add_trajectory_gcrf_frame(keplerian_trajectory):
    """Test add_trajectory with GCRF ref_frame produces GCRF states."""
    epoch, prop = keplerian_trajectory
    traj = prop.trajectory
    stop_epoch = epoch + 5 * 60.0

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=epoch,
        stop_time=stop_epoch,
    )
    seg.add_trajectory(traj)

    # States should match the trajectory's GCRF output
    epc0, _ = traj.get(0)
    expected = prop.state_gcrf(epc0)
    states = seg.states
    assert states[0].position == pytest.approx(list(expected[:3]), abs=1.0)
    assert states[0].velocity == pytest.approx(list(expected[3:6]), abs=0.001)


def test_OEMSegment_add_trajectory_itrf_frame(eop):
    """An ITRF trajectory written to an ITRF segment takes the same-frame
    shortcut, so the written states are the stored samples exactly."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6,
        brahe.CelestialFrame.ITRF,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    samples = [
        np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0]),
        np.array([brahe.R_EARTH + 501e3, 1.1e5, -2.1e5, 11.0, 7.5e3, -6.0]),
        np.array([brahe.R_EARTH + 502e3, 1.2e5, -2.2e5, 12.0, 7.4e3, -7.0]),
    ]
    for i, sample in enumerate(samples):
        traj.add(epoch + i * 60.0, sample)

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="ITRF2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 120.0,
    )
    seg.add_trajectory(traj)

    assert seg.num_states == len(samples)
    for written, sample in zip(seg.states, samples):
        np.testing.assert_array_equal(written.position, sample[:3])
        np.testing.assert_array_equal(written.velocity, sample[3:6])


def test_OEM_trajectory_round_trip(keplerian_trajectory):
    """Test full pipeline: trajectory → OEM → file → load → compare states."""
    epoch, prop = keplerian_trajectory
    traj = prop.trajectory
    stop_epoch = epoch + 5 * 60.0

    # Build OEM from trajectory
    oem = OEM(originator="ROUND_TRIP")
    oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=stop_epoch,
        trajectory=traj,
    )

    # Round-trip via serialization
    written = oem.to_string("KVN")
    oem2 = OEM.from_str(written)

    assert oem2.segments[0].num_states == len(traj)
    # Verify states survived the round-trip
    for i in range(len(traj)):
        epc_i, _ = traj.get(i)
        expected = prop.state_eme2000(epc_i)
        sv = oem2.segments[0].states[i]
        # KVN format has limited precision (~1m for position, ~0.001 m/s for velocity)
        assert sv.position == pytest.approx(list(expected[:3]), abs=1.0)
        assert sv.velocity == pytest.approx(list(expected[3:6]), abs=0.001)


def test_oem_xml_round_trip(eop):
    """Test OEM XML write then re-parse preserves data."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample3.xml")
    xml_str = oem.to_string("XML")
    oem2 = OEM.from_str(xml_str)
    assert oem2.originator == oem.originator
    assert len(oem2.segments) == len(oem.segments)
    assert oem2.segments[0].num_states == oem.segments[0].num_states
    # Position round-trip
    sv1 = oem.segments[0].states[0]
    sv2 = oem2.segments[0].states[0]
    assert sv2.position == pytest.approx(sv1.position, abs=1.0)
    assert sv2.velocity == pytest.approx(sv1.velocity, abs=1.0)


def _assert_oem_fields(oem1, oem2):
    """Assert all accessible OEM fields match."""
    assert oem2.format_version == oem1.format_version
    assert oem2.originator == oem1.originator
    assert oem2.classification == oem1.classification
    assert oem2.message_id == oem1.message_id
    assert len(oem2.segments) == len(oem1.segments)
    for seg1, seg2 in zip(oem1.segments, oem2.segments):
        assert seg2.object_name == seg1.object_name
        assert seg2.object_id == seg1.object_id
        assert seg2.center_name == seg1.center_name
        assert seg2.ref_frame == seg1.ref_frame
        assert seg2.time_system == seg1.time_system
        assert seg2.num_states == seg1.num_states
        for sv1, sv2 in zip(seg1.states, seg2.states):
            assert sv2.position == pytest.approx(sv1.position, abs=1.0)
            assert sv2.velocity == pytest.approx(sv1.velocity, abs=0.001)


def test_oem_kvn_full_round_trip(eop):
    """Full-field OEM KVN round-trip."""
    oem1 = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")
    kvn = oem1.to_string("KVN")
    oem2 = OEM.from_str(kvn)
    _assert_oem_fields(oem1, oem2)


def test_oem_xml_full_round_trip(eop):
    """Full-field OEM XML round-trip with acceleration + covariance."""
    oem1 = OEM.from_file("test_assets/ccsds/oem/OEMExample3.xml")
    xml = oem1.to_string("XML")
    oem2 = OEM.from_str(xml)
    _assert_oem_fields(oem1, oem2)


def test_oem_json_full_round_trip(eop):
    """Full-field OEM JSON round-trip."""
    oem1 = OEM.from_file("test_assets/ccsds/oem/OEMExample3.xml")
    json_str = oem1.to_string("JSON")
    oem2 = OEM.from_str(json_str)
    _assert_oem_fields(oem1, oem2)


# ─────────────────────────────────────────────
# Additional coverage: creation_date setter, standalone segment setters,
# OEM/segment repr, construction with options, message_id
# ─────────────────────────────────────────────


def test_oem_creation_date_setter(eop):
    """Test setting creation_date on OEM."""
    oem = OEM(originator="TEST")
    original = oem.creation_date
    assert isinstance(original, Epoch)

    new_date = Epoch.from_datetime(2025, 3, 15, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    oem.creation_date = new_date
    assert oem.creation_date == new_date


def test_oem_new_empty_verify_defaults(eop):
    """Create empty OEM and verify all default values."""
    oem = OEM(originator="MY_ORG")
    assert oem.originator == "MY_ORG"
    assert oem.format_version == pytest.approx(3.0)
    assert len(oem.segments) == 0
    assert isinstance(oem.creation_date, Epoch)
    assert oem.classification is None
    assert oem.message_id is None


def test_oem_standalone_segment_setters(eop):
    """Test setting properties on a standalone OEMSegment (owned mode)."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )

    # Mutate all metadata fields in owned mode
    seg.object_name = "NEW_SAT"
    assert seg.object_name == "NEW_SAT"

    seg.object_id = "2024-999A"
    assert seg.object_id == "2024-999A"

    seg.center_name = "MOON"
    assert seg.center_name == "MOON"

    seg.ref_frame = "J2000"
    assert seg.ref_frame == "J2000"

    seg.time_system = "TAI"
    assert seg.time_system == "TAI"

    new_start = Epoch.from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    new_stop = Epoch.from_datetime(2025, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    seg.start_time = new_start
    seg.stop_time = new_stop
    assert seg.start_time == new_start
    assert seg.stop_time == new_stop

    seg.interpolation = "HERMITE"
    assert seg.interpolation == "HERMITE"

    seg.interpolation_degree = 7
    assert seg.interpolation_degree == 7

    seg.comments = ["first comment", "second comment"]
    assert seg.comments == ["first comment", "second comment"]


def test_oem_standalone_segment_repr(eop):
    """Test OEMSegment repr in owned mode."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    r = repr(seg)
    assert "OEMSegment" in r
    assert "SAT1" in r
    assert "GCRF" in r


def test_oem_standalone_segment_remove_state(eop):
    """Test removing a state from a standalone segment."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    mid = Epoch.from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
    )
    seg.add_state(epoch=start, position=[7000e3, 0, 0], velocity=[0, 7500, 0])
    seg.add_state(epoch=mid, position=[6000e3, 3000e3, 0], velocity=[-2000, 6000, 0])
    assert seg.num_states == 2

    seg.remove_state(0)
    assert seg.num_states == 1
    # Remaining state should be the second one we added
    states = seg.states
    assert states[0].position[0] == pytest.approx(6000e3)


def test_oem_standalone_segment_with_interpolation(eop):
    """Test creating standalone OEMSegment with interpolation params."""
    start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    seg = OEMSegment(
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        start_time=start,
        stop_time=stop,
        interpolation="LAGRANGE",
        interpolation_degree=7,
    )
    assert seg.interpolation == "LAGRANGE"
    assert seg.interpolation_degree == 7


def test_oem_state_vector_repr(eop):
    """Test OEMStateVector repr."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    sv = OEMStateVector(epoch, [7000e3, 0, 0], [0, 7500, 0])
    r = repr(sv)
    assert "OEMStateVector" in r


def test_oem_segment_proxy_repr(eop):
    """Test OEMSegment repr in proxy mode."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    seg = oem.segments[0]
    r = repr(seg)
    assert "OEMSegment" in r
    assert "MARS GLOBAL SURVEYOR" in r


def test_oem_segments_repr(eop):
    """Test OEMSegments collection repr."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    r = repr(oem.segments)
    assert "OEMSegments" in r
    assert "3" in r


def test_oem_in_tod_loads_as_tod_trajectory(eop, clear_frame_registries):
    """Mirror of test_oem_in_tod_loads_as_tod_trajectory in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/test.oem")
    assert oem.segments[0].ref_frame == "TOD"

    traj = oem.to_trajectories()[0]
    assert traj.frame == brahe.CelestialFrame.TOD

    epc, x_tod = traj.get(0)
    assert x_tod[0] == pytest.approx(3156654.9969124, abs=1e-6)
    assert x_tod[1] == pytest.approx(-5911757.307689572, abs=1e-6)

    gcrf = traj.to_frame(brahe.CelestialFrame.GCRF)
    assert gcrf.frame == brahe.CelestialFrame.GCRF
    _, x_gcrf = gcrf.get(0)
    np.testing.assert_allclose(x_gcrf, brahe.state_tod_to_gcrf(epc, x_tod), atol=1e-9)
    assert np.linalg.norm(x_gcrf[:3] - x_tod[:3]) > 1.0e3

    oem.register_for("TOD_SAT")
    assert "TOD_SAT" in brahe.registered_objects()


def test_oem_in_teme_loads_as_teme_trajectory(eop, clear_frame_registries):
    """Mirror of test_oem_in_teme_loads_as_teme_trajectory in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/test.oem")
    oem.segments[0].ref_frame = "TEME"

    traj = oem.segment_to_trajectory(0)
    assert traj.frame == brahe.CelestialFrame.TEME

    epc, x_teme = traj.get(0)
    x_gcrf = traj.to_frame(brahe.CelestialFrame.GCRF).get(0)[1]
    np.testing.assert_allclose(x_gcrf, brahe.state_teme_to_gcrf(epc, x_teme), atol=1e-9)

    oem.register_for("TEME_SAT")
    assert "TEME_SAT" in brahe.registered_objects()


def test_OEMSegment_add_trajectory_teme_from_teme(eop):
    """A TEME trajectory written to a TEME segment stores its samples unchanged."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6,
        brahe.CelestialFrame.TEME,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    samples = [
        np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0]),
        np.array([brahe.R_EARTH + 501e3, 1.1e5, -2.1e5, 11.0, 7.5e3, -6.0]),
    ]
    for i, sample in enumerate(samples):
        traj.add(epoch + i * 60.0, sample)

    oem = OEM(originator="TEST")
    seg_idx = oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TEME",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    oem.segments[seg_idx].add_trajectory(traj)

    seg = oem.segments[seg_idx]
    assert seg.num_states == len(samples)
    for written, sample in zip(seg.states, samples):
        np.testing.assert_array_equal(written.position, sample[:3])
        np.testing.assert_array_equal(written.velocity, sample[3:6])

    traj_back = oem.segment_to_trajectory(seg_idx)
    assert traj_back.frame == brahe.CelestialFrame.TEME


def test_OEMSegment_add_trajectory_tod_from_tod(eop):
    """A TOD trajectory written to a TOD segment stores its samples unchanged."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6,
        brahe.CelestialFrame.TOD,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    samples = [
        np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0]),
        np.array([brahe.R_EARTH + 501e3, 1.1e5, -2.1e5, 11.0, 7.5e3, -6.0]),
    ]
    for i, sample in enumerate(samples):
        traj.add(epoch + i * 60.0, sample)

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TOD",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    seg.add_trajectory(traj)

    assert seg.num_states == len(samples)
    for written, sample in zip(seg.states, samples):
        np.testing.assert_array_equal(written.position, sample[:3])
        np.testing.assert_array_equal(written.velocity, sample[3:6])


def test_OEMSegment_add_trajectory_tod_from_gcrf(eop):
    """A GCRF trajectory written to a TOD segment is rotated into TOD."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6,
        brahe.CelestialFrame.GCRF,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    samples = [
        np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0]),
        np.array([brahe.R_EARTH + 501e3, 1.1e5, -2.1e5, 11.0, 7.5e3, -6.0]),
    ]
    for i, sample in enumerate(samples):
        traj.add(epoch + i * 60.0, sample)

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TOD",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    seg.add_trajectory(traj)

    assert seg.num_states == len(samples)
    for i, (written, sample) in enumerate(zip(seg.states, samples)):
        expected = brahe.state_gcrf_to_tod(epoch + i * 60.0, sample)
        np.testing.assert_allclose(written.position, expected[:3], atol=1e-6)
        np.testing.assert_allclose(written.velocity, expected[3:6], atol=1e-9)
        assert np.linalg.norm(np.array(written.position) - sample[:3]) > 1.0e3


def test_OEMSegment_add_trajectory_tod_of_epoch_from_gcrf(eop):
    """A GCRF trajectory written to a TOD segment with REF_FRAME_EPOCH is rotated at the frozen epoch."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    ref_epoch = Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6, brahe.CelestialFrame.GCRF, brahe.OrbitRepresentation.CARTESIAN, None
    )
    sample = np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0])
    traj.add(epoch, sample)
    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TOD",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
        ref_frame_epoch=ref_epoch,
    )
    assert seg.ref_frame_epoch == ref_epoch
    seg.add_trajectory(traj)
    expected = brahe.state_gcrf_to_tod(ref_epoch, sample)
    np.testing.assert_allclose(seg.states[0].position, expected[:3], atol=1e-9, rtol=0)
    np.testing.assert_allclose(seg.states[0].velocity, expected[3:], atol=1e-12, rtol=0)
    seg.ref_frame_epoch = None
    assert seg.ref_frame_epoch is None


def test_OEMSegment_add_trajectory_tod_of_epoch_from_gcrf_proxy(eop):
    """Proxy-mode mirror of test_OEMSegment_add_trajectory_tod_of_epoch_from_gcrf."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    ref_epoch = Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    traj = brahe.OrbitTrajectory(
        6, brahe.CelestialFrame.GCRF, brahe.OrbitRepresentation.CARTESIAN, None
    )
    sample = np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0])
    traj.add(epoch, sample)

    oem = OEM(originator="TEST")
    seg_idx = oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TOD",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    oem.segments[seg_idx].ref_frame_epoch = ref_epoch
    assert oem.segments[seg_idx].ref_frame_epoch == ref_epoch

    oem.segments[seg_idx].add_trajectory(traj)

    seg = oem.segments[seg_idx]
    expected = brahe.state_gcrf_to_tod(ref_epoch, sample)
    np.testing.assert_allclose(seg.states[0].position, expected[:3], atol=1e-9, rtol=0)
    np.testing.assert_allclose(seg.states[0].velocity, expected[3:], atol=1e-12, rtol=0)

    oem.segments[seg_idx].ref_frame_epoch = None
    assert oem.segments[seg_idx].ref_frame_epoch is None


def _tdr_trajectory(epoch):
    """Build a two-sample GCRF trajectory for unsupported-frame tests.

    Args:
        epoch (Epoch): Epoch of the first sample

    Returns:
        OrbitTrajectory: Trajectory holding two Cartesian GCRF samples
    """
    traj = brahe.OrbitTrajectory(
        6,
        brahe.CelestialFrame.GCRF,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    for i in range(2):
        traj.add(
            epoch + i * 60.0,
            np.array([brahe.R_EARTH + 500e3, 1.0e5, -2.0e5, 10.0, 7.6e3, -5.0]),
        )
    return traj


def test_OEMSegment_add_trajectory_unsupported_frame_owned(eop):
    """A standalone segment declared in TDR raises BraheError."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = _tdr_trajectory(epoch)

    seg = OEMSegment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TDR",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )

    with pytest.raises(brahe.BraheError, match="TDR"):
        seg.add_trajectory(traj)


def test_OEMSegment_add_trajectory_unsupported_frame_proxy(eop):
    """A segment attached to an OEM and declared in TDR raises BraheError."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    traj = _tdr_trajectory(epoch)

    oem = OEM(originator="TEST")
    seg_idx = oem.add_segment(
        object_name="SAT",
        object_id="2024-100A",
        center_name="EARTH",
        ref_frame="TDR",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )

    with pytest.raises(brahe.BraheError, match="TDR"):
        oem.segments[seg_idx].add_trajectory(traj)


def test_oem_to_trajectories_rejects_unmapped_frame(eop):
    """Mirror of test_oem_segment_to_dorbit_trajectory_rejects_unmapped_frame in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/test.oem")
    oem.segments[0].ref_frame = "TDR"

    with pytest.raises(brahe.BraheError, match="TDR"):
        oem.to_trajectories()


def test_oem_register_for_rejects_unmapped_frame(eop, clear_frame_registries):
    """Mirror of test_oem_register_for_rejects_unmapped_frame in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/test.oem")
    oem.segments[0].ref_frame = "TDR"

    with pytest.raises(brahe.BraheError, match="TDR"):
        oem.register_for("TDR_SAT")


def test_oem_in_tod_with_frame_epoch_loads_as_tod_of_epoch_trajectory(
    eop, clear_frame_registries
):
    """Mirror of test_oem_in_tod_with_frame_epoch_loads_as_tod_of_epoch_trajectory in Rust."""
    raw = OEM.from_file("test_assets/ccsds/oem/test.oem")
    oem = OEM.from_file("test_assets/ccsds/oem/test_tod_epoch.oem")
    ref_epoch = Epoch.from_string("2019-09-08T00:00:00.0Z")

    _, x_raw = raw.to_trajectories()[0].get(0)

    traj = oem.to_trajectories()[0]
    assert traj.frame == brahe.CelestialFrame.tod_of_epoch(ref_epoch)
    # Samples are stored unchanged in the declared frame.
    epc, x_stored = traj.get(0)
    np.testing.assert_array_equal(x_stored, x_raw)

    x_gcrf = traj.to_frame(brahe.CelestialFrame.GCRF).get(0)[1]
    expected = brahe.state_tod_to_gcrf(ref_epoch, x_raw)
    np.testing.assert_allclose(x_gcrf, expected, atol=1e-9)

    of_date = brahe.state_tod_to_gcrf(epc, x_raw)
    assert np.linalg.norm(expected[:3] - of_date[:3]) > 1.0

    oem.register_for("TOD_EPOCH_SAT")
    assert "TOD_EPOCH_SAT" in brahe.registered_objects()


def test_oem_in_teme_with_frame_epoch_loads_as_teme_of_epoch_trajectory(
    eop, clear_frame_registries
):
    """Mirror of test_oem_in_teme_with_frame_epoch_loads_as_teme_of_epoch_trajectory in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/test_tod_epoch.oem")
    oem.segments[0].ref_frame = "TEME"
    ref_epoch = Epoch.from_string("2019-09-08T00:00:00.0Z")

    traj = oem.to_trajectories()[0]
    assert traj.frame == brahe.CelestialFrame.teme_of_epoch(ref_epoch)
    _, x_raw = traj.get(0)

    x_gcrf = traj.to_frame(brahe.CelestialFrame.GCRF).get(0)[1]
    expected = brahe.state_teme_to_gcrf(ref_epoch, x_raw)
    np.testing.assert_allclose(x_gcrf, expected, atol=1e-9)


def test_oem_to_trajectory_example4(eop):
    """Mirror of test_oem_to_trajectory_example4 in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")

    traj = oem.to_trajectories()[0]
    assert len(traj) == 3
    assert oem.segments[0].object_name == "MARS GLOBAL SURVEYOR"
    assert traj.frame == brahe.CelestialFrame.Centered(
        brahe.NAIFId.MARS, brahe.FrameAxes.EME2000
    )

    _, state = traj.get(0)
    assert state[0] == pytest.approx(2789.619 * 1000.0, abs=1.0)
    assert state[3] == pytest.approx(4.73372 * 1000.0, abs=1.0)


def test_oem_to_trajectories_multi_segment(eop):
    """Mirror of test_oem_to_trajectories_multi_segment in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    trajs = oem.to_trajectories()
    assert len(trajs) == 3
    for traj in trajs:
        assert traj.frame == brahe.CelestialFrame.Centered(
            brahe.NAIFId.MARS_BARYCENTER, brahe.FrameAxes.EME2000
        )


def test_odm_celestial_frame_center_names(eop):
    """Mirror of test_odm_celestial_frame_center_names in Rust.

    `odm_celestial_frame` has no Python binding, so the joint
    CENTER_NAME/REF_FRAME resolution is exercised through `OEM.from_file`.
    """
    mars = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    assert mars.segments[0].center_name == "MARS"
    assert mars.to_trajectories()[0].frame == brahe.CelestialFrame.Centered(
        499, brahe.FrameAxes.EME2000
    )

    mars_barycenter = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    assert mars_barycenter.segments[0].center_name == "MARS BARYCENTER"
    assert mars_barycenter.to_trajectories()[0].frame == brahe.CelestialFrame.Centered(
        4, brahe.FrameAxes.EME2000
    )

    earth = OEM.from_file("test_assets/ccsds/oem/test.oem")
    assert earth.segments[0].center_name == "EARTH"
    assert earth.to_trajectories()[0].frame == brahe.CelestialFrame.TOD

    # Center names are matched case-insensitively.
    lowercase = OEM.from_file("test_assets/ccsds/oem/test.oem")
    lowercase.segments[0].center_name = "earth"
    assert lowercase.to_trajectories()[0].frame == brahe.CelestialFrame.TOD

    # A numeric center name resolves to the same NAIF ID as its body name.
    numeric = OEM.from_file("test_assets/ccsds/oem/OEMExample4.txt")
    numeric.segments[0].center_name = "499"
    assert numeric.to_trajectories()[0].frame == brahe.CelestialFrame.Centered(
        499, brahe.FrameAxes.EME2000
    )

    unknown = OEM.from_file("test_assets/ccsds/oem/test.oem")
    unknown.segments[0].center_name = "PLANET X"
    with pytest.raises(brahe.BraheError, match="PLANET X"):
        unknown.to_trajectories()


def test_odm_celestial_frame_frozen_tod_mars_center(eop):
    """Mirror of test_odm_celestial_frame_frozen_tod_mars_center in Rust.

    A TOD or TEME segment with a REF_FRAME_EPOCH loads in the corresponding
    of-epoch frame; without one the axes stay of-date. A REF_FRAME_EPOCH on
    any other token is ignored.
    """
    ref_epoch = Epoch.from_string("2019-09-08T00:00:00.0Z")

    frozen = OEM.from_file("test_assets/ccsds/oem/test_tod_epoch.oem")
    frozen.segments[0].center_name = "MARS"
    assert frozen.to_trajectories()[0].frame == brahe.CelestialFrame.Centered(
        499, brahe.FrameAxes.TODofEpoch(ref_epoch)
    )

    of_date = OEM.from_file("test_assets/ccsds/oem/test.oem")
    of_date.segments[0].center_name = "MARS"
    assert of_date.to_trajectories()[0].frame == brahe.CelestialFrame.Centered(
        499, brahe.FrameAxes.TOD
    )

    teme_frozen = OEM.from_file("test_assets/ccsds/oem/test_tod_epoch.oem")
    teme_frozen.segments[0].ref_frame = "TEME"
    assert teme_frozen.to_trajectories()[0].frame == brahe.CelestialFrame.teme_of_epoch(
        ref_epoch
    )

    gcrf_frozen = OEM.from_file("test_assets/ccsds/oem/test_tod_epoch.oem")
    gcrf_frozen.segments[0].ref_frame = "GCRF"
    assert gcrf_frozen.to_trajectories()[0].frame == brahe.CelestialFrame.GCRF


def test_OEMSegment_add_trajectory_uses_the_segment_center(eop):
    """A trajectory written to a Mars-centered segment keeps the segment's
    center as well as its REF_FRAME axes."""
    epoch = Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, brahe.UTC)
    mars_eme2000 = brahe.CelestialFrame.Centered(
        brahe.NAIFId.MARS, brahe.FrameAxes.EME2000
    )
    traj = brahe.OrbitTrajectory(
        6,
        mars_eme2000,
        brahe.OrbitRepresentation.CARTESIAN,
        None,
    )
    samples = [
        np.array([brahe.R_MARS + 400e3, 1.0e5, -2.0e5, 10.0, 3.3e3, -5.0]),
        np.array([brahe.R_MARS + 401e3, 1.1e5, -2.1e5, 11.0, 3.2e3, -6.0]),
    ]
    for i, sample in enumerate(samples):
        traj.add(epoch + i * 60.0, sample)

    seg = OEMSegment(
        object_name="MARS SAT",
        object_id="2024-100A",
        center_name="MARS",
        ref_frame="EME2000",
        time_system="UTC",
        start_time=epoch,
        stop_time=epoch + 60.0,
    )
    seg.add_trajectory(traj)

    assert seg.num_states == len(samples)
    for written, sample in zip(seg.states, samples):
        np.testing.assert_array_equal(written.position, sample[:3])
        np.testing.assert_array_equal(written.velocity, sample[3:6])
