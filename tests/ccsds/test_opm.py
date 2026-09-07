"""Tests for CCSDS OPM parsing, mutation, and construction — parity with Rust tests."""

import numpy as np
import pytest

import brahe
from brahe import Epoch
from brahe.ccsds import OPM, OPMManeuver


def test_opm_parse_example1(eop):
    """Mirror of test_parse_opm_example1 in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    assert opm.format_version == pytest.approx(3.0, abs=1e-10)
    assert opm.object_name == "GODZILLA 5"
    assert opm.object_id == "1998-999A"
    assert opm.ref_frame == "ITRF2000"
    assert opm.time_system == "UTC"

    # State vector (km -> m)
    pos = opm.position
    assert pos[0] == pytest.approx(6503514.0, abs=1.0)
    assert pos[1] == pytest.approx(1239647.0, abs=1.0)
    assert pos[2] == pytest.approx(-717490.0, abs=1.0)

    vel = opm.velocity
    assert vel[0] == pytest.approx(-873.160, abs=0.001)
    assert vel[1] == pytest.approx(8740.420, abs=0.001)
    assert vel[2] == pytest.approx(-4191.076, abs=0.001)

    # Epoch is a brahe Epoch object
    assert isinstance(opm.epoch, Epoch)

    # Spacecraft parameters
    assert opm.mass == pytest.approx(3000.0, abs=1e-3)

    # No Keplerian, no maneuvers, no covariance
    assert not opm.has_keplerian_elements
    assert len(opm.maneuvers) == 0


def test_opm_parse_example2_with_keplerian_and_maneuvers(eop):
    """Mirror of test_parse_opm_example2_with_keplerian_and_maneuvers in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")

    assert opm.object_name == "EUTELSAT W4"
    assert opm.ref_frame == "TOD"

    # State vector
    pos = opm.position
    assert pos[0] == pytest.approx(6655994.2, abs=1.0)

    # Keplerian elements
    assert opm.has_keplerian_elements
    assert opm.semi_major_axis == pytest.approx(41399512.3, abs=1.0)

    # 2 maneuvers via .maneuvers property — typed access
    assert len(opm.maneuvers) == 2

    m1 = opm.maneuvers[0]
    assert m1.duration == pytest.approx(132.60, abs=0.01)
    assert m1.delta_mass == pytest.approx(-18.418, abs=0.001)
    assert m1.ref_frame == "J2000"
    assert m1.dv[0] == pytest.approx(-23.257, abs=0.001)
    assert isinstance(m1.epoch_ignition, Epoch)

    m2 = opm.maneuvers[1]
    assert m2.ref_frame == "RTN"


def test_opm_parse_example5_three_maneuvers(eop):
    """Mirror of test_parse_opm_example5_with_three_maneuvers in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample5.txt")

    assert opm.ref_frame == "GCRF"
    assert opm.time_system == "GPS"
    assert len(opm.maneuvers) == 3


def test_opm_to_dict(eop):
    """Test to_dict() serialization."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    d = opm.to_dict()

    assert d["header"]["originator"] == "GSOC"
    assert d["metadata"]["object_name"] == "EUTELSAT W4"
    assert d["metadata"]["ref_frame"] == "TOD"

    # State vector
    assert d["state_vector"]["position"][0] == pytest.approx(6655994.2, abs=1.0)

    # Keplerian elements
    assert "keplerian_elements" in d
    assert d["keplerian_elements"]["semi_major_axis"] == pytest.approx(
        41399512.3, abs=1.0
    )
    assert d["keplerian_elements"]["eccentricity"] == pytest.approx(
        0.020842611, abs=1e-9
    )
    assert d["keplerian_elements"]["true_anomaly"] == pytest.approx(41.922339, abs=1e-6)

    # Maneuvers
    assert "maneuvers" in d
    assert len(d["maneuvers"]) == 2
    assert d["maneuvers"][0]["ref_frame"] == "J2000"

    # Spacecraft parameters
    assert "spacecraft_parameters" in d
    assert d["spacecraft_parameters"]["mass"] == pytest.approx(1913.0, abs=0.001)


def test_opm_to_dict_with_user_defined(eop):
    """Test to_dict() with user-defined parameters."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample4.txt")
    d = opm.to_dict()

    assert "user_defined" in d
    assert d["user_defined"]["OBJ1_TIME_LASTOB_START"] == "2020-01-29T13:30:00"


def test_opm_repr(eop):
    """Test OPM repr."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    r = repr(opm)
    assert "GODZILLA 5" in r
    assert "ITRF2000" in r


def test_opm_maneuvers_iteration(eop):
    """Test iterating over maneuvers — typed access."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")

    frames = [m.ref_frame for m in opm.maneuvers]
    assert frames == ["J2000", "RTN"]


def test_opm_maneuvers_negative_indexing(eop):
    """Test negative indexing on maneuvers."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    last = opm.maneuvers[-1]
    assert last.ref_frame == "RTN"


def test_opm_maneuvers_index_out_of_range(eop):
    """Test that out-of-range maneuver index raises IndexError."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    with pytest.raises(IndexError):
        opm.maneuvers[100]


# ─────────────────────────────────────────────
# Mutation tests
# ─────────────────────────────────────────────


def test_opm_header_setters(eop):
    """Test setting header properties on OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    opm.originator = "NEW_ORG"
    assert opm.originator == "NEW_ORG"

    opm.format_version = 2.0
    assert opm.format_version == pytest.approx(2.0)


def test_opm_metadata_setters(eop):
    """Test setting metadata on OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    opm.object_name = "NEW_SAT"
    assert opm.object_name == "NEW_SAT"

    opm.object_id = "2024-001A"
    assert opm.object_id == "2024-001A"

    opm.center_name = "MOON"
    assert opm.center_name == "MOON"

    opm.ref_frame = "GCRF"
    assert opm.ref_frame == "GCRF"

    opm.time_system = "TAI"
    assert opm.time_system == "TAI"


def test_opm_state_vector_setters(eop):
    """Test setting state vector fields on OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    opm.position = [1.0, 2.0, 3.0]
    assert opm.position == pytest.approx([1.0, 2.0, 3.0])

    opm.velocity = [4.0, 5.0, 6.0]
    assert opm.velocity == pytest.approx([4.0, 5.0, 6.0])

    new_epoch = Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.epoch = new_epoch
    assert opm.epoch == new_epoch


def test_opm_maneuver_setters(eop):
    """Test setting maneuver fields via proxy — mutations reflect back."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")

    opm.maneuvers[0].ref_frame = "RTN"
    assert opm.maneuvers[0].ref_frame == "RTN"

    opm.maneuvers[0].duration = 200.0
    assert opm.maneuvers[0].duration == pytest.approx(200.0)

    opm.maneuvers[0].delta_mass = -10.0
    assert opm.maneuvers[0].delta_mass == pytest.approx(-10.0)

    opm.maneuvers[0].dv = [1.0, 2.0, 3.0]
    assert opm.maneuvers[0].dv == pytest.approx([1.0, 2.0, 3.0])

    new_epoch = Epoch.from_datetime(2024, 3, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.maneuvers[0].epoch_ignition = new_epoch
    assert opm.maneuvers[0].epoch_ignition == new_epoch


def test_opm_add_maneuver(eop):
    """Test adding a maneuver to an OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    assert len(opm.maneuvers) == 0

    epoch = Epoch.from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    idx = opm.add_maneuver(
        epoch_ignition=epoch,
        duration=120.0,
        ref_frame="RTN",
        dv=[10.0, 0.0, 0.0],
    )
    assert idx == 0
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].duration == pytest.approx(120.0)
    assert opm.maneuvers[0].ref_frame == "RTN"
    assert opm.maneuvers[0].dv == pytest.approx([10.0, 0.0, 0.0])
    assert opm.maneuvers[0].delta_mass is None


def test_opm_add_maneuver_with_delta_mass(eop):
    """Test adding a maneuver with delta_mass."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    epoch = Epoch.from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.add_maneuver(
        epoch_ignition=epoch,
        duration=120.0,
        ref_frame="J2000",
        dv=[5.0, 0.0, 0.0],
        delta_mass=-15.0,
    )
    assert opm.maneuvers[0].delta_mass == pytest.approx(-15.0)


def test_opm_remove_maneuver(eop):
    """Test removing a maneuver."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    assert len(opm.maneuvers) == 2

    opm.remove_maneuver(0)
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].ref_frame == "RTN"


def test_opm_position_validation(eop):
    """Test validation of position vector dimensions."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    with pytest.raises(ValueError, match="length 3"):
        opm.position = [1.0, 2.0]  # only 2 elements


# ─────────────────────────────────────────────
# Standalone construction and list API tests
# ─────────────────────────────────────────────


def test_OPMManeuver_standalone_construction(eop):
    """Test creating a standalone OPMManeuver."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m = OPMManeuver(
        epoch_ignition=epoch,
        duration=120.0,
        ref_frame="RTN",
        dv=[10.0, 0.0, 0.0],
    )
    assert m.duration == pytest.approx(120.0)
    assert m.ref_frame == "RTN"
    assert m.dv == pytest.approx([10.0, 0.0, 0.0])
    assert m.delta_mass is None
    assert m.epoch_ignition == epoch

    # With delta_mass
    m2 = OPMManeuver(
        epoch_ignition=epoch,
        duration=60.0,
        ref_frame="J2000",
        dv=[5.0, 0.0, 0.0],
        delta_mass=-15.0,
    )
    assert m2.delta_mass == pytest.approx(-15.0)


def test_OPMManeuvers_append(eop):
    """Test appending a standalone maneuver to an OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    assert len(opm.maneuvers) == 0

    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m = OPMManeuver(
        epoch_ignition=epoch,
        duration=120.0,
        ref_frame="RTN",
        dv=[10.0, 0.0, 0.0],
    )

    opm.maneuvers.append(m)
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].ref_frame == "RTN"
    assert opm.maneuvers[0].duration == pytest.approx(120.0)


def test_OPMManeuvers_extend(eop):
    """Test extending maneuvers with a list."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)

    m1 = OPMManeuver(epoch_ignition=epoch, duration=60.0, ref_frame="RTN", dv=[5, 0, 0])
    m2 = OPMManeuver(
        epoch_ignition=epoch, duration=120.0, ref_frame="J2000", dv=[0, 10, 0]
    )

    opm.maneuvers.extend([m1, m2])
    assert len(opm.maneuvers) == 2
    assert opm.maneuvers[0].ref_frame == "RTN"
    assert opm.maneuvers[1].ref_frame == "J2000"


def test_OPMManeuvers_delitem(eop):
    """Test del opm.maneuvers[0]."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    assert len(opm.maneuvers) == 2

    del opm.maneuvers[0]
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].ref_frame == "RTN"


# ─────────────────────────────────────────────
# Flexible input tests (numpy arrays, lists, slices)
# ─────────────────────────────────────────────


def test_opm_set_position_velocity_numpy(eop):
    """OPM position/velocity setters accept numpy arrays."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    opm.position = np.array([1.0, 2.0, 3.0])
    opm.velocity = np.array([4.0, 5.0, 6.0])
    assert opm.position == pytest.approx([1.0, 2.0, 3.0])
    assert opm.velocity == pytest.approx([4.0, 5.0, 6.0])


def test_opm_add_maneuver_numpy_dv(eop):
    """OPM.add_maneuver accepts numpy dv."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.add_maneuver(epoch, 60.0, "J2000", np.array([10.0, 0.0, 0.0]))
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].dv == pytest.approx([10.0, 0.0, 0.0])


def test_OPMManeuver_constructor_accepts_numpy(eop):
    """OPMManeuver constructor accepts numpy dv."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    man = OPMManeuver(epoch, 60.0, "J2000", np.array([10.0, 20.0, 30.0]))
    assert man.dv == pytest.approx([10.0, 20.0, 30.0])


def test_OPMManeuver_set_dv_numpy(eop):
    """OPMManeuver.dv setter accepts numpy array."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    man = OPMManeuver(epoch, 60.0, "J2000", [0, 0, 0])
    man.dv = np.array([1.0, 2.0, 3.0])
    assert man.dv == pytest.approx([1.0, 2.0, 3.0])


# ─────────────────────────────────────────────
# .state property tests
# ─────────────────────────────────────────────


def test_opm_state_getter(eop):
    """OPM.state returns 6-element numpy array."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    state = opm.state
    assert isinstance(state, np.ndarray)
    assert len(state) == 6
    assert state[0] == pytest.approx(6503514.0, abs=1.0)
    assert state[3] == pytest.approx(-873.160, abs=0.001)


def test_opm_state_setter_list(eop):
    """OPM.state setter accepts list."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    opm.state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert opm.position == pytest.approx([1.0, 2.0, 3.0])
    assert opm.velocity == pytest.approx([4.0, 5.0, 6.0])


def test_opm_state_setter_numpy(eop):
    """OPM.state setter accepts numpy array."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    opm.state = np.array([7000e3, 0, 0, 0, 7500, 0])
    assert opm.position == pytest.approx([7000e3, 0.0, 0.0])
    assert opm.velocity == pytest.approx([0.0, 7500.0, 0.0])


def test_opm_json_round_trip(eop):
    """OPM JSON round-trip: from_file -> to_string(JSON) -> from_str -> compare."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    json_str = opm.to_string("JSON")
    opm2 = OPM.from_str(json_str)
    assert opm2.object_name == opm.object_name
    assert opm2.object_id == opm.object_id
    assert opm2.position == pytest.approx(opm.position, abs=1.0)
    assert opm2.velocity == pytest.approx(opm.velocity, abs=0.001)


def test_opm_json_uppercase_keys(eop):
    """OPM to_json_string with uppercase keys."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    json_str = opm.to_json_string(uppercase_keys=True)
    assert '"OBJECT_NAME"' in json_str
    assert '"header"' in json_str  # container keys always lowercase


def test_opm_kvn_round_trip(eop):
    """Test OPM KVN write then re-parse preserves data."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    kvn_str = opm.to_string("KVN")
    opm2 = OPM.from_str(kvn_str)
    assert opm2.object_name == opm.object_name
    assert opm2.object_id == opm.object_id
    assert opm2.position == pytest.approx(opm.position, abs=1.0)
    assert opm2.velocity == pytest.approx(opm.velocity, abs=0.001)


def test_opm_xml_round_trip(eop):
    """Test OPM XML write then re-parse preserves data."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample3.xml")
    xml_str = opm.to_string("XML")
    opm2 = OPM.from_str(xml_str)
    assert opm2.object_name == opm.object_name
    assert opm2.object_id == opm.object_id
    assert opm2.position == pytest.approx(opm.position, abs=1.0)
    assert opm2.velocity == pytest.approx(opm.velocity, abs=0.001)


def test_opm_xml_parse_example3(eop):
    """Test parsing OPM XML Example 3 (OSPREY 5)."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample3.xml")
    assert opm.object_name == "OSPREY 5"
    assert opm.object_id == "1998-999A"
    assert opm.position[0] == pytest.approx(6503514.0, abs=1.0)


def _assert_opm_fields(opm1, opm2):
    """Assert all accessible OPM fields match."""
    # Header + metadata
    assert opm2.format_version == opm1.format_version
    assert opm2.originator == opm1.originator
    assert opm2.object_name == opm1.object_name
    assert opm2.object_id == opm1.object_id
    assert opm2.center_name == opm1.center_name
    assert opm2.ref_frame == opm1.ref_frame
    assert opm2.time_system == opm1.time_system
    # State vector
    assert opm2.position == pytest.approx(opm1.position, abs=1.0)
    assert opm2.velocity == pytest.approx(opm1.velocity, abs=0.001)
    # Keplerian elements
    assert opm2.has_keplerian_elements == opm1.has_keplerian_elements
    if opm1.has_keplerian_elements:
        assert opm2.semi_major_axis == pytest.approx(opm1.semi_major_axis, abs=1.0)
        assert opm2.eccentricity == pytest.approx(opm1.eccentricity, abs=1e-9)
        assert opm2.inclination == pytest.approx(opm1.inclination, abs=1e-6)
        assert opm2.ra_of_asc_node == pytest.approx(opm1.ra_of_asc_node, abs=1e-6)
        assert opm2.arg_of_pericenter == pytest.approx(opm1.arg_of_pericenter, abs=1e-6)
    # Spacecraft parameters
    if opm1.mass is not None:
        assert opm2.mass == pytest.approx(opm1.mass, abs=0.01)
    if opm1.solar_rad_area is not None:
        assert opm2.solar_rad_area == pytest.approx(opm1.solar_rad_area, abs=0.01)
    if opm1.drag_coeff is not None:
        assert opm2.drag_coeff == pytest.approx(opm1.drag_coeff, abs=0.01)
    # Maneuvers
    assert len(opm2.maneuvers) == len(opm1.maneuvers)
    for m1, m2 in zip(opm1.maneuvers, opm2.maneuvers):
        assert m2.duration == pytest.approx(m1.duration, abs=0.01)
        assert m2.dv == pytest.approx(m1.dv, abs=0.01)


def test_opm_kvn_full_round_trip(eop):
    """Full-field OPM KVN round-trip with covariance + Keplerian + user_defined."""
    opm1 = OPM.from_file("test_assets/ccsds/opm/OPMExample4.txt")
    kvn = opm1.to_string("KVN")
    opm2 = OPM.from_str(kvn)
    _assert_opm_fields(opm1, opm2)


def test_opm_xml_full_round_trip(eop):
    """Full-field OPM XML round-trip."""
    opm1 = OPM.from_file("test_assets/ccsds/opm/OPMExample4.txt")
    xml = opm1.to_string("XML")
    opm2 = OPM.from_str(xml)
    _assert_opm_fields(opm1, opm2)


def test_opm_json_full_round_trip(eop):
    """Full-field OPM JSON round-trip."""
    opm1 = OPM.from_file("test_assets/ccsds/opm/OPMExample4.txt")
    json_str = opm1.to_string("JSON")
    opm2 = OPM.from_str(json_str)
    _assert_opm_fields(opm1, opm2)


# ─────────────────────────────────────────────
# Programmatic constructor tests
# ─────────────────────────────────────────────


def test_opm_constructor(eop):
    """Test creating an OPM programmatically with the constructor."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        originator="TEST_ORG",
        object_name="SAT1",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="GCRF",
        time_system="UTC",
        epoch=epoch,
        position=np.array([7000e3, 0.0, 0.0]),
        velocity=np.array([0.0, 7500.0, 0.0]),
    )

    assert opm.originator == "TEST_ORG"
    assert opm.object_name == "SAT1"
    assert opm.object_id == "2024-001A"
    assert opm.center_name == "EARTH"
    assert opm.ref_frame == "GCRF"
    assert opm.time_system == "UTC"
    assert opm.format_version == pytest.approx(3.0)
    assert opm.position == pytest.approx([7000e3, 0.0, 0.0])
    assert opm.velocity == pytest.approx([0.0, 7500.0, 0.0])
    assert opm.epoch == epoch
    assert not opm.has_keplerian_elements
    assert len(opm.maneuvers) == 0
    assert opm.mass is None


def test_opm_constructor_list_inputs(eop):
    """Test OPM constructor accepts plain lists for position/velocity."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "ITRF2000",
        "UTC",
        epoch,
        [6500e3, 1200e3, -700e3],
        [-800.0, 8700.0, -4200.0],
    )
    assert opm.position == pytest.approx([6500e3, 1200e3, -700e3])
    assert opm.velocity == pytest.approx([-800.0, 8700.0, -4200.0])


def test_opm_constructor_then_serialize(eop):
    """Test constructing an OPM and serializing it round-trips correctly."""
    epoch = Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "MY_ORG",
        "MYSAT",
        "2024-050A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )

    # Add a maneuver
    man_epoch = Epoch.from_datetime(2024, 6, 16, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.add_maneuver(man_epoch, 120.0, "RTN", [10.0, 0.0, 0.0], delta_mass=-5.0)

    # Round-trip through KVN
    kvn = opm.to_string("KVN")
    opm2 = OPM.from_str(kvn)
    assert opm2.object_name == "MYSAT"
    assert opm2.object_id == "2024-050A"
    assert opm2.position == pytest.approx([7000e3, 0.0, 0.0], abs=1.0)
    assert opm2.velocity == pytest.approx([0.0, 7500.0, 0.0], abs=0.001)
    assert len(opm2.maneuvers) == 1
    assert opm2.maneuvers[0].duration == pytest.approx(120.0)
    assert opm2.maneuvers[0].delta_mass == pytest.approx(-5.0)


def test_opm_constructor_then_mutate(eop):
    """Test constructing an OPM and then mutating its fields."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT1",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )

    # Mutate metadata
    opm.object_name = "NEW_SAT"
    assert opm.object_name == "NEW_SAT"

    # Mutate state
    opm.position = [8000e3, 0.0, 0.0]
    assert opm.position[0] == pytest.approx(8000e3)

    # Mutate header
    opm.originator = "NEW_ORG"
    assert opm.originator == "NEW_ORG"


# ─────────────────────────────────────────────
# Additional coverage: spacecraft props, creation_date, maneuver comments,
# constructor then to_dict, and format round-trips
# ─────────────────────────────────────────────


def test_opm_creation_date_getter_and_setter(eop):
    """Test getting and setting creation_date on OPM."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
    original_date = opm.creation_date
    assert isinstance(original_date, Epoch)

    new_date = Epoch.from_datetime(2025, 6, 15, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.creation_date = new_date
    assert opm.creation_date == new_date


def test_opm_spacecraft_parameter_getters(eop):
    """Test spacecraft parameter read access on OPM with parameters."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    # OPMExample2 has spacecraft parameters
    assert opm.mass is not None
    assert opm.mass == pytest.approx(1913.0, abs=0.1)

    assert opm.solar_rad_area is not None
    assert opm.solar_rad_coeff is not None
    assert opm.drag_area is not None
    assert opm.drag_coeff is not None


def test_opm_spacecraft_parameter_getters_none(eop):
    """Test spacecraft parameters are None when not present."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )
    assert opm.mass is None
    assert opm.solar_rad_area is None
    assert opm.solar_rad_coeff is None
    assert opm.drag_area is None
    assert opm.drag_coeff is None


def test_opm_keplerian_element_getters(eop):
    """Test all Keplerian element getters on OPM with Keplerian data."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    assert opm.has_keplerian_elements
    assert opm.semi_major_axis is not None
    assert opm.eccentricity is not None
    assert opm.inclination is not None
    assert opm.ra_of_asc_node is not None
    assert opm.arg_of_pericenter is not None
    assert opm.true_anomaly is not None


def test_opm_keplerian_element_getters_none(eop):
    """Test all Keplerian element getters return None when not present."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )
    assert not opm.has_keplerian_elements
    assert opm.semi_major_axis is None
    assert opm.eccentricity is None
    assert opm.inclination is None
    assert opm.ra_of_asc_node is None
    assert opm.arg_of_pericenter is None
    assert opm.true_anomaly is None
    assert opm.mean_anomaly is None
    assert opm.gm is None


def test_opm_maneuver_comments(eop):
    """Test accessing and setting maneuver comments."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    assert len(opm.maneuvers) >= 1
    # Verify comments property is accessible (may be empty list)
    comments = opm.maneuvers[0].comments
    assert isinstance(comments, list)


def test_opm_standalone_maneuver_comments(eop):
    """Test setting comments on a standalone OPMManeuver."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m = OPMManeuver(
        epoch_ignition=epoch,
        duration=120.0,
        ref_frame="RTN",
        dv=[10.0, 0.0, 0.0],
    )
    assert m.comments == []

    m.comments = ["Test comment 1", "Test comment 2"]
    assert m.comments == ["Test comment 1", "Test comment 2"]


def test_opm_to_dict_from_constructor(eop):
    """Test to_dict() on a programmatically-constructed OPM."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "MY_ORG",
        "SAT1",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )

    d = opm.to_dict()
    assert d["header"]["originator"] == "MY_ORG"
    assert d["metadata"]["object_name"] == "SAT1"
    assert d["metadata"]["object_id"] == "2024-001A"
    assert d["metadata"]["ref_frame"] == "GCRF"
    assert d["state_vector"]["position"] == pytest.approx([7000e3, 0.0, 0.0])
    assert d["state_vector"]["velocity"] == pytest.approx([0.0, 7500.0, 0.0])
    # No Keplerian, maneuvers, or spacecraft params in constructor-only OPM
    assert "keplerian_elements" not in d
    assert "maneuvers" not in d


def test_opm_to_dict_with_maneuvers(eop):
    """Test to_dict() includes maneuvers added after construction."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )
    man_epoch = Epoch.from_datetime(2024, 3, 2, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm.add_maneuver(man_epoch, 120.0, "RTN", [10.0, 0.0, 0.0], delta_mass=-5.0)

    d = opm.to_dict()
    assert "maneuvers" in d
    assert len(d["maneuvers"]) == 1
    assert d["maneuvers"][0]["duration"] == pytest.approx(120.0)
    assert d["maneuvers"][0]["delta_mass"] == pytest.approx(-5.0)
    assert d["maneuvers"][0]["ref_frame"] == "RTN"
    assert d["maneuvers"][0]["dv"] == pytest.approx([10.0, 0.0, 0.0])


def test_opm_constructor_serialize_all_formats(eop):
    """Test that a constructed OPM round-trips through JSON and XML."""
    epoch = Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "MY_SAT",
        "2024-100A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [6500e3, 1200e3, -700e3],
        [-800.0, 8700.0, -4200.0],
    )

    # JSON round-trip
    json_str = opm.to_string("JSON")
    opm_j = OPM.from_str(json_str)
    assert opm_j.object_name == "MY_SAT"
    assert opm_j.position == pytest.approx([6500e3, 1200e3, -700e3], abs=1.0)

    # XML round-trip
    xml_str = opm.to_string("XML")
    opm_x = OPM.from_str(xml_str)
    assert opm_x.object_name == "MY_SAT"
    assert opm_x.velocity == pytest.approx([-800.0, 8700.0, -4200.0], abs=0.001)


def test_opm_json_uppercase_from_constructor(eop):
    """Test to_json_string with uppercase keys on a constructed OPM."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )
    json_str = opm.to_json_string(uppercase_keys=True)
    assert '"OBJECT_NAME"' in json_str
    assert '"header"' in json_str


def test_opm_maneuver_construction_and_mutation(eop):
    """Build maneuver from scratch, mutate all fields, then add to OPM."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m = OPMManeuver(
        epoch_ignition=epoch,
        duration=60.0,
        ref_frame="J2000",
        dv=[1.0, 2.0, 3.0],
    )
    assert m.delta_mass is None

    # Mutate all fields
    new_epoch = Epoch.from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m.epoch_ignition = new_epoch
    assert m.epoch_ignition == new_epoch

    m.duration = 300.0
    assert m.duration == pytest.approx(300.0)

    m.ref_frame = "RTN"
    assert m.ref_frame == "RTN"

    m.dv = [10.0, 20.0, 30.0]
    assert m.dv == pytest.approx([10.0, 20.0, 30.0])

    m.delta_mass = -25.0
    assert m.delta_mass == pytest.approx(-25.0)

    # Add to OPM and verify it persists
    opm = OPM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "GCRF",
        "UTC",
        epoch,
        [7000e3, 0.0, 0.0],
        [0.0, 7500.0, 0.0],
    )
    opm.maneuvers.append(m)
    assert len(opm.maneuvers) == 1
    assert opm.maneuvers[0].duration == pytest.approx(300.0)
    assert opm.maneuvers[0].ref_frame == "RTN"
    assert opm.maneuvers[0].delta_mass == pytest.approx(-25.0)


def test_opm_maneuver_repr(eop):
    """Test repr for OPMManeuver in both owned and proxy modes."""
    epoch = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    m = OPMManeuver(epoch, 60.0, "RTN", [1.0, 0.0, 0.0])
    r = repr(m)
    assert "OPMManeuver" in r
    assert "RTN" in r

    # Proxy mode repr
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    r2 = repr(opm.maneuvers[0])
    assert "OPMManeuver" in r2


def test_opm_state_in_frame(eop):
    """Mirror of test_opm_state_in_frame in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    assert opm.ref_frame == "TOD"

    x_tod = opm.state_in_frame(brahe.CelestialFrame.TOD)
    np.testing.assert_array_equal(x_tod, opm.state)

    x_gcrf = opm.state_in_frame(brahe.CelestialFrame.GCRF)
    np.testing.assert_array_equal(x_gcrf, brahe.state_tod_to_gcrf(opm.epoch, opm.state))
    assert np.linalg.norm(x_gcrf[:3] - x_tod[:3]) > 1.0e3

    x_rf = opm.state_in_frame(brahe.ReferenceFrame.celestial(brahe.CelestialFrame.GCRF))
    np.testing.assert_array_equal(x_rf, x_gcrf)

    x_itrf = opm.state_in_frame(brahe.CelestialFrame.ITRF)
    assert np.linalg.norm(x_itrf[:3]) == pytest.approx(
        np.linalg.norm(x_tod[:3]), abs=1.0
    )


def test_opm_state_in_frame_invalid_type(eop):
    """A non-frame argument raises TypeError."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    with pytest.raises(TypeError):
        opm.state_in_frame("GCRF")


def test_opm_state_in_frame_unsupported_ref_frame(eop):
    """A REF_FRAME with no FrameAxes equivalent raises BraheError."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    opm.ref_frame = "TEME"
    with pytest.raises(brahe.BraheError, match="TEME"):
        opm.state_in_frame(brahe.CelestialFrame.GCRF)


def test_opm_state_in_frame_moon_center(eop, naif_cache_setup):
    """Mirror of test_opm_state_in_frame_moon_center in Rust."""
    brahe.load_spice_kernel("de440s")
    opm = OPM.from_file("test_assets/ccsds/opm/OPM-dummy-moon-EME2000.txt")
    assert opm.center_name == "MOON"
    raw = opm.state

    # EME2000 axes about the Moon differ from LCI by the frame bias only.
    x_lci = opm.state_in_frame(brahe.CelestialFrame.LCI)
    np.testing.assert_allclose(x_lci, brahe.state_eme2000_to_gcrf(raw), atol=1e-6)

    # Retargeting to GCRF translates by the Earth-Moon separation.
    x_gcrf = opm.state_in_frame(brahe.CelestialFrame.GCRF)
    assert np.linalg.norm(x_gcrf[:3] - x_lci[:3]) > 3.0e8


def test_opm_state_in_frame_unknown_center_errors(eop):
    """Mirror of test_opm_state_in_frame_unknown_center_errors in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
    opm.center_name = "PLANET X"
    with pytest.raises(brahe.BraheError, match="PLANET X"):
        opm.state_in_frame(brahe.CelestialFrame.GCRF)


def test_opm_mci_ref_frame_resolves(eop):
    """Mirror of test_opm_mci_ref_frame_resolves in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample7.txt")
    assert opm.ref_frame == "MCI"
    assert opm.center_name == "MARS"

    # MCI names ICRF-aligned axes about Mars, which is exactly
    # CelestialFrame.MCI, so no conversion is applied.
    x_mci = opm.state_in_frame(brahe.CelestialFrame.MCI)
    np.testing.assert_array_equal(x_mci, opm.state)

    # The token survives a round trip through the writer.
    assert "MCI" in opm.to_string("kvn")
    assert OPM.from_str(opm.to_string("kvn")).ref_frame == "MCI"


def test_opm_icrf_ref_frame_resolves(eop):
    """Mirror of test_opm_icrf_ref_frame_resolves in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample8.txt")
    assert opm.ref_frame == "ICRF"
    assert opm.center_name == "SSB"

    # ICRF and GCRF name the same orientation, so both tokens resolve to the
    # ICRF axes and the center chooses the origin. CENTER_NAME = SSB resolves
    # through NAIFId.from_name to the solar system barycenter, making the
    # message frame SSBI.
    x_ssbi = opm.state_in_frame(brahe.CelestialFrame.SSBI)
    np.testing.assert_allclose(x_ssbi, opm.state, atol=1e-9)

    # An ICRF message centered on Earth resolves to GCRF instead.
    opm_earth = OPM.from_file("test_assets/ccsds/opm/OPMExample8.txt")
    opm_earth.center_name = "EARTH"
    x_gcrf = opm_earth.state_in_frame(brahe.CelestialFrame.GCRF)
    np.testing.assert_allclose(x_gcrf, opm_earth.state, atol=1e-9)

    # The token survives a round trip through the writer.
    assert "ICRF" in opm.to_string("kvn")


def test_opm_state_in_frame_frozen_tod_frame_epoch(eop):
    """Mirror of test_opm_state_in_frame_frozen_tod_frame_epoch in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2_ref_epoch.txt")
    assert opm.ref_frame == "TOD"
    ref_epoch = Epoch.from_string("2006-06-02T00:00:00.000Z")

    x_gcrf = opm.state_in_frame(brahe.CelestialFrame.GCRF)
    expected = brahe.state_tod_to_gcrf(ref_epoch, opm.state)
    np.testing.assert_allclose(x_gcrf, expected, atol=1e-9)

    of_date = brahe.state_tod_to_gcrf(opm.epoch, opm.state)
    assert np.linalg.norm(x_gcrf[:3] - of_date[:3]) > 1.0
