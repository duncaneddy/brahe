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
    assert opm.position == [1.0, 2.0, 3.0]

    opm.velocity = [4.0, 5.0, 6.0]
    assert opm.velocity == [4.0, 5.0, 6.0]

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
