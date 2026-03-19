"""Tests for CCSDS OPM parsing — parity with Rust tests."""

import pytest
from brahe.ccsds import OPM


def test_opm_parse_example1(eop):
    """Mirror of test_parse_opm_example1 in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")

    assert opm.format_version() == pytest.approx(3.0, abs=1e-10)
    assert opm.object_name() == "GODZILLA 5"
    assert opm.object_id() == "1998-999A"
    assert opm.ref_frame() == "ITRF2000"
    assert opm.time_system() == "UTC"

    # State vector (km → m)
    pos = opm.position()
    assert pos[0] == pytest.approx(6503514.0, abs=1.0)
    assert pos[1] == pytest.approx(1239647.0, abs=1.0)
    assert pos[2] == pytest.approx(-717490.0, abs=1.0)

    vel = opm.velocity()
    assert vel[0] == pytest.approx(-873.160, abs=0.001)
    assert vel[1] == pytest.approx(8740.420, abs=0.001)
    assert vel[2] == pytest.approx(-4191.076, abs=0.001)

    # Spacecraft parameters
    assert opm.mass() == pytest.approx(3000.0, abs=1e-3)

    # No Keplerian, no maneuvers, no covariance
    assert not opm.has_keplerian_elements()
    assert opm.num_maneuvers() == 0


def test_opm_parse_example2_with_keplerian_and_maneuvers(eop):
    """Mirror of test_parse_opm_example2_with_keplerian_and_maneuvers in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")

    assert opm.object_name() == "EUTELSAT W4"
    assert opm.ref_frame() == "TOD"

    # State vector
    pos = opm.position()
    assert pos[0] == pytest.approx(6655994.2, abs=1.0)

    # Keplerian elements
    assert opm.has_keplerian_elements()
    assert opm.semi_major_axis() == pytest.approx(41399512.3, abs=1.0)

    # 2 maneuvers
    assert opm.num_maneuvers() == 2

    m1 = opm.maneuver(0)
    assert m1["duration"] == pytest.approx(132.60, abs=0.01)
    assert m1["delta_mass"] == pytest.approx(-18.418, abs=0.001)
    assert m1["ref_frame"] == "J2000"
    assert m1["dv"][0] == pytest.approx(-23.257, abs=0.001)

    m2 = opm.maneuver(1)
    assert m2["ref_frame"] == "RTN"


def test_opm_parse_example5_three_maneuvers(eop):
    """Mirror of test_parse_opm_example5_with_three_maneuvers in Rust."""
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample5.txt")

    assert opm.ref_frame() == "GCRF"
    assert opm.time_system() == "GPS"
    assert opm.num_maneuvers() == 3


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
