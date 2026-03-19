"""Tests for CCSDS OMM parsing — parity with Rust tests."""

import pytest
from brahe.ccsds import OMM


def test_omm_parse_example1(eop):
    """Mirror of test_parse_omm_example1 in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

    assert omm.format_version() == pytest.approx(3.0, abs=1e-10)
    assert omm.object_name() == "GOES 9"
    assert omm.object_id() == "1995-025A"
    assert omm.center_name() == "EARTH"
    assert omm.ref_frame() == "TEME"
    assert omm.time_system() == "UTC"
    assert omm.mean_element_theory() == "SGP/SGP4"

    # Mean elements
    assert omm.mean_motion() == pytest.approx(1.00273272, abs=1e-10)
    assert omm.eccentricity() == pytest.approx(0.0005013, abs=1e-10)
    assert omm.inclination() == pytest.approx(3.0539, abs=1e-4)
    assert omm.ra_of_asc_node() == pytest.approx(81.7939, abs=1e-4)
    assert omm.arg_of_pericenter() == pytest.approx(249.2363, abs=1e-4)
    assert omm.mean_anomaly() == pytest.approx(150.1602, abs=1e-4)
    assert omm.gm() == pytest.approx(398600.8e9, abs=1e3)

    # TLE parameters
    assert omm.ephemeris_type() == 0
    assert omm.classification_type() == "U"
    assert omm.norad_cat_id() == 23581
    assert omm.element_set_no() == 925
    assert omm.rev_at_epoch() == 4316
    assert omm.bstar() == pytest.approx(0.0001, abs=1e-10)
    assert omm.mean_motion_dot() == pytest.approx(-0.00000113, abs=1e-12)
    assert omm.mean_motion_ddot() == pytest.approx(0.0, abs=1e-15)


def test_omm_parse_example4(eop):
    """Mirror of test_parse_omm_example4 in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample4.txt")

    assert omm.object_name() == "STARLETTE"
    assert omm.object_id() == "1975-010A"
    assert omm.mean_motion() == pytest.approx(13.82309053, abs=1e-8)
    assert omm.eccentricity() == pytest.approx(0.0205751, abs=1e-7)
    assert omm.norad_cat_id() == 7646
    assert omm.bstar() == pytest.approx(-4.7102e-6, abs=1e-12)


def test_omm_parse_example5_sgp4xp(eop):
    """Mirror of test_parse_omm_example5_sgp4xp in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample5.txt")

    assert omm.mean_element_theory() == "SGP4-XP"
    assert omm.ephemeris_type() == 4


def test_omm_to_dict(eop):
    """Test to_dict() serialization."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    d = omm.to_dict()

    assert d["header"]["originator"] == "NOAA/USA"
    assert d["metadata"]["object_name"] == "GOES 9"
    assert d["metadata"]["ref_frame"] == "TEME"
    assert d["mean_elements"]["eccentricity"] == pytest.approx(0.0005013)
    assert d["tle_parameters"]["norad_cat_id"] == 23581
    assert d["tle_parameters"]["bstar"] == pytest.approx(0.0001)


def test_omm_to_dict_with_user_defined(eop):
    """Test to_dict() with user-defined parameters."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample3.txt")
    d = omm.to_dict()

    assert "user_defined" in d
    assert d["user_defined"]["EARTH_MODEL"] == "WGS-84"
    assert "spacecraft_parameters" in d
    assert d["spacecraft_parameters"]["mass"] == pytest.approx(300.0)


def test_omm_repr(eop):
    """Test OMM repr."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    r = repr(omm)
    assert "GOES 9" in r
    assert "1995-025A" in r
