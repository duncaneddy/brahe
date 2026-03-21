"""Tests for CCSDS OMM parsing and mutation — parity with Rust tests."""

import pytest
import brahe as bh
from brahe.ccsds import OMM


def test_omm_parse_example1(eop):
    """Mirror of test_parse_omm_example1 in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

    assert omm.format_version == pytest.approx(3.0, abs=1e-10)
    assert omm.object_name == "GOES 9"
    assert omm.object_id == "1995-025A"
    assert omm.center_name == "EARTH"
    assert omm.ref_frame == "TEME"
    assert omm.time_system == "UTC"
    assert omm.mean_element_theory == "SGP/SGP4"

    # Mean elements
    assert omm.mean_motion == pytest.approx(1.00273272, abs=1e-10)
    assert omm.eccentricity == pytest.approx(0.0005013, abs=1e-10)
    assert omm.inclination == pytest.approx(3.0539, abs=1e-4)
    assert omm.ra_of_asc_node == pytest.approx(81.7939, abs=1e-4)
    assert omm.arg_of_pericenter == pytest.approx(249.2363, abs=1e-4)
    assert omm.mean_anomaly == pytest.approx(150.1602, abs=1e-4)
    assert omm.gm == pytest.approx(398600.8e9, abs=1e3)

    # TLE parameters
    assert omm.ephemeris_type == 0
    assert omm.classification_type == "U"
    assert omm.norad_cat_id == 23581
    assert omm.element_set_no == 925
    assert omm.rev_at_epoch == 4316
    assert omm.bstar == pytest.approx(0.0001, abs=1e-10)
    assert omm.mean_motion_dot == pytest.approx(-0.00000113, abs=1e-12)
    assert omm.mean_motion_ddot == pytest.approx(0.0, abs=1e-15)


def test_omm_parse_example4(eop):
    """Mirror of test_parse_omm_example4 in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample4.txt")

    assert omm.object_name == "STARLETTE"
    assert omm.object_id == "1975-010A"
    assert omm.mean_motion == pytest.approx(13.82309053, abs=1e-8)
    assert omm.eccentricity == pytest.approx(0.0205751, abs=1e-7)
    assert omm.norad_cat_id == 7646
    assert omm.bstar == pytest.approx(-4.7102e-6, abs=1e-12)


def test_omm_parse_example5_sgp4xp(eop):
    """Mirror of test_parse_omm_example5_sgp4xp in Rust."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample5.txt")

    assert omm.mean_element_theory == "SGP4-XP"
    assert omm.ephemeris_type == 4


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


def test_omm_parse_example3_unsupported_time_system(eop):
    """OMMExample3.txt uses TIME_SYSTEM=MRT which is unsupported for epoch conversion."""
    with pytest.raises(Exception, match="MRT"):
        OMM.from_file("test_assets/ccsds/omm/OMMExample3.txt")


def test_omm_json_round_trip(eop):
    """OMM JSON round-trip: from_file -> to_string(JSON) -> from_str -> compare."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    json_str = omm.to_string("JSON")
    omm2 = OMM.from_str(json_str)
    assert omm2.object_name == omm.object_name
    assert omm2.object_id == omm.object_id
    assert omm2.eccentricity == pytest.approx(omm.eccentricity, abs=1e-10)
    assert omm2.inclination == pytest.approx(omm.inclination, abs=1e-10)


def test_omm_json_uppercase_keys(eop):
    """OMM to_json_string with uppercase keys."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    json_str = omm.to_json_string(uppercase_keys=True)
    assert '"OBJECT_NAME"' in json_str
    assert '"header"' in json_str  # container keys always lowercase


def test_omm_repr(eop):
    """Test OMM repr."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    r = repr(omm)
    assert "GOES 9" in r
    assert "1995-025A" in r


# ─────────────────────────────────────────────
# Mutation tests
# ─────────────────────────────────────────────


def test_omm_metadata_setters(eop):
    """Test setting metadata on OMM."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

    omm.object_name = "ISS"
    assert omm.object_name == "ISS"

    omm.object_id = "1998-067A"
    assert omm.object_id == "1998-067A"

    omm.center_name = "MOON"
    assert omm.center_name == "MOON"

    omm.ref_frame = "GCRF"
    assert omm.ref_frame == "GCRF"

    omm.time_system = "TAI"
    assert omm.time_system == "TAI"

    omm.mean_element_theory = "SGP4-XP"
    assert omm.mean_element_theory == "SGP4-XP"


def test_omm_mean_element_setters(eop):
    """Test setting mean element values on OMM."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

    omm.eccentricity = 0.001
    assert omm.eccentricity == pytest.approx(0.001)

    omm.inclination = 51.6
    assert omm.inclination == pytest.approx(51.6)

    omm.ra_of_asc_node = 100.0
    assert omm.ra_of_asc_node == pytest.approx(100.0)

    omm.arg_of_pericenter = 200.0
    assert omm.arg_of_pericenter == pytest.approx(200.0)

    omm.mean_anomaly = 300.0
    assert omm.mean_anomaly == pytest.approx(300.0)

    omm.mean_motion = 15.0
    assert omm.mean_motion == pytest.approx(15.0)

    omm.gm = 398600.4418e9
    assert omm.gm == pytest.approx(398600.4418e9)


def test_omm_tle_parameter_setters(eop):
    """Test setting TLE parameters on OMM."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

    omm.norad_cat_id = 25544
    assert omm.norad_cat_id == 25544

    omm.bstar = 0.0002
    assert omm.bstar == pytest.approx(0.0002)

    omm.mean_motion_dot = -0.00001
    assert omm.mean_motion_dot == pytest.approx(-0.00001)

    omm.mean_motion_ddot = 0.0
    assert omm.mean_motion_ddot == pytest.approx(0.0)

    omm.element_set_no = 999
    assert omm.element_set_no == 999

    omm.rev_at_epoch = 5000
    assert omm.rev_at_epoch == 5000

    omm.classification_type = "S"
    assert omm.classification_type == "S"

    omm.ephemeris_type = 2
    assert omm.ephemeris_type == 2

    omm.format_version = 2.0
    assert omm.format_version == pytest.approx(2.0)


# ─────────────────────────────────────────────
# GPRecord <-> OMM conversion tests
# ─────────────────────────────────────────────

SAMPLE_GP_JSON = (
    '{"CCSDS_OMM_VERS": "3.0", "CREATION_DATE": "2024-01-15 12:00:00",'
    ' "ORIGINATOR": "18 SDS", "OBJECT_NAME": "ISS (ZARYA)",'
    ' "OBJECT_ID": "1998-067A", "CENTER_NAME": "EARTH",'
    ' "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC",'
    ' "MEAN_ELEMENT_THEORY": "SGP4",'
    ' "EPOCH": "2024-01-15T12:00:00.000000",'
    ' "MEAN_MOTION": 15.5, "ECCENTRICITY": 0.0001,'
    ' "INCLINATION": 51.64, "RA_OF_ASC_NODE": 200.0,'
    ' "ARG_OF_PERICENTER": 100.0, "MEAN_ANOMALY": 260.0,'
    ' "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U",'
    ' "NORAD_CAT_ID": 25544, "ELEMENT_SET_NO": 999,'
    ' "REV_AT_EPOCH": 45000, "BSTAR": 0.000341,'
    ' "MEAN_MOTION_DOT": 0.00001, "MEAN_MOTION_DDOT": 0.0}'
)


def test_omm_from_gp_record(eop):
    """Test GPRecord.to_omm() conversion."""
    record = bh.GPRecord.from_json(SAMPLE_GP_JSON)
    omm = record.to_omm()

    assert omm.format_version == pytest.approx(3.0, abs=1e-10)
    assert omm.originator == "18 SDS"
    assert omm.object_name == "ISS (ZARYA)"
    assert omm.object_id == "1998-067A"
    assert omm.center_name == "EARTH"
    assert omm.ref_frame == "TEME"
    assert omm.time_system == "UTC"
    assert omm.mean_element_theory == "SGP4"

    assert omm.eccentricity == pytest.approx(0.0001, abs=1e-10)
    assert omm.inclination == pytest.approx(51.64, abs=1e-4)
    assert omm.ra_of_asc_node == pytest.approx(200.0, abs=1e-4)
    assert omm.arg_of_pericenter == pytest.approx(100.0, abs=1e-4)
    assert omm.mean_anomaly == pytest.approx(260.0, abs=1e-4)
    assert omm.mean_motion == pytest.approx(15.5, abs=1e-8)

    assert omm.ephemeris_type == 0
    assert omm.classification_type == "U"
    assert omm.norad_cat_id == 25544
    assert omm.element_set_no == 999
    assert omm.rev_at_epoch == 45000
    assert omm.bstar == pytest.approx(0.000341, abs=1e-10)


def test_omm_from_gp_record_missing_required(eop):
    """Test that missing required fields raise an error."""
    # Missing epoch
    record = bh.GPRecord.from_json(
        '{"ECCENTRICITY": 0.001, "INCLINATION": 51.64,'
        ' "RA_OF_ASC_NODE": 200.0, "ARG_OF_PERICENTER": 100.0,'
        ' "MEAN_ANOMALY": 260.0}'
    )
    with pytest.raises(Exception, match="EPOCH"):
        record.to_omm()


def test_omm_to_gp_record(eop):
    """Test OMM.to_gp_record() conversion."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    gp = omm.to_gp_record()

    assert gp.object_name == "GOES 9"
    assert gp.object_id == "1995-025A"
    assert gp.center_name == "EARTH"
    assert gp.ref_frame == "TEME"
    assert gp.time_system == "UTC"
    assert gp.eccentricity == pytest.approx(0.0005013, abs=1e-10)
    assert gp.inclination == pytest.approx(3.0539, abs=1e-4)
    assert gp.norad_cat_id == 23581
    assert gp.classification_type == "U"
    assert gp.bstar == pytest.approx(0.0001, abs=1e-10)


def test_omm_gp_record_roundtrip(eop):
    """Test GPRecord -> OMM -> GPRecord preserves fields."""
    record = bh.GPRecord.from_json(SAMPLE_GP_JSON)
    omm = record.to_omm()
    roundtripped = omm.to_gp_record()

    assert roundtripped.object_name == record.object_name
    assert roundtripped.object_id == record.object_id
    assert roundtripped.center_name == record.center_name
    assert roundtripped.ref_frame == record.ref_frame
    assert roundtripped.time_system == record.time_system
    assert roundtripped.mean_element_theory == record.mean_element_theory

    assert roundtripped.eccentricity == pytest.approx(record.eccentricity, abs=1e-10)
    assert roundtripped.inclination == pytest.approx(record.inclination, abs=1e-10)
    assert roundtripped.ra_of_asc_node == pytest.approx(
        record.ra_of_asc_node, abs=1e-10
    )
    assert roundtripped.arg_of_pericenter == pytest.approx(
        record.arg_of_pericenter, abs=1e-10
    )
    assert roundtripped.mean_anomaly == pytest.approx(record.mean_anomaly, abs=1e-10)
    assert roundtripped.mean_motion == pytest.approx(record.mean_motion, abs=1e-10)

    assert roundtripped.norad_cat_id == record.norad_cat_id
    assert roundtripped.classification_type == record.classification_type
    assert roundtripped.rev_at_epoch == record.rev_at_epoch
    assert roundtripped.bstar == pytest.approx(record.bstar, abs=1e-10)


def test_omm_from_gp_record_static_method(eop):
    """Test OMM.from_gp_record() static method alternative."""
    record = bh.GPRecord.from_json(SAMPLE_GP_JSON)
    omm = OMM.from_gp_record(record)

    assert omm.object_name == "ISS (ZARYA)"
    assert omm.eccentricity == pytest.approx(0.0001, abs=1e-10)


def test_omm_kvn_round_trip(eop):
    """Test OMM KVN write then re-parse preserves data."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    kvn_str = omm.to_string("KVN")
    omm2 = OMM.from_str(kvn_str)
    assert omm2.object_name == omm.object_name
    assert omm2.object_id == omm.object_id
    assert omm2.eccentricity == pytest.approx(omm.eccentricity, abs=1e-10)
    assert omm2.inclination == pytest.approx(omm.inclination, abs=1e-10)
    assert omm2.mean_motion == pytest.approx(omm.mean_motion, abs=1e-10)


def test_omm_xml_round_trip(eop):
    """Test OMM XML write then re-parse preserves data."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample2.xml")
    xml_str = omm.to_string("XML")
    omm2 = OMM.from_str(xml_str)
    assert omm2.object_name == omm.object_name
    assert omm2.object_id == omm.object_id
    assert omm2.eccentricity == pytest.approx(omm.eccentricity, abs=1e-10)
    assert omm2.mean_motion == pytest.approx(omm.mean_motion, abs=1e-10)


def test_omm_xml_parse_example4(eop):
    """Test parsing OMM XML Example 4 (STARLETTE)."""
    omm = OMM.from_file("test_assets/ccsds/omm/OMMExample4.xml")
    assert omm.object_name == "STARLETTE"
    assert omm.object_id == "1975-010A"
    assert omm.mean_motion == pytest.approx(13.82309053, abs=1e-8)


def _assert_omm_fields(omm1, omm2):
    """Assert all accessible OMM fields match."""
    # Header + metadata
    assert omm2.format_version == omm1.format_version
    assert omm2.originator == omm1.originator
    assert omm2.object_name == omm1.object_name
    assert omm2.object_id == omm1.object_id
    assert omm2.center_name == omm1.center_name
    assert omm2.ref_frame == omm1.ref_frame
    assert omm2.time_system == omm1.time_system
    assert omm2.mean_element_theory == omm1.mean_element_theory
    # Mean elements
    assert omm2.eccentricity == pytest.approx(omm1.eccentricity, abs=1e-10)
    assert omm2.inclination == pytest.approx(omm1.inclination, abs=1e-6)
    assert omm2.ra_of_asc_node == pytest.approx(omm1.ra_of_asc_node, abs=1e-6)
    assert omm2.arg_of_pericenter == pytest.approx(omm1.arg_of_pericenter, abs=1e-6)
    assert omm2.mean_anomaly == pytest.approx(omm1.mean_anomaly, abs=1e-6)
    if omm1.mean_motion is not None:
        assert omm2.mean_motion == pytest.approx(omm1.mean_motion, abs=1e-10)
    if omm1.gm is not None:
        assert omm2.gm == pytest.approx(omm1.gm, abs=1e3)
    # TLE parameters
    if omm1.norad_cat_id is not None:
        assert omm2.norad_cat_id == omm1.norad_cat_id
    if omm1.element_set_no is not None:
        assert omm2.element_set_no == omm1.element_set_no
    if omm1.rev_at_epoch is not None:
        assert omm2.rev_at_epoch == omm1.rev_at_epoch
    if omm1.bstar is not None:
        assert omm2.bstar == pytest.approx(omm1.bstar, abs=1e-10)
    if omm1.mean_motion_dot is not None:
        assert omm2.mean_motion_dot == pytest.approx(omm1.mean_motion_dot, abs=1e-12)


def test_omm_kvn_full_round_trip(eop):
    """Full-field OMM KVN round-trip with covariance + TLE params."""
    omm1 = OMM.from_file("test_assets/ccsds/omm/OMMExample2.txt")
    kvn = omm1.to_string("KVN")
    omm2 = OMM.from_str(kvn)
    _assert_omm_fields(omm1, omm2)


def test_omm_xml_full_round_trip(eop):
    """Full-field OMM XML round-trip."""
    omm1 = OMM.from_file("test_assets/ccsds/omm/OMMExample2.txt")
    xml = omm1.to_string("XML")
    omm2 = OMM.from_str(xml)
    _assert_omm_fields(omm1, omm2)


def test_omm_json_full_round_trip(eop):
    """Full-field OMM JSON round-trip."""
    omm1 = OMM.from_file("test_assets/ccsds/omm/OMMExample2.txt")
    json_str = omm1.to_string("JSON")
    omm2 = OMM.from_str(json_str)
    _assert_omm_fields(omm1, omm2)


# ─────────────────────────────────────────────
# Programmatic constructor tests
# ─────────────────────────────────────────────


def test_omm_constructor(eop):
    """Test creating an OMM programmatically with the constructor."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omm = OMM(
        originator="NOAA",
        object_name="ISS",
        object_id="1998-067A",
        center_name="EARTH",
        ref_frame="TEME",
        time_system="UTC",
        mean_element_theory="SGP/SGP4",
        epoch=epoch,
        eccentricity=0.0001,
        inclination=51.64,
        ra_of_asc_node=200.0,
        arg_of_pericenter=100.0,
        mean_anomaly=260.0,
        mean_motion=15.5,
    )

    assert omm.originator == "NOAA"
    assert omm.object_name == "ISS"
    assert omm.object_id == "1998-067A"
    assert omm.center_name == "EARTH"
    assert omm.ref_frame == "TEME"
    assert omm.time_system == "UTC"
    assert omm.mean_element_theory == "SGP/SGP4"
    assert omm.format_version == pytest.approx(3.0)
    assert omm.eccentricity == pytest.approx(0.0001)
    assert omm.inclination == pytest.approx(51.64)
    assert omm.ra_of_asc_node == pytest.approx(200.0)
    assert omm.arg_of_pericenter == pytest.approx(100.0)
    assert omm.mean_anomaly == pytest.approx(260.0)
    assert omm.mean_motion == pytest.approx(15.5)
    assert omm.epoch == epoch


def test_omm_constructor_with_gm(eop):
    """Test OMM constructor with GM parameter."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omm = OMM(
        "NOAA",
        "SAT1",
        "2024-001A",
        "EARTH",
        "TEME",
        "UTC",
        "SGP/SGP4",
        epoch,
        0.001,
        51.6,
        200.0,
        100.0,
        260.0,
        gm=398600.4418e9,
    )
    assert omm.gm == pytest.approx(398600.4418e9)


def test_omm_constructor_minimal(eop):
    """Test OMM constructor with only required fields (no mean_motion/gm)."""
    epoch = bh.Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omm = OMM(
        "ORG",
        "SAT",
        "2024-001A",
        "EARTH",
        "TEME",
        "UTC",
        "SGP4",
        epoch,
        0.01,
        97.8,
        15.0,
        30.0,
        45.0,
    )
    assert omm.eccentricity == pytest.approx(0.01)
    assert omm.inclination == pytest.approx(97.8)
    assert omm.mean_motion is None
    assert omm.gm is None


def test_omm_constructor_then_serialize(eop):
    """Test constructing an OMM then round-tripping through KVN."""
    epoch = bh.Epoch.from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omm = OMM(
        "MY_ORG",
        "MYSAT",
        "2024-050A",
        "EARTH",
        "TEME",
        "UTC",
        "SGP/SGP4",
        epoch,
        0.0001,
        51.64,
        200.0,
        100.0,
        260.0,
        mean_motion=15.5,
    )

    # Round-trip through KVN
    kvn = omm.to_string("KVN")
    omm2 = OMM.from_str(kvn)
    assert omm2.object_name == "MYSAT"
    assert omm2.object_id == "2024-050A"
    assert omm2.eccentricity == pytest.approx(0.0001, abs=1e-10)
    assert omm2.inclination == pytest.approx(51.64, abs=1e-4)
    assert omm2.mean_motion == pytest.approx(15.5, abs=1e-8)


def test_omm_constructor_then_mutate(eop):
    """Test constructing an OMM and mutating its fields."""
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omm = OMM(
        "ORG",
        "SAT1",
        "2024-001A",
        "EARTH",
        "TEME",
        "UTC",
        "SGP/SGP4",
        epoch,
        0.0001,
        51.64,
        200.0,
        100.0,
        260.0,
        mean_motion=15.5,
    )

    omm.object_name = "NEW_SAT"
    assert omm.object_name == "NEW_SAT"

    omm.eccentricity = 0.01
    assert omm.eccentricity == pytest.approx(0.01)

    omm.mean_motion = 14.0
    assert omm.mean_motion == pytest.approx(14.0)
