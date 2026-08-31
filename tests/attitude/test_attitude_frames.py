import pytest

import brahe as bh


def test_attitude_frame_reference():
    frame = bh.AttitudeFrame.reference_frame(bh.ReferenceFrame.GCRF)
    assert frame.is_reference_frame
    assert not frame.is_spacecraft_body_frame
    assert frame.as_reference_frame == bh.ReferenceFrame.GCRF
    assert str(frame) == "GCRF"


def test_attitude_frame_orbit_relative():
    frame = bh.AttitudeFrame.orbit_relative_frame("RTN", "ROTATING")
    assert frame.is_orbit_relative_frame
    assert frame.as_reference_frame is None
    assert str(frame) == "RTN (rotating)"
    with pytest.raises(ValueError):
        bh.AttitudeFrame.orbit_relative_frame("XYZ", "ROTATING")


def test_attitude_frame_orbit_relative_rejects_inertial_only_kinds():
    with pytest.raises(ValueError):
        bh.AttitudeFrame.orbit_relative_frame("EQW", "ROTATING")
    with pytest.raises(ValueError):
        bh.AttitudeFrame.orbit_relative_frame("PQW", "ROTATING")
    assert bh.AttitudeFrame.orbit_relative_frame(
        "EQW", "INERTIAL"
    ).is_orbit_relative_frame
    assert bh.AttitudeFrame.orbit_relative_frame(
        "PQW", "INERTIAL"
    ).is_orbit_relative_frame


def test_attitude_frame_spacecraft():
    frame = bh.AttitudeFrame.spacecraft_body_frame("SC_BODY", "1")
    assert frame.is_spacecraft_body_frame
    assert str(frame) == "SC_BODY_1"
    bare = bh.AttitudeFrame.spacecraft_body_frame("SC_BODY")
    assert str(bare) == "SC_BODY"
    with pytest.raises(ValueError):
        bh.AttitudeFrame.spacecraft_body_frame("WHEEL", "1")


def test_attitude_frame_equality():
    a = bh.AttitudeFrame.reference_frame(bh.ReferenceFrame.EME2000)
    b = bh.AttitudeFrame.reference_frame(bh.ReferenceFrame.EME2000)
    assert a == b
    assert a != bh.AttitudeFrame.spacecraft_body_frame("SC_BODY", "1")
