import pytest

import brahe as bh


def test_attitude_frame_reference():
    frame = bh.AttitudeFrame.reference(bh.ReferenceFrame.GCRF)
    assert frame.is_reference
    assert not frame.is_spacecraft
    assert frame.reference_frame == bh.ReferenceFrame.GCRF
    assert str(frame) == "GCRF"


def test_attitude_frame_orbit_relative():
    frame = bh.AttitudeFrame.orbit_relative("RTN", "ROTATING")
    assert frame.is_orbit_relative
    assert frame.reference_frame is None
    assert str(frame) == "RTN (rotating)"
    with pytest.raises(ValueError):
        bh.AttitudeFrame.orbit_relative("XYZ", "ROTATING")


def test_attitude_frame_spacecraft():
    frame = bh.AttitudeFrame.spacecraft("SC_BODY", "1")
    assert frame.is_spacecraft
    assert str(frame) == "SC_BODY_1"
    bare = bh.AttitudeFrame.spacecraft("SC_BODY")
    assert str(bare) == "SC_BODY"
    with pytest.raises(ValueError):
        bh.AttitudeFrame.spacecraft("WHEEL", "1")


def test_attitude_frame_equality():
    a = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    b = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    assert a == b
    assert a != bh.AttitudeFrame.spacecraft("SC_BODY", "1")
