"""Tests for CCSDS APM parsing, mutation, and construction — parity with Rust tests."""

import math

import numpy as np
import pytest

import brahe as bh
from brahe import AngleFormat, Epoch, EulerAngle, EulerAngleOrder, Quaternion
from brahe.ccsds import (
    APM,
    APMAngularVelocity,
    APMEulerState,
    APMInertia,
    APMManeuver,
    APMQuaternionState,
    APMSpin,
)


def test_apm_parse_example_g1_quaternion(eop):
    """Mirror of test_parse_apm_example_g1_quaternion in Rust."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")

    # Header
    assert apm.format_version == pytest.approx(2.0, abs=1e-10)
    assert apm.originator == "GSFC"
    assert apm.message_id == "A7015Z1"

    # Metadata
    assert apm.object_name == "TRMM"
    assert apm.object_id == "1997-074A"
    assert apm.center_name == "EARTH"
    assert apm.time_system == "UTC"

    # Data top-level
    assert isinstance(apm.epoch, Epoch)

    # Quaternion block
    assert len(apm.quaternion_states) == 1
    q = apm.quaternion_states[0]
    assert q.ref_frame_a == "SC_BODY_1"
    assert q.ref_frame_b == "ITRF1997"
    v = q.quaternion.to_vector(scalar_first=False)
    assert v[0] == pytest.approx(0.00005, abs=1e-4)
    assert v[1] == pytest.approx(0.87543, abs=1e-4)
    assert v[2] == pytest.approx(0.40949, abs=1e-4)
    assert v[3] == pytest.approx(0.25678, abs=1e-4)
    assert q.quaternion_derivative is None
    assert q.comments == []

    assert len(apm.euler_states) == 0
    assert len(apm.angular_velocities) == 0
    assert len(apm.spins) == 0
    assert len(apm.inertias) == 0
    assert len(apm.maneuvers) == 0


def test_apm_parse_example_g2_euler_angles(eop):
    """Mirror of test_parse_apm_example_g2_euler_angles in Rust."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG2.txt")

    assert apm.originator == "GSFC"
    assert apm.message_id == "A7015Z2"
    assert apm.object_name == "GOES-P"
    assert apm.object_id == "2006-003A"
    assert apm.center_name == "EARTH"

    assert len(apm.euler_states) == 1
    e = apm.euler_states[0]
    assert e.ref_frame_a == "BODY_FRAME_A"
    assert e.ref_frame_b == "ITRF1997"
    assert e.angles.order == EulerAngleOrder.YXY
    assert e.angles.phi == pytest.approx(math.radians(-26.78), abs=1e-10)
    assert e.angles.theta == pytest.approx(math.radians(46.26), abs=1e-10)
    assert e.angles.psi == pytest.approx(math.radians(144.10), abs=1e-10)
    assert e.rates is None
    assert e.comments == ["Euler angles"]
    assert len(apm.quaternion_states) == 0


def test_apm_parse_example_g3_multi_quat_inertia_maneuver(eop):
    """Mirror of test_parse_apm_example_g3_multi_quat_inertia_maneuver in Rust."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG3.txt")

    assert apm.originator == "JPL"
    assert apm.message_id == "900018"
    assert apm.object_name == "MARS SPIRIT"

    # Two quaternion blocks
    assert len(apm.quaternion_states) == 2
    q0 = apm.quaternion_states[0]
    assert q0.ref_frame_a == "ITRF1997"
    assert q0.ref_frame_b == "INSTRUMENT_A"
    v0 = q0.quaternion.to_vector(scalar_first=False)
    assert v0[0] == pytest.approx(0.03123, abs=1e-4)
    assert v0[1] == pytest.approx(0.78543, abs=1e-4)
    assert v0[2] == pytest.approx(0.39158, abs=1e-4)
    assert v0[3] == pytest.approx(0.47832, abs=1e-4)
    assert q0.comments == ["Attitude state quaternion (ref frame = ITRF1997)"]

    q1 = apm.quaternion_states[1]
    assert q1.ref_frame_a == "ICRF"
    v1 = q1.quaternion.to_vector(scalar_first=False)
    assert v1[0] == pytest.approx(0.02478, abs=1e-4)
    assert v1[1] == pytest.approx(0.78576, abs=1e-4)
    assert v1[2] == pytest.approx(0.39552, abs=1e-4)
    assert v1[3] == pytest.approx(0.47491, abs=1e-4)
    assert q1.comments == ["Attitude state quaternion (ref frame = ICRF)"]

    # Inertia block
    assert len(apm.inertias) == 1
    inertia = apm.inertias[0]
    assert inertia.inertia_ref_frame == "SC_BODY_1"
    assert inertia.ixx == pytest.approx(6080.0, abs=1e-9)
    assert inertia.iyy == pytest.approx(5245.5, abs=1e-9)
    assert inertia.izz == pytest.approx(8067.3, abs=1e-9)
    assert inertia.ixy == pytest.approx(-135.9, abs=1e-9)
    assert inertia.ixz == pytest.approx(89.3, abs=1e-9)
    assert inertia.iyz == pytest.approx(-90.7, abs=1e-9)
    assert inertia.comments == ["Spacecraft Inertia Parameters"]

    # Maneuver block
    assert len(apm.maneuvers) == 1
    man = apm.maneuvers[0]
    assert man.duration == pytest.approx(3.0, abs=1e-9)
    assert man.ref_frame == "ICRF"
    assert man.torque[0] == pytest.approx(-1.25, abs=1e-9)
    assert man.torque[1] == pytest.approx(-0.5, abs=1e-9)
    assert man.torque[2] == pytest.approx(0.5, abs=1e-9)
    assert man.delta_mass is None
    assert man.comments == [
        "Data follows for 1 planned maneuver.",
        "First attitude maneuver for: MARS SPIRIT",
        "Impulsive, torque direction fixed in body frame",
    ]

    assert len(apm.euler_states) == 0
    assert len(apm.angular_velocities) == 0
    assert len(apm.spins) == 0


def test_apm_v1_version_rejected(eop):
    """Mirror of test_parse_apm_v1_version_rejected in Rust."""
    with pytest.raises(Exception, match="version 1.0"):
        APM.from_file("test_assets/ccsds/apm/APM-v1-version.txt")


def test_apm_no_blocks_rejected(eop):
    """Mirror of test_parse_apm_no_blocks_rejected in Rust."""
    with pytest.raises(Exception, match="at least one logical block"):
        APM.from_file("test_assets/ccsds/apm/APM-no-blocks.txt")


def test_apm_bad_euler_seq_rejected(eop):
    """Mirror of test_parse_apm_bad_euler_seq_rejected in Rust."""
    with pytest.raises(Exception, match="invalid EULER_ROT_SEQ value"):
        APM.from_file("test_assets/ccsds/apm/APM-bad-euler-seq.txt")


def test_apm_missing_ref_frame_rejected(eop):
    """Mirror of test_parse_apm_missing_ref_frame_rejected in Rust."""
    with pytest.raises(Exception, match="REF_FRAME_A"):
        APM.from_file("test_assets/ccsds/apm/APM-missing-ref-frame.txt")


def _assert_apm_fields_match(a: APM, b: APM):
    """Compares every value field of two APM messages, including all
    comment vectors (header, metadata, data, and per-block), mirroring
    assert_apm_fields_match in Rust.
    """
    assert a.format_version == pytest.approx(b.format_version, abs=1e-9)
    assert a.originator == b.originator
    assert a.classification == b.classification
    assert a.message_id == b.message_id

    assert a.object_name == b.object_name
    assert a.object_id == b.object_id
    assert a.center_name == b.center_name
    assert a.time_system == b.time_system
    assert a.metadata_comments == b.metadata_comments

    assert abs(a.epoch - b.epoch) < 1e-6

    da, db = a.to_dict(), b.to_dict()
    assert da["header"]["comments"] == db["header"]["comments"]
    assert da["comments"] == db["comments"]

    assert len(a.quaternion_states) == len(b.quaternion_states)
    for qa, qb, dqa, dqb in zip(
        a.quaternion_states,
        b.quaternion_states,
        da["quaternion_states"],
        db["quaternion_states"],
    ):
        assert qa.ref_frame_a == qb.ref_frame_a
        assert qa.ref_frame_b == qb.ref_frame_b
        assert dqa["comments"] == dqb["comments"]
        va = qa.quaternion.to_vector(scalar_first=False)
        vb = qb.quaternion.to_vector(scalar_first=False)
        assert va == pytest.approx(vb, abs=1e-9)
        assert (qa.quaternion_derivative is None) == (qb.quaternion_derivative is None)
        if qa.quaternion_derivative is not None:
            assert qa.quaternion_derivative == pytest.approx(
                qb.quaternion_derivative, abs=1e-9
            )

    assert len(a.euler_states) == len(b.euler_states)
    for ea, eb, dea, deb in zip(
        a.euler_states, b.euler_states, da["euler_states"], db["euler_states"]
    ):
        assert ea.ref_frame_a == eb.ref_frame_a
        assert ea.ref_frame_b == eb.ref_frame_b
        assert dea["comments"] == deb["comments"]
        assert ea.angles.order == eb.angles.order
        assert ea.angles.phi == pytest.approx(eb.angles.phi, abs=1e-9)
        assert ea.angles.theta == pytest.approx(eb.angles.theta, abs=1e-9)
        assert ea.angles.psi == pytest.approx(eb.angles.psi, abs=1e-9)
        assert (ea.rates is None) == (eb.rates is None)
        if ea.rates is not None:
            assert ea.rates == pytest.approx(eb.rates, abs=1e-9)

    assert len(a.angular_velocities) == len(b.angular_velocities)
    for va, vb, dva, dvb in zip(
        a.angular_velocities,
        b.angular_velocities,
        da["angular_velocities"],
        db["angular_velocities"],
    ):
        assert va.ref_frame_a == vb.ref_frame_a
        assert va.ref_frame_b == vb.ref_frame_b
        assert va.angvel_frame == vb.angvel_frame
        assert dva["comments"] == dvb["comments"]
        assert va.angular_velocity == pytest.approx(vb.angular_velocity, abs=1e-9)

    assert len(a.spins) == len(b.spins)
    for sa, sb, dsa, dsb in zip(a.spins, b.spins, da["spins"], db["spins"]):
        assert sa.ref_frame_a == sb.ref_frame_a
        assert sa.ref_frame_b == sb.ref_frame_b
        assert dsa["comments"] == dsb["comments"]
        assert sa.spin_alpha == pytest.approx(sb.spin_alpha, abs=1e-9)
        assert sa.spin_delta == pytest.approx(sb.spin_delta, abs=1e-9)
        assert sa.spin_angle == pytest.approx(sb.spin_angle, abs=1e-9)
        assert sa.spin_angle_vel == pytest.approx(sb.spin_angle_vel, abs=1e-9)
        assert sa.nutation_type == sb.nutation_type
        if sa.nutation_type == "ANGLE":
            assert sa.nutation == pytest.approx(sb.nutation, abs=1e-9)
            assert sa.nutation_period == pytest.approx(sb.nutation_period, abs=1e-9)
            assert sa.nutation_phase == pytest.approx(sb.nutation_phase, abs=1e-9)
        elif sa.nutation_type == "MOMENTUM":
            assert sa.momentum_alpha == pytest.approx(sb.momentum_alpha, abs=1e-9)
            assert sa.momentum_delta == pytest.approx(sb.momentum_delta, abs=1e-9)
            assert sa.nutation_vel == pytest.approx(sb.nutation_vel, abs=1e-9)

    assert len(a.inertias) == len(b.inertias)
    for ia, ib, dia, dib in zip(a.inertias, b.inertias, da["inertias"], db["inertias"]):
        assert ia.inertia_ref_frame == ib.inertia_ref_frame
        assert dia["comments"] == dib["comments"]
        assert ia.ixx == pytest.approx(ib.ixx, abs=1e-6)
        assert ia.iyy == pytest.approx(ib.iyy, abs=1e-6)
        assert ia.izz == pytest.approx(ib.izz, abs=1e-6)
        assert ia.ixy == pytest.approx(ib.ixy, abs=1e-6)
        assert ia.ixz == pytest.approx(ib.ixz, abs=1e-6)
        assert ia.iyz == pytest.approx(ib.iyz, abs=1e-6)

    assert len(a.maneuvers) == len(b.maneuvers)
    for ma, mb, dma, dmb in zip(
        a.maneuvers, b.maneuvers, da["maneuvers"], db["maneuvers"]
    ):
        assert abs(ma.epoch_start - mb.epoch_start) < 1e-6
        assert ma.duration == pytest.approx(mb.duration, abs=1e-9)
        assert ma.ref_frame == mb.ref_frame
        assert ma.torque == pytest.approx(mb.torque, abs=1e-9)
        assert ma.delta_mass == mb.delta_mass
        assert dma["comments"] == dmb["comments"]


@pytest.mark.parametrize(
    "fixture", ["APMExampleG1.txt", "APMExampleG2.txt", "APMExampleG3.txt"]
)
@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_apm_round_trip(eop, fixture, fmt):
    """Mirror of the test_apm_g{1,2,3}_{kvn,xml,json}_round_trip Rust tests."""
    apm1 = APM.from_file(f"test_assets/ccsds/apm/{fixture}")
    content = apm1.to_string(fmt)
    apm2 = APM.from_str(content)
    _assert_apm_fields_match(apm1, apm2)


def test_apm_from_str_detects_format(eop):
    """Mirror of test_apm_from_str_detects_format in Rust."""
    apm1 = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")

    kvn = apm1.to_string("KVN")
    xml = apm1.to_string("XML")
    json_str = apm1.to_string("JSON")

    assert APM.from_str(kvn).object_name == apm1.object_name
    assert APM.from_str(xml).object_name == apm1.object_name
    assert APM.from_str(json_str).object_name == apm1.object_name


def test_apm_from_file_g1(eop):
    """Mirror of test_apm_from_file_g1 in Rust."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")
    assert apm.object_name == "TRMM"


def test_apm_from_file_nonexistent(eop):
    """Mirror of test_apm_from_file_nonexistent in Rust."""
    with pytest.raises(Exception, match="Failed to read APM file"):
        APM.from_file("nonexistent_file.txt")


def test_apm_to_string_no_blocks_rejected_all_formats(eop):
    """Mirror of test_apm_to_string_no_blocks_rejected_all_formats in Rust."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    with pytest.raises(Exception, match="at least one logical block"):
        apm.to_string("KVN")
    with pytest.raises(Exception, match="at least one logical block"):
        apm.to_string("XML")
    with pytest.raises(Exception, match="at least one logical block"):
        apm.to_string("JSON")
    with pytest.raises(Exception, match="at least one logical block"):
        apm.to_json_string(uppercase_keys=True)


def test_apm_json_round_trip_key_cases(eop):
    """Mirror of test_apm_json_round_trip_key_cases in Rust."""
    apm1 = APM.from_file("test_assets/ccsds/apm/APMExampleG3.txt")

    json_lower = apm1.to_json_string(uppercase_keys=False)
    assert '"object_name"' in json_lower
    assert '"OBJECT_NAME"' not in json_lower
    apm_lower = APM.from_str(json_lower)
    _assert_apm_fields_match(apm1, apm_lower)

    json_upper = apm1.to_json_string(uppercase_keys=True)
    assert '"OBJECT_NAME"' in json_upper
    assert '"object_name"' not in json_upper
    apm_upper = APM.from_str(json_upper)
    _assert_apm_fields_match(apm1, apm_upper)


def test_apm_to_dict(eop):
    """Test to_dict() serialization."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")
    d = apm.to_dict()

    assert d["header"]["originator"] == "GSFC"
    assert d["metadata"]["object_name"] == "TRMM"
    assert d["metadata"]["center_name"] == "EARTH"
    assert len(d["quaternion_states"]) == 1
    assert d["quaternion_states"][0]["ref_frame_a"] == "SC_BODY_1"


def test_apm_metadata_comments(eop):
    """G-1's METADATA-section comments are reachable via the
    metadata_comments property and to_dict(), mirroring the field
    asserted directly on apm.metadata.comments in the Rust
    test_parse_apm_example_g1_quaternion test."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")

    expected = [
        "GEOCENTRIC, CARTESIAN, EARTH FIXED",
        "OBJECT_ID: 1997-074A",
        "$ITIM = 1997 NOV 21 22:26:18.40000000, $ original launch time",
    ]
    assert apm.metadata_comments == expected
    assert apm.to_dict()["metadata"]["comments"] == expected

    apm.metadata_comments = ["NEW COMMENT"]
    assert apm.metadata_comments == ["NEW COMMENT"]
    assert apm.to_dict()["metadata"]["comments"] == ["NEW COMMENT"]


# ------------------------------------------------------------------
# Builder construction (programmatic APM assembly)
# ------------------------------------------------------------------


def test_apm_builder_quaternion_state(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch, center_name="EARTH")
    assert not apm.has_blocks

    state = APMQuaternionState(
        "ICRF",
        "SC_BODY_1",
        Quaternion(1.0, 0.0, 0.0, 0.0),
        quaternion_derivative=np.array([0.1, 0.0, 0.0, 0.0]),
    )
    idx = apm.add_quaternion_state(state)
    assert idx == 0
    assert apm.has_blocks
    assert len(apm.quaternion_states) == 1
    assert apm.quaternion_states[0].quaternion_derivative == pytest.approx(
        [0.1, 0.0, 0.0, 0.0]
    )

    written = apm.to_string("KVN")
    apm2 = APM.from_str(written)
    assert len(apm2.quaternion_states) == 1
    assert apm2.quaternion_states[0].quaternion_derivative == pytest.approx(
        [0.1, 0.0, 0.0, 0.0], abs=1e-9
    )


def test_apm_builder_euler_state(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    angles = EulerAngle(EulerAngleOrder.ZXZ, 10.0, 20.0, 30.0, AngleFormat.DEGREES)
    state = APMEulerState(
        "ICRF", "SC_BODY_1", angles, rates=np.array([0.01, 0.02, 0.03])
    )
    apm.add_euler_state(state)

    assert len(apm.euler_states) == 1
    e = apm.euler_states[0]
    assert e.angles.order == EulerAngleOrder.ZXZ
    assert e.rates == pytest.approx([0.01, 0.02, 0.03])

    written = apm.to_string("KVN")
    apm2 = APM.from_str(written)
    e2 = apm2.euler_states[0]
    assert e2.angles.order == EulerAngleOrder.ZXZ
    assert e2.angles.phi == pytest.approx(math.radians(10.0), abs=1e-9)
    assert e2.angles.theta == pytest.approx(math.radians(20.0), abs=1e-9)
    assert e2.angles.psi == pytest.approx(math.radians(30.0), abs=1e-9)
    assert e2.rates == pytest.approx([0.01, 0.02, 0.03], abs=1e-9)


def test_apm_builder_angular_velocity(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    block = APMAngularVelocity(
        "ICRF", "SC_BODY_1", "SC_BODY_1", np.array([0.001, 0.0, 0.0])
    )
    apm.add_angular_velocity(block)

    assert len(apm.angular_velocities) == 1
    assert apm.angular_velocities[0].angular_velocity == pytest.approx(
        [0.001, 0.0, 0.0]
    )


def test_apm_builder_spin_nutation_angle(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    spin = APMSpin("ICRF", "SC_BODY_1", 10.0, 20.0, 30.0, 1.0, AngleFormat.DEGREES)
    assert spin.nutation_type == "NONE"
    spin.set_nutation_angle(5.0, 100.0, 15.0, AngleFormat.DEGREES)
    assert spin.nutation_type == "ANGLE"
    assert spin.nutation == pytest.approx(math.radians(5.0))
    assert spin.nutation_period == pytest.approx(100.0)
    assert spin.nutation_phase == pytest.approx(math.radians(15.0))

    apm.add_spin(spin)
    assert len(apm.spins) == 1

    written = apm.to_string("KVN")
    apm2 = APM.from_str(written)
    s2 = apm2.spins[0]
    assert s2.nutation_type == "ANGLE"
    assert s2.spin_alpha == pytest.approx(math.radians(10.0), abs=1e-9)
    assert s2.spin_delta == pytest.approx(math.radians(20.0), abs=1e-9)
    assert s2.spin_angle == pytest.approx(math.radians(30.0), abs=1e-9)
    assert s2.spin_angle_vel == pytest.approx(math.radians(1.0), abs=1e-9)
    assert s2.nutation == pytest.approx(math.radians(5.0), abs=1e-9)
    assert s2.nutation_period == pytest.approx(100.0, abs=1e-9)
    assert s2.nutation_phase == pytest.approx(math.radians(15.0), abs=1e-9)

    # Owned-copy semantics: a spin fetched from apm.spins is an independent
    # copy — mutating it does not affect the parent APM.
    fetched = apm2.spins[0]
    fetched.set_nutation_momentum(1.0, 2.0, 3.0, AngleFormat.DEGREES)
    assert fetched.nutation_type == "MOMENTUM"
    assert apm2.spins[0].nutation_type == "ANGLE"


def test_apm_builder_spin_nutation_momentum(eop):
    spin = APMSpin("ICRF", "SC_BODY_1", 10.0, 20.0, 30.0, 1.0, AngleFormat.DEGREES)
    spin.set_nutation_momentum(7.0, 8.0, 0.5, AngleFormat.DEGREES)
    assert spin.nutation_type == "MOMENTUM"
    assert spin.momentum_alpha == pytest.approx(math.radians(7.0))
    assert spin.momentum_delta == pytest.approx(math.radians(8.0))
    assert spin.nutation_vel == pytest.approx(math.radians(0.5))


def test_apm_builder_inertia(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    inertia = APMInertia("SC_BODY_1", 6080.0, 5245.5, 8067.3, -135.9, 89.3, -90.7)
    matrix = inertia.inertia_matrix
    assert matrix[0, 0] == pytest.approx(6080.0)
    assert matrix[0, 1] == pytest.approx(135.9)
    assert matrix[0, 2] == pytest.approx(-89.3)
    assert matrix[1, 2] == pytest.approx(90.7)
    assert matrix[0, 1] == matrix[1, 0]

    apm.add_inertia(inertia)
    assert len(apm.inertias) == 1


def test_apm_builder_maneuver(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    man = APMManeuver(epoch, 3.0, "ICRF", np.array([-1.25, -0.5, 0.5]), delta_mass=-0.5)
    apm.add_maneuver(man)

    assert len(apm.maneuvers) == 1
    assert apm.maneuvers[0].delta_mass == pytest.approx(-0.5)
    assert apm.maneuvers[0].torque == pytest.approx([-1.25, -0.5, 0.5])

    written = apm.to_string("KVN")
    apm2 = APM.from_str(written)
    assert apm2.maneuvers[0].delta_mass == pytest.approx(-0.5)


def test_apm_maneuver_positive_delta_mass_rejected(eop):
    # CCSDS 504.0-B-2 table 3-3 requires MAN_DELTA_MASS <= 0.
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    with pytest.raises(Exception, match="MAN_DELTA_MASS"):
        APMManeuver(epoch, 3.0, "ICRF", np.array([-1.25, -0.5, 0.5]), delta_mass=0.5)


def test_apm_header_metadata_setters(eop):
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)

    apm.format_version = 2.0
    assert apm.format_version == 2.0

    apm.originator = "NASA"
    assert apm.originator == "NASA"

    apm.classification = "UNCLASSIFIED"
    assert apm.classification == "UNCLASSIFIED"
    apm.classification = None
    assert apm.classification is None

    apm.message_id = "MSG-001"
    assert apm.message_id == "MSG-001"

    apm.object_name = "SAT2"
    assert apm.object_name == "SAT2"

    apm.object_id = "2024-002B"
    assert apm.object_id == "2024-002B"

    apm.center_name = "MOON"
    assert apm.center_name == "MOON"

    apm.time_system = "TAI"
    assert apm.time_system == "TAI"

    new_epoch = Epoch.from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm.epoch = new_epoch
    assert abs(apm.epoch - new_epoch) < 1e-9

    new_creation_date = Epoch.from_datetime(
        2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
    )
    apm.creation_date = new_creation_date
    assert abs(apm.creation_date - new_creation_date) < 1e-9


def test_apm_epoch_written_in_metadata_time_system(eop):
    """Mirror of test_apm_epoch_written_in_metadata_time_system in Rust:
    EPOCH must be written in the metadata TIME_SYSTEM, not the Epoch's own
    internal time system."""
    epoch_utc = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "TAI", epoch_utc)
    apm.add_quaternion_state(
        APMQuaternionState("ICRF", "SC_BODY_1", Quaternion(1.0, 0.0, 0.0, 0.0))
    )

    kvn = apm.to_string("KVN")

    epoch_tai = epoch_utc.to_time_system(bh.TimeSystem.TAI)
    tai_clock = str(epoch_tai).split(" ")[1]
    utc_clock = str(epoch_utc).split(" ")[1]
    assert tai_clock != utc_clock
    assert tai_clock in kvn
    assert f"EPOCH = 2024-03-01T{tai_clock}" in kvn

    apm2 = APM.from_str(kvn)
    assert abs(apm2.epoch - epoch_utc) < 1e-9


def test_apm_to_dict_epoch_written_in_metadata_time_system(eop):
    """to_dict()["epoch"] must match the metadata TIME_SYSTEM, the same as
    the KVN writer, not the Epoch's own internal time system."""
    epoch_utc = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "TAI", epoch_utc)
    apm.add_quaternion_state(
        APMQuaternionState("ICRF", "SC_BODY_1", Quaternion(1.0, 0.0, 0.0, 0.0))
    )

    kvn = apm.to_string("KVN")
    epoch_line = next(line for line in kvn.splitlines() if line.startswith("EPOCH"))
    kvn_epoch_str = epoch_line.split(" = ", 1)[1]

    assert apm.to_dict()["epoch"] == kvn_epoch_str


def test_apm_to_dict_creation_date_written_in_utc(eop):
    """Mirror of test_apm_creation_date_written_in_utc in Rust: CREATION_DATE
    is a UTC field, so to_dict() must convert a header epoch held in another
    time system rather than relabel it, matching the KVN writer."""
    epoch_utc = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "TAI", epoch_utc)
    apm.creation_date = Epoch.from_datetime(
        2024, 3, 1, 12, 0, 37.0, 0.0, bh.TimeSystem.TAI
    )
    apm.add_quaternion_state(
        APMQuaternionState("ICRF", "SC_BODY_1", Quaternion(1.0, 0.0, 0.0, 0.0))
    )

    kvn = apm.to_string("KVN")
    creation_line = next(
        line for line in kvn.splitlines() if line.startswith("CREATION_DATE")
    )
    kvn_creation_str = creation_line.split(" = ", 1)[1]

    # 2024-03-01T12:00:37 TAI is 2024-03-01T12:00:00 UTC.
    assert kvn_creation_str == "2024-03-01T12:00:00.000"
    assert apm.to_dict()["header"]["creation_date"] == kvn_creation_str


# ------------------------------------------------------------------
# Additional KVN block parsing (mirrors synthetic-content Rust tests in
# src/ccsds/kvn/parser.rs)
# ------------------------------------------------------------------

_APM_PREFIX = (
    "CCSDS_APM_VERS = 2.0\n"
    "CREATION_DATE = 2003-09-30T19:23:57\n"
    "ORIGINATOR = BRAHE\n"
    "OBJECT_NAME = TESTSAT\n"
    "OBJECT_ID = 2024-001A\n"
    "CENTER_NAME = EARTH\n"
    "TIME_SYSTEM = UTC\n"
    "EPOCH = 2003-09-30T14:28:15.1172\n"
)


def test_apm_parse_angular_velocity_block(eop):
    """Mirror of test_parse_apm_angular_velocity_block in Rust."""
    content = _APM_PREFIX + (
        "ANGVEL_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "ANGVEL_FRAME = SC_BODY_1\n"
        "ANGVEL_X = 1.0\n"
        "ANGVEL_Y = 2.0\n"
        "ANGVEL_Z = 3.0\n"
        "ANGVEL_STOP\n"
    )
    apm = APM.from_str(content)
    assert len(apm.angular_velocities) == 1
    av = apm.angular_velocities[0]
    assert av.angvel_frame == "SC_BODY_1"
    assert av.angular_velocity == pytest.approx(
        [math.radians(1.0), math.radians(2.0), math.radians(3.0)], abs=1e-10
    )


def test_apm_parse_spin_nutation_angle_triple(eop):
    """Mirror of test_parse_apm_spin_nutation_angle_triple in Rust."""
    content = _APM_PREFIX + (
        "SPIN_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "SPIN_ALPHA = 10.0\n"
        "SPIN_DELTA = 20.0\n"
        "SPIN_ANGLE = 30.0\n"
        "SPIN_ANGLE_VEL = 1.0\n"
        "NUTATION = 5.0\n"
        "NUTATION_PER = 100.0\n"
        "NUTATION_PHASE = 15.0\n"
        "SPIN_STOP\n"
    )
    apm = APM.from_str(content)
    assert len(apm.spins) == 1
    s = apm.spins[0]
    assert s.nutation_type == "ANGLE"
    assert s.nutation == pytest.approx(math.radians(5.0), abs=1e-10)
    assert s.nutation_period == pytest.approx(100.0, abs=1e-10)
    assert s.nutation_phase == pytest.approx(math.radians(15.0), abs=1e-10)


def test_apm_parse_spin_momentum_triple(eop):
    """Mirror of test_parse_apm_spin_momentum_triple in Rust."""
    content = _APM_PREFIX + (
        "SPIN_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "SPIN_ALPHA = 10.0\n"
        "SPIN_DELTA = 20.0\n"
        "SPIN_ANGLE = 30.0\n"
        "SPIN_ANGLE_VEL = 1.0\n"
        "MOMENTUM_ALPHA = 7.0\n"
        "MOMENTUM_DELTA = 8.0\n"
        "NUTATION_VEL = 0.5\n"
        "SPIN_STOP\n"
    )
    apm = APM.from_str(content)
    assert len(apm.spins) == 1
    s = apm.spins[0]
    assert s.nutation_type == "MOMENTUM"
    assert s.momentum_alpha == pytest.approx(math.radians(7.0), abs=1e-10)
    assert s.momentum_delta == pytest.approx(math.radians(8.0), abs=1e-10)
    assert s.nutation_vel == pytest.approx(math.radians(0.5), abs=1e-10)


def test_apm_parse_spin_both_triples_rejected(eop):
    """Mirror of test_parse_apm_spin_both_triples_rejected in Rust."""
    content = _APM_PREFIX + (
        "SPIN_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "SPIN_ALPHA = 10.0\n"
        "SPIN_DELTA = 20.0\n"
        "SPIN_ANGLE = 30.0\n"
        "SPIN_ANGLE_VEL = 1.0\n"
        "NUTATION = 5.0\n"
        "NUTATION_PER = 100.0\n"
        "NUTATION_PHASE = 15.0\n"
        "MOMENTUM_ALPHA = 7.0\n"
        "MOMENTUM_DELTA = 8.0\n"
        "NUTATION_VEL = 0.5\n"
        "SPIN_STOP\n"
    )
    with pytest.raises(Exception, match="cannot contain both"):
        APM.from_str(content)


def test_apm_parse_spin_partial_triple_rejected(eop):
    """Mirror of test_parse_apm_spin_partial_triple_rejected in Rust."""
    content = _APM_PREFIX + (
        "SPIN_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "SPIN_ALPHA = 10.0\n"
        "SPIN_DELTA = 20.0\n"
        "SPIN_ANGLE = 30.0\n"
        "SPIN_ANGLE_VEL = 1.0\n"
        "NUTATION = 5.0\n"
        "SPIN_STOP\n"
    )
    with pytest.raises(Exception, match="incomplete spin nutation triple"):
        APM.from_str(content)


def test_apm_parse_quaternion_derivative_round_trip(eop):
    """Mirror of test_parse_apm_quaternion_derivative_round_trip in Rust."""
    content = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
        "QC = 1.0\n"
        "Q1_DOT = 0.1\n"
        "Q2_DOT = 0.2\n"
        "Q3_DOT = 0.3\n"
        "QC_DOT = 0.4\n"
        "QUAT_STOP\n"
    )
    apm = APM.from_str(content)
    d = apm.quaternion_states[0].quaternion_derivative
    # Stored scalar-first: [QC_DOT, Q1_DOT, Q2_DOT, Q3_DOT]
    assert d == pytest.approx([0.4, 0.1, 0.2, 0.3], abs=1e-10)


def test_apm_parse_quaternion_partial_derivative_rejected(eop):
    """Mirror of test_parse_apm_quaternion_partial_derivative_rejected in Rust."""
    content = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
        "QC = 1.0\n"
        "Q1_DOT = 0.1\n"
        "QUAT_STOP\n"
    )
    with pytest.raises(Exception, match="incomplete quaternion derivative"):
        APM.from_str(content)


def test_apm_parse_user_defined_rejected(eop):
    """Mirror of test_parse_apm_user_defined_rejected in Rust.

    USER_DEFINED_* is not part of APM (504.0-B-2 restricts APM's data
    section to the six logical blocks in table 3-1; USER_DEFINED_* is
    ODM/ACM-only per section 3.2.4.2), so it must be rejected like any
    other unrecognized keyword.
    """
    content = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
        "QC = 1.0\n"
        "QUAT_STOP\n"
        "USER_DEFINED_BATTERY_STATE = NOMINAL\n"
    )
    with pytest.raises(Exception, match="not part of APM"):
        APM.from_str(content)


def test_apm_parse_unknown_block_keyword_rejected(eop):
    """Mirror of test_parse_apm_unknown_block_keyword_rejected in Rust."""
    content = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "BOGUS_KEY = 1.0\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
        "QC = 1.0\n"
        "QUAT_STOP\n"
    )
    with pytest.raises(Exception, match="unexpected keyword 'BOGUS_KEY'"):
        APM.from_str(content)


def test_apm_parse_unterminated_block_rejected(eop):
    """Mirror of test_parse_apm_unterminated_block_rejected in Rust: a
    QUAT_START block with no QUAT_STOP must error at EOF, naming the
    unterminated block."""
    content = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
    )
    with pytest.raises(Exception, match="unterminated"):
        APM.from_str(content)


def test_apm_parse_epoch_trailing_z(eop):
    """Mirror of test_parse_ccsds_datetime_trailing_z in Rust, exercised via
    APM.from_str since parse_ccsds_datetime is not directly exposed to
    Python."""
    content_no_z = _APM_PREFIX + (
        "QUAT_START\n"
        "REF_FRAME_A = ICRF\n"
        "REF_FRAME_B = SC_BODY_1\n"
        "Q1 = 0.0\n"
        "Q2 = 0.0\n"
        "Q3 = 0.0\n"
        "QC = 1.0\n"
        "QUAT_STOP\n"
    )
    content_z = content_no_z.replace(
        "EPOCH = 2003-09-30T14:28:15.1172\n", "EPOCH = 2003-09-30T14:28:15.1172Z\n"
    )
    assert content_z != content_no_z

    apm_no_z = APM.from_str(content_no_z)
    apm_z = APM.from_str(content_z)
    assert abs(apm_no_z.epoch - apm_z.epoch) < 1e-9


# ------------------------------------------------------------------
# XML-specific parsing (mirrors src/ccsds/xml/parser.rs)
# ------------------------------------------------------------------


def test_apm_parse_xml_example_g10(eop):
    """Mirror of test_parse_apm_xml_example_g10 in Rust."""
    apm = APM.from_file("test_assets/ccsds/apm/APMExampleG10.xml")

    assert apm.format_version == pytest.approx(2.0, abs=1e-10)
    assert apm.originator == "GSFC"
    assert apm.message_id == "A7015Z1"
    assert apm.classification is None

    assert apm.object_name == "TRMM"
    assert apm.object_id == "1997-074A"
    assert apm.center_name == "EARTH"
    assert apm.time_system == "UTC"

    assert len(apm.quaternion_states) == 1
    q = apm.quaternion_states[0]
    assert q.ref_frame_a == "SC_BODY_1"
    assert q.ref_frame_b == "ITRF1997"
    assert q.comments == ["Attitude state vector quaternion"]
    v = q.quaternion.to_vector(scalar_first=False)
    assert v[0] == pytest.approx(0.00005, abs=1e-4)
    assert v[1] == pytest.approx(0.87543, abs=1e-4)
    assert v[2] == pytest.approx(0.40949, abs=1e-4)
    assert v[3] == pytest.approx(0.25678, abs=1e-4)
    assert q.quaternion_derivative is None

    assert len(apm.euler_states) == 0
    assert len(apm.angular_velocities) == 0
    assert len(apm.spins) == 0
    assert len(apm.inertias) == 0
    assert len(apm.maneuvers) == 0


def test_apm_parse_xml_quaternion_derivative(eop):
    """Mirror of test_parse_apm_xml_quaternion_derivative in Rust."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<apm id="CCSDS_APM_VERS" version="2.0">
   <header>
      <CREATION_DATE>2003-09-30T19:23:57</CREATION_DATE>
      <ORIGINATOR>GSFC</ORIGINATOR>
   </header>
   <body>
      <segment>
         <metadata>
            <OBJECT_NAME>TRMM</OBJECT_NAME>
            <OBJECT_ID>1997-074A</OBJECT_ID>
            <TIME_SYSTEM>UTC</TIME_SYSTEM>
         </metadata>
         <data>
            <EPOCH>2003-09-30T14:28:15.1172</EPOCH>
            <quaternionState>
               <REF_FRAME_A>ICRF</REF_FRAME_A>
               <REF_FRAME_B>SC_BODY_1</REF_FRAME_B>
               <quaternion>
                  <Q1>0.0</Q1>
                  <Q2>0.0</Q2>
                  <Q3>0.0</Q3>
                  <QC>1.0</QC>
               </quaternion>
               <quaternionDot>
                  <Q1_DOT>0.1</Q1_DOT>
                  <Q2_DOT>0.2</Q2_DOT>
                  <Q3_DOT>0.3</Q3_DOT>
                  <QC_DOT>0.4</QC_DOT>
               </quaternionDot>
            </quaternionState>
         </data>
      </segment>
   </body>
</apm>
"""
    apm = APM.from_str(content)
    d = apm.quaternion_states[0].quaternion_derivative
    assert d == pytest.approx([0.4, 0.1, 0.2, 0.3], abs=1e-10)

    written = apm.to_string("XML")
    assert "<quaternionDot>" in written
    apm2 = APM.from_str(written)
    d2 = apm2.quaternion_states[0].quaternion_derivative
    assert d2 == pytest.approx([0.4, 0.1, 0.2, 0.3], abs=1e-10)


def test_apm_parse_xml_v1_version_rejected(eop):
    """Mirror of test_parse_apm_xml_v1_version_rejected in Rust."""
    with open("test_assets/ccsds/apm/APMExampleG10.xml") as f:
        content = f.read().replace('version="2.0"', 'version="1.0"')
    with pytest.raises(Exception, match="version 1.0"):
        APM.from_str(content)


def _apm_for_xml_test():
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)
    apm.add_quaternion_state(
        APMQuaternionState("ICRF", "SC_BODY_1", Quaternion(1.0, 0.0, 0.0, 0.0))
    )
    return apm


def test_apm_write_xml_root_has_xmlns_xsi(eop):
    """Mirror of test_write_apm_xml_root_has_xmlns_xsi in Rust."""
    apm = _apm_for_xml_test()
    xml = apm.to_string("XML")
    assert (
        '<apm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'id="CCSDS_APM_VERS" version="2.0">' in xml
    )

    reparsed = APM.from_str(xml)
    assert reparsed.object_name == "SAT1"


def test_apm_write_xml_escapes_free_text(eop):
    """Mirror of test_write_apm_xml_escapes_free_text in Rust."""
    apm = _apm_for_xml_test()
    apm.originator = "A & B <test>"

    xml = apm.to_string("XML")
    assert "<ORIGINATOR>A &amp; B &lt;test&gt;</ORIGINATOR>" in xml
    assert "<ORIGINATOR>A & B <test></ORIGINATOR>" not in xml

    reparsed = APM.from_str(xml)
    assert reparsed.originator == "A & B <test>"


def _build_apm_all_blocks():
    """Mirror of apm_all_blocks() in src/ccsds/apm.rs: one of every logical
    block type, with the optional sub-fields of each block populated."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch, center_name="EARTH")

    apm.add_quaternion_state(
        APMQuaternionState(
            "ICRF",
            "SC_BODY_1",
            Quaternion(0.5, 0.5, 0.5, 0.5),
            quaternion_derivative=np.array([0.01, 0.02, 0.03, 0.04]),
        )
    )

    angles = EulerAngle(EulerAngleOrder.ZXZ, 10.0, 20.0, 30.0, AngleFormat.DEGREES)
    apm.add_euler_state(
        APMEulerState("ICRF", "SC_BODY_1", angles, rates=np.array([0.01, 0.02, 0.03]))
    )

    spin_angle = APMSpin(
        "ICRF", "SC_BODY_1", 10.0, 20.0, 30.0, 1.0, AngleFormat.DEGREES
    )
    spin_angle.set_nutation_angle(5.0, 100.0, 15.0, AngleFormat.DEGREES)
    apm.add_spin(spin_angle)

    spin_momentum = APMSpin(
        "ICRF", "SC_BODY_1", 40.0, 50.0, 60.0, 2.0, AngleFormat.DEGREES
    )
    spin_momentum.set_nutation_momentum(7.0, 8.0, 0.5, AngleFormat.DEGREES)
    apm.add_spin(spin_momentum)

    apm.add_angular_velocity(
        APMAngularVelocity(
            "ICRF", "SC_BODY_1", "SC_BODY_1", np.array([0.001, 0.002, 0.003])
        )
    )

    apm.add_inertia(
        APMInertia("SC_BODY_1", 6080.0, 5245.5, 8067.3, -135.9, 89.3, -90.7)
    )

    apm.add_maneuver(
        APMManeuver(epoch, 3.0, "ICRF", np.array([-1.25, -0.5, 0.5]), delta_mass=-0.25)
    )

    return apm


def _assert_apm_all_blocks_subset(apm):
    """Checks the representative subset of fields called out in the task:
    the quaternion derivative array, both spin nutation variants, and
    maneuver delta_mass."""
    assert len(apm.quaternion_states) == 1
    assert apm.quaternion_states[0].quaternion_derivative == pytest.approx(
        [0.01, 0.02, 0.03, 0.04], abs=1e-9
    )

    assert len(apm.euler_states) == 1
    assert apm.euler_states[0].rates == pytest.approx([0.01, 0.02, 0.03], abs=1e-9)

    assert len(apm.spins) == 2
    s0, s1 = apm.spins
    assert s0.nutation_type == "ANGLE"
    assert s0.nutation == pytest.approx(math.radians(5.0), abs=1e-9)
    assert s0.nutation_period == pytest.approx(100.0, abs=1e-9)
    assert s0.nutation_phase == pytest.approx(math.radians(15.0), abs=1e-9)
    assert s1.nutation_type == "MOMENTUM"
    assert s1.momentum_alpha == pytest.approx(math.radians(7.0), abs=1e-9)
    assert s1.momentum_delta == pytest.approx(math.radians(8.0), abs=1e-9)
    assert s1.nutation_vel == pytest.approx(math.radians(0.5), abs=1e-9)

    assert len(apm.angular_velocities) == 1
    assert len(apm.inertias) == 1
    assert len(apm.maneuvers) == 1
    assert apm.maneuvers[0].delta_mass == pytest.approx(-0.25, abs=1e-9)


@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_apm_all_blocks_round_trip(eop, fmt):
    """Mirror of the Rust test_apm_all_blocks_{kvn,xml}_round_trip and
    test_apm_all_blocks_json_round_trip_key_cases tests: round-trips an APM
    containing every logical block type, with all optional sub-fields set,
    through each wire format."""
    apm1 = _build_apm_all_blocks()
    content = apm1.to_string(fmt)
    apm2 = APM.from_str(content)
    _assert_apm_all_blocks_subset(apm2)


def test_apm_all_blocks_json_round_trip_key_cases(eop):
    """Mirror of test_apm_all_blocks_json_round_trip_key_cases in Rust."""
    apm1 = _build_apm_all_blocks()

    json_lower = apm1.to_json_string(uppercase_keys=False)
    apm_lower = APM.from_str(json_lower)
    _assert_apm_all_blocks_subset(apm_lower)

    json_upper = apm1.to_json_string(uppercase_keys=True)
    apm_upper = APM.from_str(json_upper)
    _assert_apm_all_blocks_subset(apm_upper)


@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_apm_simple_spin_round_trip(eop, fmt):
    """A spin block with no nutation (APMSpin's default) is never written
    and round-tripped elsewhere in this file — every other spin test sets
    nutation_type to ANGLE or MOMENTUM before serializing."""
    epoch = Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    apm = APM("BRAHE", "SAT1", "2024-001A", "UTC", epoch)
    apm.add_quaternion_state(
        APMQuaternionState("ICRF", "SC_BODY_1", Quaternion(1.0, 0.0, 0.0, 0.0))
    )
    apm.add_spin(
        APMSpin("ICRF", "SC_BODY_1", 10.0, 20.0, 30.0, 1.0, AngleFormat.DEGREES)
    )

    content = apm.to_string(fmt)
    apm2 = APM.from_str(content)
    assert apm2.spins[0].nutation_type == "NONE"
