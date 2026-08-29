"""Tests for CCSDS AEM parsing, mutation, construction, and AttitudeTrajectory interop —
parity with Rust tests."""

import math

import numpy as np
import pytest

import brahe as bh
from brahe.ccsds import AEM, AEMAttitudeState, AEMSegment
from brahe.trajectories import AttitudeTrajectory


def test_aem_parse_example_g4_quaternion(eop):
    """Mirror of test_parse_aem_example_g4 in Rust."""
    aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")

    assert aem.format_version == pytest.approx(2.0, abs=1e-10)
    assert aem.originator == "NASA/JPL"
    assert aem.message_id == "A7015Z3"
    assert len(aem.segments) == 2

    seg0 = aem.segments[0]
    assert seg0.object_name == "MARS GLOBAL SURVEYOR"
    assert seg0.object_id == "1996-062A"
    assert seg0.center_name == "MARS BARYCENTER"
    assert seg0.ref_frame_a == "EME2000"
    assert seg0.ref_frame_b == "SC_BODY_1"
    assert seg0.time_system == "UTC"
    assert seg0.attitude_type == "QUATERNION"
    assert seg0.interpolation_method == "HERMITE"
    assert seg0.interpolation_degree == 7
    assert seg0.useable_start_time is not None
    assert seg0.useable_stop_time is not None
    assert len(seg0.states) == 4

    st0 = seg0.states[0]
    assert st0.attitude_type == "QUATERNION"
    v = st0.quaternion.to_vector(scalar_first=False)
    assert v[0] == pytest.approx(0.56748, abs=1e-4)
    assert v[1] == pytest.approx(0.03146, abs=1e-4)
    assert v[2] == pytest.approx(0.45689, abs=1e-4)
    assert v[3] == pytest.approx(0.68427, abs=1e-4)
    assert st0.angular_velocity is None
    assert st0.euler_angles is None
    assert st0.spin_alpha is None

    seg1 = aem.segments[1]
    assert seg1.object_name == "mars global surveyor"
    assert seg1.interpolation_method is None
    assert seg1.interpolation_degree is None
    assert len(seg1.states) == 4


def test_aem_parse_example_g5_spin(eop):
    """Mirror of test_parse_aem_example_g5 in Rust."""
    aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG5.txt")
    assert aem.originator == "GSFC"
    assert len(aem.segments) == 1

    seg = aem.segments[0]
    assert seg.object_name == "ST5-224"
    assert seg.object_id == "2006-224A"
    assert seg.ref_frame_a == "J2000"
    assert seg.attitude_type == "SPIN"
    assert len(seg.states) == 8

    st0 = seg.states[0]
    assert st0.attitude_type == "SPIN"
    assert st0.spin_alpha == pytest.approx(math.radians(268.62511), abs=1e-4)
    assert st0.spin_delta == pytest.approx(math.radians(68.448486), abs=1e-4)
    assert st0.spin_angle == pytest.approx(math.radians(159.69509), abs=1e-4)
    assert st0.spin_angle_vel == pytest.approx(math.radians(-109.96528), abs=1e-4)
    assert st0.quaternion is None
    assert st0.nutation is None
    assert st0.momentum_alpha is None


@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_aem_round_trip_g4(eop, fmt):
    """Mirror of the AEM G-4 round-trip Rust tests."""
    aem1 = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")
    content = aem1.to_string(fmt)
    aem2 = AEM.from_str(content)

    assert aem2.originator == aem1.originator
    assert len(aem2.segments) == len(aem1.segments)
    for seg1, seg2 in zip(aem1.segments, aem2.segments):
        assert seg2.object_name == seg1.object_name
        assert seg2.attitude_type == seg1.attitude_type
        assert len(seg2.states) == len(seg1.states)
        for st1, st2 in zip(seg1.states, seg2.states):
            np.testing.assert_allclose(
                st2.quaternion.to_vector(scalar_first=False),
                st1.quaternion.to_vector(scalar_first=False),
                atol=1e-6,
            )


@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_aem_round_trip_g5(eop, fmt):
    aem1 = AEM.from_file("test_assets/ccsds/aem/AEMExampleG5.txt")
    content = aem1.to_string(fmt)
    aem2 = AEM.from_str(content)

    st1 = aem1.segments[0].states[0]
    st2 = aem2.segments[0].states[0]
    assert st2.spin_alpha == pytest.approx(st1.spin_alpha, abs=1e-6)
    assert st2.spin_angle_vel == pytest.approx(st1.spin_angle_vel, abs=1e-6)


def test_aem_from_str_detects_format(eop):
    aem1 = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")
    kvn = aem1.to_string("KVN")
    xml = aem1.to_string("XML")
    json_str = aem1.to_string("JSON")

    assert AEM.from_str(kvn).originator == aem1.originator
    assert AEM.from_str(xml).originator == aem1.originator
    assert AEM.from_str(json_str).originator == aem1.originator


def test_aem_from_file_nonexistent():
    with pytest.raises(Exception, match="Failed to read AEM file"):
        AEM.from_file("nonexistent_file.txt")


def test_aem_to_string_no_segments_rejected_all_formats():
    aem = AEM("BRAHE")
    with pytest.raises(Exception, match="at least one segment"):
        aem.to_string("KVN")
    with pytest.raises(Exception, match="at least one segment"):
        aem.to_string("XML")
    with pytest.raises(Exception, match="at least one segment"):
        aem.to_string("JSON")


def test_aem_to_string_empty_segment_rejected():
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
    )
    aem = AEM("BRAHE")
    aem.add_segment(seg)
    with pytest.raises(Exception, match="at least one attitude state"):
        aem.to_string("KVN")


def test_aem_json_round_trip_key_cases(eop):
    aem1 = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")
    json_lower = aem1.to_json_string(uppercase_keys=False)
    assert '"object_name"' in json_lower
    assert '"OBJECT_NAME"' not in json_lower
    aem2 = AEM.from_str(json_lower)
    assert aem2.originator == aem1.originator


def test_aem_to_dict():
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
    )
    seg.add_state(
        AEMAttitudeState.from_quaternion(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    )
    aem = AEM("BRAHE")
    aem.add_segment(seg)

    d = aem.to_dict()
    assert d["header"]["originator"] == "BRAHE"
    assert len(d["segments"]) == 1
    assert d["segments"][0]["object_name"] == "SAT1"
    assert len(d["segments"][0]["states"]) == 1


def test_aem_segment_builder_quaternion_angvel_round_trip():
    """Build an AEM in code with the QUATERNION/ANGVEL type and round-trip it."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "QUATERNION/ANGVEL",
        angvel_frame="SC_BODY_1",
    )
    omega = np.array([0.001, -0.002, 0.003])
    seg.add_state(
        AEMAttitudeState.from_quaternion_angvel(
            t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0), omega
        )
    )
    idx = seg.add_state(
        AEMAttitudeState.from_quaternion_angvel(
            t1, bh.Quaternion(0.9998, 0.0, 0.0, 0.0196), omega
        )
    )
    assert idx == 1
    assert len(seg) == 2
    assert seg.angvel_frame == "SC_BODY_1"

    aem = AEM("BRAHE")
    idx = aem.add_segment(seg)
    assert idx == 0

    content = aem.to_string("KVN")
    aem2 = AEM.from_str(content)
    st = aem2.segments[0].states[0]
    assert st.attitude_type == "QUATERNION/ANGVEL"
    np.testing.assert_allclose(st.angular_velocity, omega, atol=1e-6)


def test_aem_segment_add_state_wrong_type_errors():
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
    )
    with pytest.raises(Exception, match="does not match"):
        seg.add_state(AEMAttitudeState.from_spin(t0, 0.1, 0.2, 0.3, 0.4))


def test_aem_segment_add_state_non_increasing_epoch_errors():
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
    )
    seg.add_state(
        AEMAttitudeState.from_quaternion(t1, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    )
    with pytest.raises(Exception, match="strictly increasing"):
        seg.add_state(
            AEMAttitudeState.from_quaternion(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
        )


def test_aem_segment_add_state_euler_order_mismatch_errors():
    """Mirror of test_aem_segment_push_state_euler_order_mismatch_errors_naming_both_orders
    in Rust."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "EULER_ANGLE",
        euler_rot_seq=bh.EulerAngleOrder.ZXZ,
    )
    mismatched_angles = bh.EulerAngle(
        bh.EulerAngleOrder.XYZ, 10.0, 20.0, 30.0, bh.AngleFormat.DEGREES
    )
    with pytest.raises(Exception, match="XYZ"):
        seg.add_state(AEMAttitudeState.from_euler_angle(t0, mismatched_angles))


def test_aem_euler_angle_state_round_trip():
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    order = bh.EulerAngleOrder.ZXZ
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "EULER_ANGLE",
        euler_rot_seq=order,
    )
    assert seg.euler_rot_seq == order

    angles = bh.EulerAngle(order, 10.0, 20.0, 30.0, bh.AngleFormat.DEGREES)
    seg.add_state(AEMAttitudeState.from_euler_angle(t0, angles))
    seg.add_state(AEMAttitudeState.from_euler_angle(t1, angles))

    aem = AEM("BRAHE")
    aem.add_segment(seg)
    content = aem.to_string("KVN")
    aem2 = AEM.from_str(content)
    st = aem2.segments[0].states[0]
    assert st.attitude_type == "EULER_ANGLE"
    assert st.euler_angles.phi == pytest.approx(math.radians(10.0), abs=1e-6)
    assert st.rates is None


# ---------------------------------------------------------------------------
# AttitudeTrajectory interop
# ---------------------------------------------------------------------------


def test_aem_g4_segment_to_attitude_trajectory(eop):
    """Mirror of test_aem_g4_segment_to_attitude_trajectory in Rust."""
    aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")

    # Segment 1 (0-indexed) carries no INTERPOLATION_METHOD and defaults to
    # SLERP; segment 0 sets INTERPOLATION_METHOD = hermite (see the Hermite
    # error test below).
    traj = aem.segment_to_attitude_trajectory(1)

    assert len(traj) == 4
    assert traj.frame_a == bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    assert traj.frame_b == bh.AttitudeFrame.spacecraft("SC_BODY", "1")
    assert traj.interpolation_method == "SLERP"
    assert not traj.has_rates

    segment = aem.segments[1]
    q_first = segment.states[0].quaternion
    q_last = segment.states[-1].quaternion
    np.testing.assert_allclose(
        traj.quaternion(traj.start_epoch).to_vector(scalar_first=True),
        q_first.to_vector(scalar_first=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        traj.quaternion(traj.end_epoch).to_vector(scalar_first=True),
        q_last.to_vector(scalar_first=True),
        atol=1e-12,
    )


def test_aem_g5_spin_conversion_errors(eop):
    """Mirror of test_aem_g5_spin_conversion_errors in Rust."""
    aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG5.txt")
    with pytest.raises(Exception, match="SPIN"):
        aem.segment_to_attitude_trajectory(0)


def test_aem_g4_hermite_interpolation_method_errors(eop):
    """Mirror of test_aem_g4_hermite_interpolation_method_errors in Rust."""
    aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")
    with pytest.raises(Exception, match="HERMITE"):
        aem.segment_to_attitude_trajectory(0)


def test_aem_lagrange_interpolation_degree_zero_errors():
    """Mirror of test_aem_lagrange_interpolation_degree_zero_errors in Rust
    (src/ccsds/interop.rs)."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "QUATERNION",
        interpolation_method="LAGRANGE",
        interpolation_degree=0,
    )
    seg.add_state(
        AEMAttitudeState.from_quaternion(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    )
    aem = AEM("BRAHE")
    aem.add_segment(seg)

    with pytest.raises(Exception, match="degree"):
        aem.segment_to_attitude_trajectory(0)


def test_aem_to_attitude_trajectories_multi_segment():
    """Mirror of test_aem_to_attitude_trajectories_multi_segment in Rust."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0

    aem = AEM("BRAHE")
    for _ in range(2):
        seg = AEMSegment(
            "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
        )
        seg.add_state(
            AEMAttitudeState.from_quaternion(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
        )
        seg.add_state(
            AEMAttitudeState.from_quaternion(
                t1, bh.Quaternion(0.9998, 0.0, 0.0, 0.0196)
            )
        )
        aem.add_segment(seg)

    trajs = aem.to_attitude_trajectories()
    assert len(trajs) == 2
    for traj in trajs:
        assert len(traj) == 2


def test_aem_angvel_frame_a_reexpression():
    """Mirror of test_aem_angvel_frame_a_reexpression in Rust: ANGVEL_FRAME naming
    REF_FRAME_A must be re-expressed into REF_FRAME_B via omega_B = R(q) * omega_A."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "QUATERNION/ANGVEL",
        angvel_frame="EME2000",
    )

    quaternion = bh.Quaternion.from_euler_angle(
        bh.EulerAngle(bh.EulerAngleOrder.ZYX, 0.3, -0.4, 0.2, bh.AngleFormat.RADIANS)
    )
    omega_a = np.array([0.01, -0.02, 0.03])
    seg.add_state(AEMAttitudeState.from_quaternion_angvel(t0, quaternion, omega_a))

    aem = AEM("BRAHE")
    aem.add_segment(seg)

    traj = aem.segment_to_attitude_trajectory(0)
    stored_omega = traj.angular_velocity(t0)

    expected_omega = quaternion.to_rotation_matrix().to_matrix() @ omega_a
    np.testing.assert_allclose(stored_omega, expected_omega, atol=1e-12)

    # Sanity check: the re-expressed rate must differ from the raw
    # frame-A value (otherwise the re-expression path silently no-ops).
    assert np.linalg.norm(stored_omega - omega_a) > 1e-6


def test_aem_angvel_frame_neither_a_nor_b_errors_via_validate():
    """Mirror of test_aem_angvel_frame_neither_a_nor_b_errors_via_validate in Rust."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    seg = AEMSegment(
        "SAT1",
        "2024-001A",
        "EME2000",
        "SC_BODY_1",
        "UTC",
        t0,
        t1,
        "QUATERNION/ANGVEL",
        angvel_frame="ITRF2014",
    )
    seg.add_state(
        AEMAttitudeState.from_quaternion_angvel(
            t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0), np.array([0.01, -0.02, 0.03])
        )
    )
    aem = AEM("BRAHE")
    aem.add_segment(seg)

    with pytest.raises(Exception, match="ANGVEL_FRAME"):
        aem.segment_to_attitude_trajectory(0)


def test_aem_from_attitude_trajectory_without_rates():
    """Mirror of test_aem_from_attitude_trajectory_without_rates in Rust."""
    frame_a = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    frame_b = bh.AttitudeFrame.spacecraft("SC_BODY")
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    traj.add(t0 + 60.0, bh.Quaternion(0.9998, 0.0, 0.0, 0.0196))

    aem = AEM.from_attitude_trajectory(traj, "SAT1", "2024-001A", "BRAHE", "UTC")
    assert len(aem.segments) == 1
    seg = aem.segments[0]
    assert seg.object_name == "SAT1"
    assert seg.attitude_type == "QUATERNION"
    assert len(seg.states) == 2


def test_aem_from_attitude_trajectory_with_rates():
    frame_a = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    frame_b = bh.AttitudeFrame.spacecraft("SC_BODY")
    traj = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    omega = np.array([0.001, 0.0, 0.0])
    traj.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0), omega)
    traj.add(t0 + 60.0, bh.Quaternion(0.9998, 0.0, 0.0, 0.0196), omega)

    aem = AEM.from_attitude_trajectory(traj, "SAT1", "2024-001A", "BRAHE", "UTC")
    seg = aem.segments[0]
    assert seg.attitude_type == "QUATERNION/ANGVEL"
    assert seg.angvel_frame == "SC_BODY"


def test_aem_from_attitude_trajectory_empty_errors():
    """Mirror of test_aem_from_attitude_trajectory_empty_errors in Rust."""
    frame_a = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    frame_b = bh.AttitudeFrame.spacecraft("SC_BODY")
    traj = AttitudeTrajectory(frame_a, frame_b)
    with pytest.raises(Exception, match="empty"):
        AEM.from_attitude_trajectory(traj, "SAT1", "2024-001A", "BRAHE", "UTC")


def _assert_aem_state_match(a, b):
    """Compares every field an AEMAttitudeState variant carries, mirroring the Rust
    assert_aem_attitude_data_match helper used by test_aem_all_types_synthetic_three_way_round_trip."""
    assert a.attitude_type == b.attitude_type
    t = a.attitude_type

    if t in ("QUATERNION", "QUATERNION/DERIVATIVE", "QUATERNION/ANGVEL"):
        np.testing.assert_allclose(
            a.quaternion.to_vector(scalar_first=True),
            b.quaternion.to_vector(scalar_first=True),
            atol=1e-9,
        )
    if t == "QUATERNION/DERIVATIVE":
        np.testing.assert_allclose(a.derivative, b.derivative, atol=1e-9)
    if t in ("QUATERNION/ANGVEL", "EULER_ANGLE/ANGVEL"):
        np.testing.assert_allclose(a.angular_velocity, b.angular_velocity, atol=1e-9)
    if t in ("EULER_ANGLE", "EULER_ANGLE/DERIVATIVE", "EULER_ANGLE/ANGVEL"):
        assert a.euler_angles.order == b.euler_angles.order
        assert a.euler_angles.phi == pytest.approx(b.euler_angles.phi, abs=1e-9)
        assert a.euler_angles.theta == pytest.approx(b.euler_angles.theta, abs=1e-9)
        assert a.euler_angles.psi == pytest.approx(b.euler_angles.psi, abs=1e-9)
    if t == "EULER_ANGLE/DERIVATIVE":
        np.testing.assert_allclose(a.rates, b.rates, atol=1e-9)
    if t in ("SPIN", "SPIN/NUTATION", "SPIN/NUTATION_MOM"):
        assert a.spin_alpha == pytest.approx(b.spin_alpha, abs=1e-9)
        assert a.spin_delta == pytest.approx(b.spin_delta, abs=1e-9)
        assert a.spin_angle == pytest.approx(b.spin_angle, abs=1e-9)
        assert a.spin_angle_vel == pytest.approx(b.spin_angle_vel, abs=1e-9)
    if t == "SPIN/NUTATION":
        assert a.nutation == pytest.approx(b.nutation, abs=1e-9)
        assert a.nutation_period == pytest.approx(b.nutation_period, abs=1e-6)
        assert a.nutation_phase == pytest.approx(b.nutation_phase, abs=1e-9)
    if t == "SPIN/NUTATION_MOM":
        assert a.momentum_alpha == pytest.approx(b.momentum_alpha, abs=1e-9)
        assert a.momentum_delta == pytest.approx(b.momentum_delta, abs=1e-9)
        assert a.nutation_vel == pytest.approx(b.nutation_vel, abs=1e-9)


def _build_all_types_aem():
    """Builds a synthetic AEM with nine single-state segments, one per AEMAttitudeType
    variant. Mirror of build_all_types_aem in Rust (src/ccsds/aem.rs)."""
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    t1 = t0 + 60.0
    order = bh.EulerAngleOrder.ZXZ
    quaternion = bh.Quaternion(0.5, 0.5, 0.5, 0.5)
    euler_angles = bh.EulerAngle(
        order,
        math.radians(30.0),
        math.radians(45.0),
        math.radians(60.0),
        bh.AngleFormat.RADIANS,
    )

    variants = [
        ("QUATERNION", {}, AEMAttitudeState.from_quaternion(t0, quaternion)),
        (
            "QUATERNION/DERIVATIVE",
            {},
            AEMAttitudeState.from_quaternion_derivative(
                t0, quaternion, np.array([0.01, 0.02, 0.03, 0.04])
            ),
        ),
        (
            "QUATERNION/ANGVEL",
            {"angvel_frame": "SC_BODY_1"},
            AEMAttitudeState.from_quaternion_angvel(
                t0, quaternion, np.array([0.001, 0.002, 0.003])
            ),
        ),
        (
            "EULER_ANGLE",
            {"euler_rot_seq": order},
            AEMAttitudeState.from_euler_angle(t0, euler_angles),
        ),
        (
            "EULER_ANGLE/DERIVATIVE",
            {"euler_rot_seq": order},
            AEMAttitudeState.from_euler_angle_derivative(
                t0, euler_angles, np.array([0.001, 0.002, 0.003])
            ),
        ),
        (
            "EULER_ANGLE/ANGVEL",
            {"euler_rot_seq": order, "angvel_frame": "SC_BODY_1"},
            AEMAttitudeState.from_euler_angle_angvel(
                t0, euler_angles, np.array([0.001, 0.002, 0.003])
            ),
        ),
        ("SPIN", {}, AEMAttitudeState.from_spin(t0, 0.1, 0.2, 0.3, 0.4)),
        (
            "SPIN/NUTATION",
            {},
            AEMAttitudeState.from_spin_nutation(
                t0, 0.1, 0.2, 0.3, 0.4, 0.05, 120.0, 0.06
            ),
        ),
        (
            "SPIN/NUTATION_MOM",
            {},
            AEMAttitudeState.from_spin_nutation_mom(
                t0, 0.1, 0.2, 0.3, 0.4, 0.07, 0.08, 0.09
            ),
        ),
    ]

    aem = AEM("BRAHE")
    for attitude_type, kwargs, state in variants:
        seg = AEMSegment(
            f"SAT-{attitude_type}",
            "2024-001A",
            "EME2000",
            "SC_BODY_1",
            "UTC",
            t0,
            t1,
            attitude_type,
            center_name="EARTH",
            **kwargs,
        )
        seg.add_state(state)
        aem.add_segment(seg)
    return aem


@pytest.mark.parametrize("fmt", ["KVN", "XML", "JSON"])
def test_aem_all_types_three_way_round_trip(fmt):
    """Nine-variant round trip through KVN/XML/JSON. Python mirror of the Rust
    test_aem_all_types_synthetic_three_way_round_trip (build_all_types_aem)."""
    aem1 = _build_all_types_aem()
    assert len(aem1.segments) == 9

    content = aem1.to_string(fmt)
    aem2 = AEM.from_str(content)
    assert len(aem2.segments) == 9

    for seg1, seg2 in zip(aem1.segments, aem2.segments):
        assert seg2.object_name == seg1.object_name
        assert seg2.object_id == seg1.object_id
        assert seg2.attitude_type == seg1.attitude_type
        assert len(seg2.states) == len(seg1.states) == 1
        _assert_aem_state_match(seg1.states[0], seg2.states[0])


def test_aem_try_from_single_segment_helper_three_way():
    """AEM -> AttitudeTrajectory -> AEM three-way round trip through the interop helpers."""
    frame_a = bh.AttitudeFrame.reference(bh.ReferenceFrame.EME2000)
    frame_b = bh.AttitudeFrame.spacecraft("SC_BODY", "1")
    traj1 = AttitudeTrajectory(frame_a, frame_b)
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj1.add(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    traj1.add(t0 + 60.0, bh.Quaternion(0.9998, 0.0, 0.0, 0.0196))

    aem = AEM.from_attitude_trajectory(traj1, "SAT1", "2024-001A", "BRAHE", "UTC")
    traj2 = aem.segment_to_attitude_trajectory(0)

    assert traj2.frame_a == traj1.frame_a
    assert traj2.frame_b == traj1.frame_b
    assert len(traj2) == len(traj1)
    np.testing.assert_allclose(
        traj2.quaternion(t0).to_vector(scalar_first=True),
        traj1.quaternion(t0).to_vector(scalar_first=True),
        atol=1e-12,
    )
