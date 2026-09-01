"""Tests for the unified ReferenceFrame/BodyFrame types and the frame/object registries.

Mirrors the Rust tests in src/frames/frame.rs, registry.rs, object_registry.rs,
and graph.rs.
"""

import numpy as np
import pytest

import brahe as bh
from brahe.ccsds import OEM


def test_frame_constructors_and_display():
    f = bh.ReferenceFrame.RTN("SC")
    assert str(f) == "RTN (rotating)@SC"
    assert f.is_bound()
    assert f.object() == "SC"
    # EQW/PQW default to the inertial variant, so construction never errors
    assert str(bh.ReferenceFrame.PQW("SC")) == "PQW (inertial)@SC"
    assert str(bh.ReferenceFrame.CSS("SC", "1")) == "CSS_1@SC"
    assert str(bh.ReferenceFrame.SC_BODY("SC")) == "SC_BODY@SC"
    unbound = bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY())
    assert not unbound.is_bound()
    assert unbound.object() is None
    assert str(unbound) == "SC_BODY"


def test_frame_family_constructors_display():
    assert str(bh.ReferenceFrame.LVLH("SC")) == "LVLH (rotating)@SC"
    assert str(bh.ReferenceFrame.NTW("SC")) == "NTW (rotating)@SC"
    assert str(bh.ReferenceFrame.TNW("SC")) == "TNW (rotating)@SC"
    assert str(bh.ReferenceFrame.SEZ("SC")) == "SEZ (rotating)@SC"
    assert str(bh.ReferenceFrame.VNC("SC")) == "VNC (rotating)@SC"
    assert str(bh.ReferenceFrame.NSW("SC")) == "NSW (rotating)@SC"
    assert str(bh.ReferenceFrame.EQW("SC")) == "EQW (inertial)@SC"
    assert str(bh.ReferenceFrame.ACC("SC", "1")) == "ACC_1@SC"
    assert str(bh.ReferenceFrame.AST("SC", "1")) == "AST_1@SC"
    assert str(bh.ReferenceFrame.DSS("SC", "1")) == "DSS_1@SC"
    assert str(bh.ReferenceFrame.ESA("SC", "1")) == "ESA_1@SC"
    assert str(bh.ReferenceFrame.GYRO_FRAME("SC", "1")) == "GYRO_FRAME_1@SC"
    assert str(bh.ReferenceFrame.IMU_FRAME("SC", "1")) == "IMU_FRAME_1@SC"
    assert str(bh.ReferenceFrame.INSTRUMENT("SC", "A")) == "INSTRUMENT_A@SC"
    assert str(bh.ReferenceFrame.MTA("SC", "1")) == "MTA_1@SC"
    assert str(bh.ReferenceFrame.RW("SC", "4")) == "RW_4@SC"
    assert str(bh.ReferenceFrame.SA("SC", "1")) == "SA_1@SC"
    assert str(bh.ReferenceFrame.SENSOR("SC", "10")) == "SENSOR_10@SC"
    assert str(bh.ReferenceFrame.STARTRACKER("SC", "2")) == "STARTRACKER_2@SC"
    assert str(bh.ReferenceFrame.TAM("SC", "1")) == "TAM_1@SC"
    assert str(bh.ReferenceFrame.ACTUATOR("SC", "1")) == "ACTUATOR_1@SC"


def test_frame_eq_and_repr():
    assert bh.ReferenceFrame.RTN("SC") == bh.ReferenceFrame.RTN("SC")
    assert bh.ReferenceFrame.RTN("SC") != bh.ReferenceFrame.RTN("SC2")
    assert bh.ReferenceFrame.RTN("SC") != bh.ReferenceFrame.LVLH("SC")
    assert "RTN" in repr(bh.ReferenceFrame.RTN("SC"))


def test_orbit_relative_validation():
    with pytest.raises(ValueError):
        bh.ReferenceFrame.orbit_relative(
            bh.OrbitRelativeKind.EQW, bh.OrbitRelativeVariant.ROTATING, None
        )
    ok = bh.ReferenceFrame.orbit_relative(
        bh.OrbitRelativeKind.RTN, bh.OrbitRelativeVariant.INERTIAL, "SC"
    )
    assert str(ok) == "RTN (inertial)@SC"


def test_orbit_relative_rejects_non_enum_arguments():
    with pytest.raises(TypeError):
        bh.ReferenceFrame.orbit_relative("RTN", bh.OrbitRelativeVariant.ROTATING, "SC")
    with pytest.raises(TypeError):
        bh.ReferenceFrame.orbit_relative(bh.OrbitRelativeKind.RTN, "rotating", "SC")


def test_orbit_relative_kind_and_variant_display():
    assert str(bh.OrbitRelativeKind.RTN) == "RTN"
    assert str(bh.OrbitRelativeKind.EQW) == "EQW"
    assert str(bh.OrbitRelativeVariant.ROTATING) == "rotating"
    assert str(bh.OrbitRelativeVariant.INERTIAL) == "inertial"
    assert bh.OrbitRelativeKind.RTN == bh.OrbitRelativeKind.RTN
    assert bh.OrbitRelativeKind.RTN != bh.OrbitRelativeKind.LVLH
    assert bh.OrbitRelativeVariant.ROTATING != bh.OrbitRelativeVariant.INERTIAL


def test_frame_celestial_constructor():
    gcrf = bh.ReferenceFrame.celestial(bh.CelestialFrame.GCRF)
    assert gcrf.is_bound()
    assert gcrf.object() is None
    assert str(gcrf) == "GCRF"


def test_body_frame_display_all_variants():
    cases = [
        (bh.BodyFrame.ACC("1"), "ACC_1"),
        (bh.BodyFrame.ACTUATOR(), "ACTUATOR"),
        (bh.BodyFrame.AST("1"), "AST_1"),
        (bh.BodyFrame.CSS("2"), "CSS_2"),
        (bh.BodyFrame.DSS("1"), "DSS_1"),
        (bh.BodyFrame.ESA("1"), "ESA_1"),
        (bh.BodyFrame.GYRO_FRAME("1"), "GYRO_FRAME_1"),
        (bh.BodyFrame.IMU_FRAME("2"), "IMU_FRAME_2"),
        (bh.BodyFrame.INSTRUMENT("A"), "INSTRUMENT_A"),
        (bh.BodyFrame.MTA("1"), "MTA_1"),
        (bh.BodyFrame.RW("4"), "RW_4"),
        (bh.BodyFrame.SA("1"), "SA_1"),
        (bh.BodyFrame.SC_BODY(), "SC_BODY"),
        (bh.BodyFrame.SENSOR("10"), "SENSOR_10"),
        (bh.BodyFrame.STARTRACKER("2"), "STARTRACKER_2"),
        (bh.BodyFrame.TAM("1"), "TAM_1"),
    ]
    for frame, expected in cases:
        assert str(frame) == expected


def test_register_frame_validation(clear_frame_registries):
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    # Parent chain must exist: CSS -> SC_BODY fails before SC_BODY registered
    with pytest.raises(bh.BraheError, match="SC_BODY@SC"):
        bh.register_frame(
            bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q
        )
    # Valid chain
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q)
    bh.register_frame(
        bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q
    )
    # Replace that would self-cycle rejected: SC_BODY reparented onto CSS
    with pytest.raises(bh.BraheError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"), bh.ReferenceFrame.CSS("SC", "1"), q
        )
    assert bh.unregister_frame(bh.ReferenceFrame.CSS("SC", "1"))
    assert not bh.unregister_frame(bh.ReferenceFrame.CSS("SC", "1"))


def test_register_frame_rejects_non_body_target(clear_frame_registries):
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    unbound = bh.ReferenceFrame.body(None, bh.BodyFrame.SC_BODY())
    with pytest.raises(bh.BraheError, match="bound Body frame"):
        bh.register_frame(unbound, bh.CelestialFrame.GCRF, q)
    with pytest.raises(bh.BraheError, match="bound Body frame"):
        bh.register_frame(bh.ReferenceFrame.RTN("SC"), bh.CelestialFrame.GCRF, q)


def test_register_frame_bad_provider_type_raises(clear_frame_registries):
    with pytest.raises(TypeError):
        bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, 5)


def test_register_frame_rotation_matrix_provider(clear_frame_registries):
    r = bh.RotationMatrix(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, r)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    got = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc
    )
    np.testing.assert_allclose(got, r.to_matrix(), atol=1e-15)


def test_register_frame_euler_angle_provider(clear_frame_registries):
    e = bh.EulerAngle(bh.EulerAngleOrder.ZYX, 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, e)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    got = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc
    )
    np.testing.assert_allclose(got, e.to_rotation_matrix().to_matrix(), atol=1e-14)


def test_register_frame_euler_axis_provider(clear_frame_registries):
    a = bh.EulerAxis(np.array([0.0, 0.0, 1.0]), 0.5, bh.AngleFormat.RADIANS)
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, a)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    got = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc
    )
    np.testing.assert_allclose(got, a.to_rotation_matrix().to_matrix(), atol=1e-14)


def test_register_frame_constant_provider_rejects_omega(clear_frame_registries):
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"),
            bh.CelestialFrame.GCRF,
            q,
            omega=lambda epc: np.zeros(3),
        )
    with pytest.raises(ValueError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"),
            bh.CelestialFrame.GCRF,
            q,
            numerical_rates_step=1.0,
        )


def test_register_frame_rejects_non_callable_omega(clear_frame_registries):
    with pytest.raises(TypeError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"),
            bh.CelestialFrame.GCRF,
            lambda epc: np.eye(3),
            omega=5,
        )


def test_register_frame_rejects_bad_numerical_rates_step(clear_frame_registries):
    with pytest.raises(ValueError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"),
            bh.CelestialFrame.GCRF,
            lambda epc: np.eye(3),
            numerical_rates_step=0.0,
        )
    with pytest.raises(ValueError):
        bh.register_frame(
            bh.ReferenceFrame.SC_BODY("SC"),
            bh.CelestialFrame.GCRF,
            lambda epc: np.eye(3),
            numerical_rates_step=-1.0,
        )


def test_register_frame_replace_revalidates_chain(clear_frame_registries):
    q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q)
    # Replace with a still-valid parent chain: succeeds.
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.ITRF, q)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    r = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc
    )
    expected = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.CelestialFrame.ITRF, epc
    )
    np.testing.assert_allclose(r, expected, atol=1e-15)


def test_object_registry_round_trip_and_errors(clear_frame_registries):
    x = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    bh.register_object("SC", lambda epc: x, bh.CelestialFrame.GCRF)
    assert bh.registered_objects() == ["SC"]
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    got = bh.state_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("SC"), epc, x
    )
    expected = bh.state_eci_to_rtn(x, x)
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert bh.unregister_object("SC")
    assert not bh.unregister_object("SC")


def test_registered_objects_sorted(clear_frame_registries):
    x = np.zeros(6)
    bh.register_object("ZULU", lambda epc: x, bh.CelestialFrame.GCRF)
    bh.register_object("ALFA", lambda epc: x, bh.CelestialFrame.GCRF)
    assert bh.registered_objects() == ["ALFA", "ZULU"]


def test_register_object_bad_provider_type_raises(clear_frame_registries):
    with pytest.raises(TypeError):
        bh.register_object("SC", 5, bh.CelestialFrame.GCRF)


def test_register_object_callable_bad_return_raises(clear_frame_registries):
    bh.register_object("SC", lambda epc: np.zeros(3), bh.CelestialFrame.GCRF)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    with pytest.raises(RuntimeError):
        bh.rotation_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("SC"), epc
        )


def test_register_object_orbit_trajectory_dimension_error(clear_frame_registries):
    traj = bh.OrbitTrajectory(
        9, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
    )
    with pytest.raises(bh.BraheError, match="6-dimensional"):
        bh.register_object("SC", traj, bh.CelestialFrame.GCRF)


def test_register_object_orbit_trajectory_keplerian_representation_rejected(
    clear_frame_registries,
):
    epc0 = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    oe = np.array([[bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0]])
    traj = bh.OrbitTrajectory.from_orbital_data(
        [epc0],
        oe,
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.KEPLERIAN,
        bh.AngleFormat.DEGREES,
        None,
    )
    with pytest.raises(ValueError, match="Cartesian"):
        bh.register_object("SC", traj, bh.CelestialFrame.GCRF)


def test_register_object_orbit_trajectory(clear_frame_registries):
    epc0 = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    epochs = [epc0]
    x = np.array([[bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0]])
    traj = bh.OrbitTrajectory.from_orbital_data(
        epochs, x, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, None
    )
    bh.register_object("SC", traj, bh.CelestialFrame.GCRF)
    # The object's own position expressed in its own RTN frame is the origin.
    got = bh.position_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("SC"), epc0, x[0, :3]
    )
    np.testing.assert_allclose(got, [0.0, 0.0, 0.0], atol=1e-6)


def test_body_chain_matches_manual_composition(clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    q_body = bh.Quaternion.from_euler_axis(
        bh.EulerAxis(np.array([0.0, 0.0, 1.0]), 0.3, bh.AngleFormat.RADIANS)
    )
    q_css = bh.Quaternion.from_euler_axis(
        bh.EulerAxis(np.array([1.0, 0.0, 0.0]), 1.1, bh.AngleFormat.RADIANS)
    )
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q_body)
    bh.register_frame(
        bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q_css
    )
    r = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "1"), epc
    )
    expected = (
        q_css.to_rotation_matrix().to_matrix() @ q_body.to_rotation_matrix().to_matrix()
    )
    np.testing.assert_allclose(r, expected, atol=1e-14)
    r_inv = bh.rotation_frame_to_frame(
        bh.ReferenceFrame.CSS("SC", "1"), bh.CelestialFrame.GCRF, epc
    )
    np.testing.assert_allclose(r_inv, expected.T, atol=1e-14)


def test_missing_link_error_names_fix(clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    with pytest.raises(RuntimeError, match="register_frame"):
        bh.rotation_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "1"), epc
        )
    with pytest.raises(RuntimeError, match="not registered"):
        bh.rotation_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("A"), epc
        )


def test_rtn_rotation_matches_relative_motion(clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    x = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    bh.register_object("A", lambda epc: x, bh.CelestialFrame.GCRF)
    r = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("A"), epc
    )
    np.testing.assert_allclose(r, bh.rotation_eci_to_rtn(x), atol=1e-14)


def test_two_object_rtn_matches_state_eci_to_rtn(clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    x_a = bh.state_koe_to_eci(
        np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0]),
        bh.AngleFormat.DEGREES,
    )
    x_b = bh.state_koe_to_eci(
        np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.2]),
        bh.AngleFormat.DEGREES,
    )
    bh.register_object("A", lambda epc: x_a, bh.CelestialFrame.GCRF)
    got = bh.state_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("A"), epc, x_b
    )
    np.testing.assert_allclose(got, bh.state_eci_to_rtn(x_a, x_b), atol=1e-9)


def test_sun_vector_in_sensor_frame(eop, clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    x_sc = bh.state_koe_to_eci(
        np.array([bh.R_EARTH + 500e3, 0.0, 97.8, 15.0, 30.0, 45.0]),
        bh.AngleFormat.DEGREES,
    )
    bh.register_object("SC", lambda epc: x_sc, bh.CelestialFrame.GCRF)
    q_body = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    q_css = bh.Quaternion.from_euler_axis(
        bh.EulerAxis(np.array([0.0, 1.0, 0.0]), 0.7, bh.AngleFormat.RADIANS)
    )
    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q_body)
    bh.register_frame(
        bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q_css
    )

    sun_gcrf = bh.sun_position(epc)
    got = bh.position_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.CSS("SC", "1"), epc, sun_gcrf
    )
    manual = (
        q_css.to_rotation_matrix().to_matrix()
        @ q_body.to_rotation_matrix().to_matrix()
        @ (sun_gcrf - x_sc[:3])
    )
    np.testing.assert_allclose(got, manual, atol=150.0)


def test_state_transform_missing_rates_errors(eop, clear_frame_registries):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    t0 = epc - 10.0

    def spinning(e):
        dt = e - t0
        c, s = np.cos(0.001 * dt), np.sin(0.001 * dt)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, spinning)
    bh.register_object("SC", lambda epc: np.zeros(6), bh.CelestialFrame.GCRF)

    with pytest.raises(RuntimeError, match="with_numerical_rates"):
        bh.state_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc, np.zeros(6)
        )

    # The position transform needs no rates and still succeeds.
    bh.position_frame_to_frame(
        bh.CelestialFrame.GCRF,
        bh.ReferenceFrame.SC_BODY("SC"),
        epc,
        np.array([1.0, 2.0, 3.0]),
    )


def test_register_frame_numerical_rates_step_matches_explicit_omega(
    clear_frame_registries,
):
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    t0 = epc - 10.0
    rate = 1.0e-3

    def spinning(e):
        dt = e - t0
        c, s = np.cos(rate * dt), np.sin(rate * dt)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    bh.register_frame(
        bh.ReferenceFrame.SC_BODY("SC"),
        bh.CelestialFrame.GCRF,
        spinning,
        numerical_rates_step=0.1,
    )
    bh.register_object("SC", lambda epc: np.zeros(6), bh.CelestialFrame.GCRF)
    numerical = bh.state_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc, x
    )
    bh.clear_frame_registry()
    bh.clear_object_registry()

    bh.register_frame(
        bh.ReferenceFrame.SC_BODY("SC"),
        bh.CelestialFrame.GCRF,
        spinning,
        omega=lambda e: np.array([0.0, 0.0, rate]),
    )
    bh.register_object("SC", lambda epc: np.zeros(6), bh.CelestialFrame.GCRF)
    analytic = bh.state_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epc, x
    )

    np.testing.assert_allclose(numerical, analytic, atol=1e-6)


def test_batch_matches_singular_for_frame_args(clear_frame_registries):
    epc0 = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    epochs = [epc0 + i * 60.0 for i in range(5)]
    x_a = bh.state_koe_to_eci(
        np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0]),
        bh.AngleFormat.DEGREES,
    )
    bh.register_object("A", lambda epc: x_a, bh.CelestialFrame.GCRF)
    batch = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("A"), epochs
    )
    for i, epc in enumerate(epochs):
        singular = bh.rotation_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("A"), epc
        )
        np.testing.assert_array_equal(batch[i], singular)


def test_batch_callback_provider_reacquires_gil_from_worker_threads(
    clear_frame_registries,
):
    """A large batch forces evaluation on the Rayon thread pool (see
    set_vectorization_length_threshold); each worker thread must be able to
    reacquire the GIL to call back into the Python rotation/state callbacks
    registered via register_frame/register_object."""
    epc0 = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    n = 32
    epochs = [epc0 + i * 60.0 for i in range(n)]
    t0 = epc0
    rate = 1.0e-3

    def spin(epc):
        theta = rate * (epc - t0)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, spin)

    original_threshold = bh.get_vectorization_length_threshold()
    bh.set_vectorization_length_threshold(1)
    try:
        batch = bh.rotation_frame_to_frame(
            bh.CelestialFrame.GCRF, bh.ReferenceFrame.SC_BODY("SC"), epochs
        )
    finally:
        bh.set_vectorization_length_threshold(original_threshold)

    assert batch.shape == (n, 3, 3)
    for i, epc in enumerate(epochs):
        np.testing.assert_allclose(batch[i], spin(epc), atol=1e-12)


def test_oem_register_for(clear_frame_registries):
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample5.txt")
    oem.register_for("ISS")
    assert "ISS" in bh.registered_objects()


def test_register_object_from_naif_moon(naif_cache_setup, clear_frame_registries):
    bh.register_object_from_naif("MOON", 301)
    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
    got = bh.rotation_frame_to_frame(
        bh.CelestialFrame.GCRF, bh.ReferenceFrame.RTN("MOON"), epc
    )
    direct = bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    np.testing.assert_allclose(got, bh.rotation_eci_to_rtn(direct), atol=1e-14)
