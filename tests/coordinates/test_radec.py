import math

import numpy as np
import pytest
import brahe as bh
from brahe import AngleFormat


RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def test_position_radec_to_inertial():
    # ra=0, dec=0, r=1 -> +X axis
    x = bh.position_radec_to_inertial(np.array([0.0, 0.0, 1.0]), AngleFormat.DEGREES)
    assert x[0] == pytest.approx(1.0, abs=1e-12)
    assert x[1] == pytest.approx(0.0, abs=1e-12)
    assert x[2] == pytest.approx(0.0, abs=1e-12)

    # ra=45, dec=45, r=2 -> analytic values
    x = bh.position_radec_to_inertial(np.array([45.0, 45.0, 2.0]), AngleFormat.DEGREES)
    assert x[0] == pytest.approx(2.0 * (math.cos(math.radians(45.0)) ** 2), abs=1e-12)
    assert x[2] == pytest.approx(2.0 * math.sin(math.radians(45.0)), abs=1e-12)

    x = bh.position_radec_to_inertial(np.array([90.0, 0.0, 1.0]), AngleFormat.DEGREES)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[1] == pytest.approx(1.0, abs=1e-12)
    assert x[2] == pytest.approx(0.0, abs=1e-12)

    x = bh.position_radec_to_inertial(np.array([0.0, 90.0, 1.0]), AngleFormat.DEGREES)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[1] == pytest.approx(0.0, abs=1e-12)
    assert x[2] == pytest.approx(1.0, abs=1e-12)


def test_position_radec_inertial_round_trip():
    r = 7000e3
    for ra in [0.0, 90.0, 181.0, 359.0]:
        for dec in [-89.0, -45.0, 0.0, 45.0, 89.0]:
            x_radec = np.array([ra, dec, r])
            x_inertial = bh.position_radec_to_inertial(x_radec, AngleFormat.DEGREES)
            x_radec_back = bh.position_inertial_to_radec(
                x_inertial, AngleFormat.DEGREES
            )

            assert x_radec_back[0] == pytest.approx(ra, abs=1e-9)
            assert x_radec_back[1] == pytest.approx(dec, abs=1e-9)
            assert x_radec_back[2] == pytest.approx(r, abs=1e-6)


def test_position_inertial_to_radec_ra_normalization():
    # x=(1, -1e-3, 0): atan2 gives small negative angle -> expect ra in [0,360)
    x_inertial = np.array([1.0, -1e-3, 0.0])
    x_radec = bh.position_inertial_to_radec(x_inertial, AngleFormat.DEGREES)

    assert x_radec[0] >= 0.0 and x_radec[0] < 360.0
    assert x_radec[0] == pytest.approx(359.9427042395855, abs=1e-9)
    assert x_radec[1] == pytest.approx(0.0, abs=1e-9)
    assert x_radec[2] == pytest.approx(math.sqrt(1.0 + 1e-6), abs=1e-9)


def test_position_inertial_to_radec_polar_singularity():
    # x=(0,0,7000e3) -> ra=0, dec=90, range=7000e3
    x_inertial = np.array([0.0, 0.0, 7000e3])
    x_radec = bh.position_inertial_to_radec(x_inertial, AngleFormat.DEGREES)

    assert x_radec[0] == pytest.approx(0.0, abs=1e-12)
    assert x_radec[1] == pytest.approx(90.0, abs=1e-12)
    assert x_radec[2] == pytest.approx(7000e3, abs=1e-9)


def test_state_radec_inertial_round_trip():
    # LEO-like state: r=[7000e3, 0, 0], v=[0, 6.5e3, 3.0e3]
    x_inertial = np.array([7000e3, 0.0, 0.0, 0.0, 6.5e3, 3.0e3])
    x_radec = bh.state_inertial_to_radec(x_inertial, AngleFormat.DEGREES)
    x_inertial_back = bh.state_radec_to_inertial(x_radec, AngleFormat.DEGREES)

    for k in range(6):
        assert x_inertial_back[k] == pytest.approx(x_inertial[k], abs=1e-6)


def test_state_inertial_to_radec_rates():
    # Circular equatorial orbit r=[a,0,0], v=[0,vc,0]:
    # range_rate=0, dec_rate=0, ra_rate = vc/a rad/s (convert per angle_format)
    a = 7000e3
    vc = 7500.0
    x_inertial = np.array([a, 0.0, 0.0, 0.0, vc, 0.0])

    x_radec = bh.state_inertial_to_radec(x_inertial, AngleFormat.RADIANS)
    assert x_radec[5] == pytest.approx(0.0, abs=1e-9)  # range_rate
    assert x_radec[4] == pytest.approx(0.0, abs=1e-9)  # dec_rate
    assert x_radec[3] == pytest.approx(vc / a, abs=1e-12)  # ra_rate

    x_radec_deg = bh.state_inertial_to_radec(x_inertial, AngleFormat.DEGREES)
    assert x_radec_deg[3] == pytest.approx((vc / a) * RAD2DEG, abs=1e-9)


def test_state_inertial_to_radec_polar_velocity_resolution():
    # r=[0,0,7000e3], v=[100.0, 0.0, 0.0]: ra from velocity components:
    # sin(ra)=v_j/sqrt(v_i^2+v_j^2) -> ra=0 here; with v=[0,100,0] -> ra=90 deg
    x_inertial = np.array([0.0, 0.0, 7000e3, 100.0, 0.0, 0.0])
    x_radec = bh.state_inertial_to_radec(x_inertial, AngleFormat.DEGREES)
    assert x_radec[0] == pytest.approx(0.0, abs=1e-12)
    assert x_radec[1] == pytest.approx(90.0, abs=1e-12)
    assert x_radec[2] == pytest.approx(7000e3, abs=1e-9)

    x_inertial = np.array([0.0, 0.0, 7000e3, 0.0, 100.0, 0.0])
    x_radec = bh.state_inertial_to_radec(x_inertial, AngleFormat.DEGREES)
    assert x_radec[0] == pytest.approx(90.0, abs=1e-12)
    assert x_radec[1] == pytest.approx(90.0, abs=1e-12)
    assert x_radec[2] == pytest.approx(7000e3, abs=1e-9)


def test_radec_degrees_radians_parity():
    # Same input expressed in both formats produces identical Cartesian output
    ra_deg = 37.5
    dec_deg = -12.3
    r = 12345.0

    x_deg = bh.position_radec_to_inertial(
        np.array([ra_deg, dec_deg, r]), AngleFormat.DEGREES
    )
    x_rad = bh.position_radec_to_inertial(
        np.array([ra_deg * DEG2RAD, dec_deg * DEG2RAD, r]), AngleFormat.RADIANS
    )

    for k in range(3):
        assert x_deg[k] == pytest.approx(x_rad[k], abs=1e-12)

    ra_dot_deg = 0.01
    dec_dot_deg = -0.02
    r_dot = 5.0

    s_deg = bh.state_radec_to_inertial(
        np.array([ra_deg, dec_deg, r, ra_dot_deg, dec_dot_deg, r_dot]),
        AngleFormat.DEGREES,
    )
    s_rad = bh.state_radec_to_inertial(
        np.array(
            [
                ra_deg * DEG2RAD,
                dec_deg * DEG2RAD,
                r,
                ra_dot_deg * DEG2RAD,
                dec_dot_deg * DEG2RAD,
                r_dot,
            ]
        ),
        AngleFormat.RADIANS,
    )

    for k in range(6):
        assert s_deg[k] == pytest.approx(s_rad[k], abs=1e-12)


def test_radec_azel_round_trip(eop):
    epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
    site = np.array([-122.17, 37.43, 100.0])  # Stanford, deg/deg/m
    for ra, dec in [(0.0, 0.0), (101.28, -16.72), (279.23, 38.78)]:
        range_ = 12345.6
        azel = bh.position_radec_to_azel(
            np.array([ra, dec, range_]), site, epc, AngleFormat.DEGREES
        )
        radec = bh.position_azel_to_radec(azel, site, epc, AngleFormat.DEGREES)

        assert radec[0] == pytest.approx(ra, abs=1e-9)
        assert radec[1] == pytest.approx(dec, abs=1e-9)
        assert radec[2] == pytest.approx(range_, abs=1e-9)
        assert azel[2] == pytest.approx(range_, abs=1e-9)


def test_radec_to_azel_zenith(eop):
    epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
    # lon=0, lat=0 so geodetic == geocentric; altitude is irrelevant for direction.
    site = np.array([0.0, 0.0, 0.0])

    # Compute the site's zenith direction in ECI: site ECEF position ->
    # rotate to ECI -> convert to ra/dec (geocentric zenith).
    site_ecef = bh.position_geodetic_to_ecef(site, AngleFormat.DEGREES)
    site_eci = bh.rotation_ecef_to_eci(epc) @ site_ecef
    zenith_radec = bh.position_inertial_to_radec(site_eci, AngleFormat.DEGREES)

    azel = bh.position_radec_to_azel(
        np.array([zenith_radec[0], zenith_radec[1], 1.0]),
        site,
        epc,
        AngleFormat.DEGREES,
    )

    assert azel[1] == pytest.approx(90.0, abs=1e-6)


def test_apply_proper_motion_zero_motion():
    # Zero proper motion with no parallax/radial velocity leaves (ra, dec)
    # unchanged regardless of the epoch span.
    epoch_from = bh.Epoch.from_mjd(51544.5, bh.TimeSystem.TT)
    epoch_to = bh.Epoch.from_mjd(51544.5 + 50.0 * 365.25, bh.TimeSystem.TT)

    ra, dec = bh.apply_proper_motion(
        123.456,
        -45.678,
        0.0,
        0.0,
        None,
        None,
        epoch_from,
        epoch_to,
        AngleFormat.DEGREES,
    )

    assert ra == pytest.approx(123.456, abs=1e-13)
    assert dec == pytest.approx(-45.678, abs=1e-13)


def test_apply_proper_motion_linear_small_angle():
    # At dec=0, mu_ra* == mu_ra (cos(dec) = 1), so a pure RA proper motion of
    # 1000 mas/yr over 10 years produces a 10000 mas = 10 arcsec shift in RA.
    epoch_from = bh.Epoch.from_mjd(51544.5, bh.TimeSystem.TT)
    epoch_to = bh.Epoch.from_mjd(51544.5 + 10.0 * 365.25, bh.TimeSystem.TT)

    ra, dec = bh.apply_proper_motion(
        10.0,
        0.0,
        1000.0,
        0.0,
        None,
        None,
        epoch_from,
        epoch_to,
        AngleFormat.DEGREES,
    )

    delta_ra_arcsec = (ra - 10.0) * 3600.0
    assert delta_ra_arcsec == pytest.approx(10.0, abs=1e-3)
    assert dec == pytest.approx(0.0, abs=1e-12)


def test_apply_proper_motion_round_trip():
    # Forward propagation by tau, followed by propagation of the resulting
    # position by -tau using the same (un-negated) proper motion, recovers
    # the starting direction to sub-microarcsecond precision in the linear
    # (no parallax/radial velocity) case.
    ra0 = 45.0
    dec0 = -20.0
    pm_ra = 25.0
    pm_dec = 15.0

    epoch_from = bh.Epoch.from_mjd(51544.5, bh.TimeSystem.TT)
    epoch_to = bh.Epoch.from_mjd(51544.5 + 5.0 * 365.25, bh.TimeSystem.TT)

    ra1, dec1 = bh.apply_proper_motion(
        ra0,
        dec0,
        pm_ra,
        pm_dec,
        None,
        None,
        epoch_from,
        epoch_to,
        AngleFormat.DEGREES,
    )
    ra2, dec2 = bh.apply_proper_motion(
        ra1,
        dec1,
        pm_ra,
        pm_dec,
        None,
        None,
        epoch_to,
        epoch_from,
        AngleFormat.DEGREES,
    )

    u0 = bh.position_radec_to_inertial(np.array([ra0, dec0, 1.0]), AngleFormat.DEGREES)
    u2 = bh.position_radec_to_inertial(np.array([ra2, dec2, 1.0]), AngleFormat.DEGREES)
    sep_rad = math.atan2(np.linalg.norm(np.cross(u0, u2)), np.dot(u0, u2))
    sep_uas = sep_rad * RAD2DEG * 3600.0 * 1e6

    assert sep_uas < 1.0, f"round-trip separation {sep_uas} uas exceeds 1 uas"


def test_apply_proper_motion_barnard():
    # Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
    ra0 = 269.45402305
    dec0 = 4.66828815
    pm_ra = -797.84
    pm_dec = 10326.93
    plx = 549.30
    rv = -106.8

    # J1991.25 in MJD (TT): JD = 2451545.0 + (1991.25-2000.0)*365.25 = 2448349.0625
    epoch_from = bh.Epoch.from_mjd(48348.5625, bh.TimeSystem.TT)
    epoch_to = bh.Epoch.from_mjd(48348.5625 + 10.0 * 365.25, bh.TimeSystem.TT)

    ra_lin, dec_lin = bh.apply_proper_motion(
        ra0,
        dec0,
        pm_ra,
        pm_dec,
        None,
        None,
        epoch_from,
        epoch_to,
        AngleFormat.DEGREES,
    )
    ra_full, dec_full = bh.apply_proper_motion(
        ra0,
        dec0,
        pm_ra,
        pm_dec,
        plx,
        rv,
        epoch_from,
        epoch_to,
        AngleFormat.DEGREES,
    )

    u0 = bh.position_radec_to_inertial(np.array([ra0, dec0, 1.0]), AngleFormat.DEGREES)
    u_lin = bh.position_radec_to_inertial(
        np.array([ra_lin, dec_lin, 1.0]), AngleFormat.DEGREES
    )
    u_full = bh.position_radec_to_inertial(
        np.array([ra_full, dec_full, 1.0]), AngleFormat.DEGREES
    )

    # Total displacement over 10 years, small-angle approximation.
    expected_arcsec = math.sqrt(pm_ra**2 + pm_dec**2) * 10.0 / 1000.0
    sep_full_arcsec = (
        math.atan2(np.linalg.norm(np.cross(u0, u_full)), np.dot(u0, u_full))
        * RAD2DEG
        * 3600.0
    )
    assert sep_full_arcsec == pytest.approx(expected_arcsec, abs=0.1)

    # Perspective acceleration (from parallax/radial velocity) shifts the
    # propagated position by more than 1 mas relative to the linear
    # (proper-motion-only) propagation.
    perspective_shift_mas = (
        math.atan2(np.linalg.norm(np.cross(u_lin, u_full)), np.dot(u_lin, u_full))
        * RAD2DEG
        * 3600.0
        * 1000.0
    )
    assert perspective_shift_mas > 1.0, (
        f"perspective acceleration shift {perspective_shift_mas} mas too small"
    )


def test_position_radec_to_inertial_input_types():
    for x in ([0.0, 0.0, 1.0], (0.0, 0.0, 1.0), np.array([0.0, 0.0, 1.0])):
        r = bh.position_radec_to_inertial(x, AngleFormat.DEGREES)
        assert r == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
