import numpy as np
import pytest
import brahe
from pytest import approx


@pytest.fixture()
def static_eop():
    pm_x = 0.0349282 * brahe.AS2RAD
    pm_y = 0.4833163 * brahe.AS2RAD
    ut1_utc = -0.072073685
    dX = 0.0001750 * brahe.AS2RAD * 1.0e-3
    dY = -0.0002259 * brahe.AS2RAD * 1.0e-3
    eop = brahe.StaticEOPProvider.from_values(pm_x, pm_y, ut1_utc, dX, dY, 0.0)
    brahe.set_global_eop_provider(eop)


def test_bias_precession_nutation(static_eop):
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0, 0.0, brahe.UTC)

    rc2i = brahe.bias_precession_nutation(epc)

    tol = 1e-8
    assert rc2i[0, 0] == approx(+0.999999746339445, abs=tol)
    assert rc2i[0, 1] == approx(-0.000000005138822, abs=tol)
    assert rc2i[0, 2] == approx(-0.000712264730072, abs=tol)

    assert rc2i[1, 0] == approx(-0.000000026475227, abs=tol)
    assert rc2i[1, 1] == approx(+0.999999999014975, abs=tol)
    assert rc2i[1, 2] == approx(-0.000044385242827, abs=tol)

    assert rc2i[2, 0] == approx(+0.000712264729599, abs=tol)
    assert rc2i[2, 1] == approx(+0.000044385250426, abs=tol)
    assert rc2i[2, 2] == approx(+0.999999745354420, abs=tol)


def test_earth_rotation(static_eop):
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    r = brahe.earth_rotation(epc) @ brahe.bias_precession_nutation(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317573127, abs=tol)
    assert r[0, 1] == approx(+0.230363826247709, abs=tol)
    assert r[0, 2] == approx(-0.000703332818845, abs=tol)

    assert r[1, 0] == approx(-0.230363798804182, abs=tol)
    assert r[1, 1] == approx(+0.973104570735574, abs=tol)
    assert r[1, 2] == approx(+0.000120888549586, abs=tol)

    assert r[2, 0] == approx(+0.000712264729599, abs=tol)
    assert r[2, 1] == approx(+0.000044385250426, abs=tol)
    assert r[2, 2] == approx(+0.999999745354420, abs=tol)


def test_eci_to_ecef(static_eop):
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    r = brahe.rotation_eci_to_ecef(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317697535, abs=tol)
    assert r[0, 1] == approx(+0.230363826239128, abs=tol)
    assert r[0, 2] == approx(-0.000703163482198, abs=tol)

    assert r[1, 0] == approx(-0.230363800456037, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(+0.000118545366625, abs=tol)

    assert r[2, 0] == approx(+0.000711560162668, abs=tol)
    assert r[2, 1] == approx(+0.000046626403995, abs=tol)
    assert r[2, 2] == approx(+0.999999745754024, abs=tol)


def test_ecef_to_eci(static_eop):
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    r = brahe.rotation_ecef_to_eci(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317697535, abs=tol)
    assert r[0, 1] == approx(-0.230363800456037, abs=tol)
    assert r[0, 2] == approx(+0.000711560162668, abs=tol)

    assert r[1, 0] == approx(+0.230363826239128, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(+0.000046626403995, abs=tol)

    assert r[2, 0] == approx(-0.000703163482198, abs=tol)
    assert r[2, 1] == approx(+0.000118545366625, abs=tol)
    assert r[2, 2] == approx(+0.999999745754024, abs=tol)


def test_position_eci_to_ecef(eop):
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    p_eci = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_ecef = brahe.position_eci_to_ecef(epc, p_eci)

    assert p_eci[0] != p_ecef[0]
    assert p_eci[1] != p_ecef[1]
    assert p_eci[2] != p_ecef[2]


def test_position_ecef_to_eci(eop):
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    p_ecef = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_eci = brahe.position_ecef_to_eci(epc, p_ecef)

    assert p_eci[0] != p_ecef[0]
    assert p_eci[1] != p_ecef[1]
    assert p_eci[2] != p_ecef[2]


def test_state_eci_to_ecef_circular(eop):
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    eci = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)

    # Perform circular transformations
    ecef = brahe.state_eci_to_ecef(epc, eci)
    eci2 = brahe.state_ecef_to_eci(epc, ecef)
    ecef2 = brahe.state_eci_to_ecef(epc, eci2)

    tol = 1e-6
    # Check equivalence of ECI coordinates
    assert eci2[0] == approx(eci[0], abs=tol)
    assert eci2[1] == approx(eci[1], abs=tol)
    assert eci2[2] == approx(eci[2], abs=tol)
    assert eci2[3] == approx(eci[3], abs=tol)
    assert eci2[4] == approx(eci[4], abs=tol)
    assert eci2[5] == approx(eci[5], abs=tol)
    # Check equivalence of ECEF coordinates
    assert ecef2[0] == approx(ecef[0], abs=tol)
    assert ecef2[1] == approx(ecef[1], abs=tol)
    assert ecef2[2] == approx(ecef[2], abs=tol)
    assert ecef2[3] == approx(ecef[3], abs=tol)
    assert ecef2[4] == approx(ecef[4], abs=tol)
    assert ecef2[5] == approx(ecef[5], abs=tol)


def test_rotation_gcrf_to_itrf(static_eop):
    """Test the explicit GCRF -> ITRF transformation"""
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    r = brahe.rotation_gcrf_to_itrf(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317697535, abs=tol)
    assert r[0, 1] == approx(+0.230363826239128, abs=tol)
    assert r[0, 2] == approx(-0.000703163482198, abs=tol)

    assert r[1, 0] == approx(-0.230363800456037, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(+0.000118545366625, abs=tol)

    assert r[2, 0] == approx(+0.000711560162668, abs=tol)
    assert r[2, 1] == approx(+0.000046626403995, abs=tol)
    assert r[2, 2] == approx(+0.999999745754024, abs=tol)


def test_rotation_itrf_to_gcrf(static_eop):
    """Test the explicit ITRF -> GCRF transformation"""
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    r = brahe.rotation_itrf_to_gcrf(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317697535, abs=tol)
    assert r[0, 1] == approx(-0.230363800456037, abs=tol)
    assert r[0, 2] == approx(+0.000711560162668, abs=tol)

    assert r[1, 0] == approx(+0.230363826239128, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(+0.000046626403995, abs=tol)

    assert r[2, 0] == approx(-0.000703163482198, abs=tol)
    assert r[2, 1] == approx(+0.000118545366625, abs=tol)
    assert r[2, 2] == approx(+0.999999745754024, abs=tol)


def test_position_gcrf_to_itrf(eop):
    """Test position transformation from GCRF to ITRF"""
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    p_gcrf = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_itrf = brahe.position_gcrf_to_itrf(epc, p_gcrf)

    assert p_gcrf[0] != p_itrf[0]
    assert p_gcrf[1] != p_itrf[1]
    assert p_gcrf[2] != p_itrf[2]


def test_position_itrf_to_gcrf(eop):
    """Test position transformation from ITRF to GCRF"""
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    p_itrf = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_gcrf = brahe.position_itrf_to_gcrf(epc, p_itrf)

    assert p_gcrf[0] != p_itrf[0]
    assert p_gcrf[1] != p_itrf[1]
    assert p_gcrf[2] != p_itrf[2]


def test_state_gcrf_to_itrf(eop):
    """Test state transformation from GCRF to ITRF"""
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    gcrf = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)

    # Transform to ITRF
    itrf = brahe.state_gcrf_to_itrf(epc, gcrf)

    # Verify transformation occurred
    assert gcrf[0] != itrf[0]
    assert gcrf[1] != itrf[1]
    assert gcrf[2] != itrf[2]


def test_state_itrf_to_gcrf_circular(eop):
    """Test round-trip state transformation GCRF -> ITRF -> GCRF"""
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, brahe.UTC)

    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    gcrf = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)

    # Perform circular transformations
    itrf = brahe.state_gcrf_to_itrf(epc, gcrf)
    gcrf2 = brahe.state_itrf_to_gcrf(epc, itrf)
    itrf2 = brahe.state_gcrf_to_itrf(epc, gcrf2)

    tol = 1e-6
    # Check equivalence of GCRF coordinates
    assert gcrf2[0] == approx(gcrf[0], abs=tol)
    assert gcrf2[1] == approx(gcrf[1], abs=tol)
    assert gcrf2[2] == approx(gcrf[2], abs=tol)
    assert gcrf2[3] == approx(gcrf[3], abs=tol)
    assert gcrf2[4] == approx(gcrf[4], abs=tol)
    assert gcrf2[5] == approx(gcrf[5], abs=tol)
    # Check equivalence of ITRF coordinates
    assert itrf2[0] == approx(itrf[0], abs=tol)
    assert itrf2[1] == approx(itrf[1], abs=tol)
    assert itrf2[2] == approx(itrf[2], abs=tol)
    assert itrf2[3] == approx(itrf[3], abs=tol)
    assert itrf2[4] == approx(itrf[4], abs=tol)
    assert itrf2[5] == approx(itrf[5], abs=tol)


def test_gcrf_itrf_eci_ecef_equivalence(eop):
    """Test that GCRF/ITRF functions are equivalent to ECI/ECEF functions"""
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, brahe.UTC)

    # Test rotation matrices
    r_gcrf_itrf = brahe.rotation_gcrf_to_itrf(epc)
    r_eci_ecef = brahe.rotation_eci_to_ecef(epc)
    assert np.allclose(r_gcrf_itrf, r_eci_ecef)

    r_itrf_gcrf = brahe.rotation_itrf_to_gcrf(epc)
    r_ecef_eci = brahe.rotation_ecef_to_eci(epc)
    assert np.allclose(r_itrf_gcrf, r_ecef_eci)

    # Test position transformations
    p = np.array([brahe.R_EARTH + 500e3, brahe.R_EARTH + 400e3, brahe.R_EARTH + 300e3])

    p_itrf_gcrf = brahe.position_gcrf_to_itrf(epc, p)
    p_ecef_eci = brahe.position_eci_to_ecef(epc, p)
    assert np.allclose(p_itrf_gcrf, p_ecef_eci)

    p_gcrf_itrf = brahe.position_itrf_to_gcrf(epc, p)
    p_eci_ecef = brahe.position_ecef_to_eci(epc, p)
    assert np.allclose(p_gcrf_itrf, p_eci_ecef)

    # Test state transformations
    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    state = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)

    state_itrf_gcrf = brahe.state_gcrf_to_itrf(epc, state)
    state_ecef_eci = brahe.state_eci_to_ecef(epc, state)
    assert np.allclose(state_itrf_gcrf, state_ecef_eci)

    state_gcrf_itrf = brahe.state_itrf_to_gcrf(epc, state_itrf_gcrf)
    state_eci_ecef = brahe.state_ecef_to_eci(epc, state_ecef_eci)
    assert np.allclose(state_gcrf_itrf, state_eci_ecef)


# EME2000 <> GCRF transformation tests
def test_bias_eme2000():
    """Test the bias matrix computation for GCRF to EME2000"""
    r_eme2000 = brahe.bias_eme2000()

    # Independently define expected values
    dξ = -16.6170e-3 * brahe.AS2RAD  # radians
    dη = -6.8192e-3 * brahe.AS2RAD  # radians
    dα = -14.6e-3 * brahe.AS2RAD  # radians

    tol = 1e-9
    assert r_eme2000[0, 0] == approx(1.0 - 0.5 * (dα**2 + dξ**2), abs=tol)
    assert r_eme2000[0, 1] == approx(dα, abs=tol)
    assert r_eme2000[0, 2] == approx(-dξ, abs=tol)

    assert r_eme2000[1, 0] == approx(-dα - dη * dξ, abs=tol)
    assert r_eme2000[1, 1] == approx(1.0 - 0.5 * (dα**2 + dη**2), abs=tol)
    assert r_eme2000[1, 2] == approx(-dη, abs=tol)

    assert r_eme2000[2, 0] == approx(dξ - dη * dα, abs=tol)
    assert r_eme2000[2, 1] == approx(dη + dξ * dα, abs=tol)
    assert r_eme2000[2, 2] == approx(1.0 - 0.5 * (dη**2 + dξ**2), abs=tol)


def test_rotation_gcrf_to_eme2000():
    """Test GCRF to EME2000 rotation matrix matches bias matrix"""
    r_e2g = brahe.rotation_gcrf_to_eme2000()
    r_eme2000 = brahe.bias_eme2000()

    tol = 1e-9
    for i in range(3):
        for j in range(3):
            assert r_e2g[i, j] == approx(r_eme2000[i, j], abs=tol)


def test_rotation_eme2000_to_gcrf():
    """Test EME2000 to GCRF rotation matrix is transpose of inverse"""
    r_g2e = brahe.rotation_eme2000_to_gcrf()
    r_e2g = brahe.rotation_gcrf_to_eme2000().T

    tol = 1e-9
    for i in range(3):
        for j in range(3):
            assert r_g2e[i, j] == approx(r_e2g[i, j], abs=tol)


def test_position_gcrf_to_eme2000():
    """Test position transformation from GCRF to EME2000"""
    p_gcrf = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_eme2000 = brahe.position_gcrf_to_eme2000(p_gcrf)
    r_e2g = brahe.rotation_gcrf_to_eme2000()

    p_eme2000_expected = r_e2g @ p_gcrf

    tol = 1e-9
    assert p_eme2000[0] == approx(p_eme2000_expected[0], abs=tol)
    assert p_eme2000[1] == approx(p_eme2000_expected[1], abs=tol)
    assert p_eme2000[2] == approx(p_eme2000_expected[2], abs=tol)


def test_position_eme2000_to_gcrf():
    """Test position transformation from EME2000 to GCRF"""
    p_eme2000 = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_gcrf = brahe.position_eme2000_to_gcrf(p_eme2000)
    r_g2e = brahe.rotation_eme2000_to_gcrf()

    p_gcrf_expected = r_g2e @ p_eme2000

    tol = 1e-9
    assert p_gcrf[0] == approx(p_gcrf_expected[0], abs=tol)
    assert p_gcrf[1] == approx(p_gcrf_expected[1], abs=tol)
    assert p_gcrf[2] == approx(p_gcrf_expected[2], abs=tol)


def test_state_gcrf_to_eme2000():
    """Test state transformation from GCRF to EME2000"""
    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    gcrf = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)
    eme2000 = brahe.state_gcrf_to_eme2000(gcrf)
    r_e2g = brahe.rotation_gcrf_to_eme2000()

    r_gcrf = gcrf[:3]
    v_gcrf = gcrf[3:]

    p_expected = r_e2g @ r_gcrf
    v_expected = r_e2g @ v_gcrf

    tol = 1e-9
    assert eme2000[0] == approx(p_expected[0], abs=tol)
    assert eme2000[1] == approx(p_expected[1], abs=tol)
    assert eme2000[2] == approx(p_expected[2], abs=tol)
    assert eme2000[3] == approx(v_expected[0], abs=tol)
    assert eme2000[4] == approx(v_expected[1], abs=tol)
    assert eme2000[5] == approx(v_expected[2], abs=tol)


def test_state_eme2000_to_gcrf():
    """Test state transformation from EME2000 to GCRF"""
    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    eme2000 = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)
    gcrf = brahe.state_eme2000_to_gcrf(eme2000)
    r_g2e = brahe.rotation_eme2000_to_gcrf()

    r_eme2000 = eme2000[:3]
    v_eme2000 = eme2000[3:]

    p_expected = r_g2e @ r_eme2000
    v_expected = r_g2e @ v_eme2000

    tol = 1e-9
    assert gcrf[0] == approx(p_expected[0], abs=tol)
    assert gcrf[1] == approx(p_expected[1], abs=tol)
    assert gcrf[2] == approx(p_expected[2], abs=tol)
    assert gcrf[3] == approx(v_expected[0], abs=tol)
    assert gcrf[4] == approx(v_expected[1], abs=tol)
    assert gcrf[5] == approx(v_expected[2], abs=tol)


def test_eme2000_gcrf_roundtrip():
    """Test round-trip state transformation EME2000 -> GCRF -> EME2000"""
    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    eme2000 = brahe.state_osculating_to_cartesian(oe, brahe.AngleFormat.DEGREES)

    # Perform circular transformations
    gcrf = brahe.state_eme2000_to_gcrf(eme2000)
    eme2000_2 = brahe.state_gcrf_to_eme2000(gcrf)
    gcrf_2 = brahe.state_eme2000_to_gcrf(eme2000_2)

    tol = 1e-7
    # Check equivalence of EME2000 coordinates
    assert eme2000_2[0] == approx(eme2000[0], abs=tol)
    assert eme2000_2[1] == approx(eme2000[1], abs=tol)
    assert eme2000_2[2] == approx(eme2000[2], abs=tol)
    assert eme2000_2[3] == approx(eme2000[3], abs=tol)
    assert eme2000_2[4] == approx(eme2000[4], abs=tol)
    assert eme2000_2[5] == approx(eme2000[5], abs=tol)
    # Check equivalence of GCRF coordinates
    assert gcrf_2[0] == approx(gcrf[0], abs=tol)
    assert gcrf_2[1] == approx(gcrf[1], abs=tol)
    assert gcrf_2[2] == approx(gcrf[2], abs=tol)
    assert gcrf_2[3] == approx(gcrf[3], abs=tol)
    assert gcrf_2[4] == approx(gcrf[4], abs=tol)
    assert gcrf_2[5] == approx(gcrf[5], abs=tol)
