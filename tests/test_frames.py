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
    eci = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

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
    gcrf = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

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
    gcrf = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

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
    state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

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
    gcrf = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
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
    eme2000 = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
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
    eme2000 = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

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


# IAU/WGCCRE body rotation model tests


def test_iau_rotation_model_ids():
    ids = brahe.iau_rotation_model_ids()
    assert 499 in ids  # Mars
    assert ids == sorted(ids)


def test_rotation_icrf_to_body_fixed_iau_orthonormal():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    for naif_id in brahe.iau_rotation_model_ids():
        r = brahe.rotation_icrf_to_body_fixed_iau(naif_id, epc)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-10)


def test_rotation_icrf_to_body_fixed_iau_unknown_body_raises():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    with pytest.raises(RuntimeError):
        brahe.rotation_icrf_to_body_fixed_iau(999999, epc)


# Mars reference frame tests (MCI, MCMF)


def test_rotation_mci_to_mcmf_orthonormal():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r = brahe.rotation_mci_to_mcmf(epc)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_rotation_mcmf_to_mci_is_transpose():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_fwd = brahe.rotation_mci_to_mcmf(epc)
    r_inv = brahe.rotation_mcmf_to_mci(epc)
    np.testing.assert_allclose(r_inv, r_fwd.T, atol=1e-12)


def test_position_mci_to_mcmf_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_mci = np.array([brahe.R_MARS + 400e3, 1e6, 2e6])
    x_mcmf = brahe.position_mci_to_mcmf(epc, x_mci)
    x_mci2 = brahe.position_mcmf_to_mci(epc, x_mcmf)
    np.testing.assert_allclose(x_mci2, x_mci, atol=1e-6)


def test_state_mci_to_mcmf_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_mci = np.array([brahe.R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0])
    x_mcmf = brahe.state_mci_to_mcmf(epc, x_mci)
    x_mci2 = brahe.state_mcmf_to_mci(epc, x_mcmf)
    np.testing.assert_allclose(x_mci2, x_mci, atol=1e-6)


def test_state_mci_to_mcmf_transport_term():
    # Velocity of a body-fixed point: numerically differentiate R(t)*r and
    # compare with the analytic transport term. Catches sign/frame errors.
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_inertial = np.array([brahe.R_MARS + 400e3, 1e6, 2e6])
    x = np.array([r_inertial[0], r_inertial[1], r_inertial[2], 0.0, 0.0, 0.0])
    dt = 1.0  # s
    p0 = brahe.position_mci_to_mcmf(epc, r_inertial)
    p1 = brahe.position_mci_to_mcmf(epc + dt, r_inertial)
    v_fd = (p1 - p0) / dt
    v_analytic = brahe.state_mci_to_mcmf(epc, x)[3:6]
    np.testing.assert_allclose(v_analytic, v_fd, atol=1e-2)


def test_mcmf_surface_point_is_stationary():
    # A point rotating with Mars has near-zero MCMF velocity
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_mcmf = np.array([brahe.R_MARS, 0.0, 0.0])
    x_mci = brahe.state_mcmf_to_mci(
        epc, np.array([r_mcmf[0], r_mcmf[1], r_mcmf[2], 0.0, 0.0, 0.0])
    )
    back = brahe.state_mci_to_mcmf(epc, x_mci)
    np.testing.assert_allclose(back[3:6], 0.0, atol=1e-9)


def test_state_eci_to_mci_matches_spk():
    # x_mci = x_eci - state_of_mars_relative_to_earth
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
    # DE kernel first (Moon/Earth legs for the rest of this module), then the
    # satellite kernel: loading mar099s alone would suppress the de440s
    # auto-initialization, which only fires on an empty registry.
    brahe.load_spice_kernel("de440s")
    brahe.load_spice_kernel(
        "mar099s"
    )  # 499 reference leg (transform auto-loads it too)
    offset = brahe.spk_state(brahe.NAIFId.MARS, brahe.NAIFId.EARTH, epc)
    expected = x - offset
    got = brahe.state_eci_to_mci(epc, x)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_state_eci_to_mci_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_eci = np.array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
    x_mci = brahe.state_eci_to_mci(epc, x_eci)
    x_eci2 = brahe.state_mci_to_eci(epc, x_mci)
    np.testing.assert_allclose(x_eci2, x_eci, atol=1e-6)

    p_eci = x_eci[:3]
    p_mci = brahe.position_eci_to_mci(epc, p_eci)
    p_eci2 = brahe.position_mci_to_eci(epc, p_mci)
    np.testing.assert_allclose(p_eci2, p_eci, atol=1e-6)


# Lunar reference frame tests (LCI, LFPA, LFME)


def test_rotation_lfpa_to_lfme_is_small_constant():
    r = brahe.rotation_lfpa_to_lfme()
    angle = np.arccos((np.trace(r) - 1.0) / 2.0)
    assert 4.0e-4 < angle < 6.0e-4
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_rotation_lfme_to_lfpa_is_lfpa_to_lfme_transpose():
    r_pa_to_me = brahe.rotation_lfpa_to_lfme()
    r_me_to_pa = brahe.rotation_lfme_to_lfpa()
    np.testing.assert_allclose(r_me_to_pa, r_pa_to_me.T, atol=1e-15)


def test_rotation_lci_to_lfpa_orthonormal():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r = brahe.rotation_lci_to_lfpa(epc)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_rotation_lfpa_to_lci_is_transpose():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_fwd = brahe.rotation_lci_to_lfpa(epc)
    r_inv = brahe.rotation_lfpa_to_lci(epc)
    np.testing.assert_allclose(r_inv, r_fwd.T, atol=1e-12)


def test_rotation_lci_to_lfpa_matches_pck():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r = brahe.rotation_lci_to_lfpa(epc)
    r_pck = brahe.pck_rotation_matrix(31008, epc)
    np.testing.assert_array_equal(r, r_pck.to_matrix())  # bit-identical: same code path


def test_position_lci_to_lfpa_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_lci = np.array([brahe.R_MOON + 100e3, 1e5, 2e5])
    x_lfpa = brahe.position_lci_to_lfpa(epc, x_lci)
    x_lci2 = brahe.position_lfpa_to_lci(epc, x_lfpa)
    np.testing.assert_allclose(x_lci2, x_lci, atol=1e-6)


def test_state_lci_to_lfpa_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_lci = np.array([brahe.R_MOON + 100e3, 1e5, 2e5, 0.0, 1.6e3, 0.0])
    x_lfpa = brahe.state_lci_to_lfpa(epc, x_lci)
    x_lci2 = brahe.state_lfpa_to_lci(epc, x_lfpa)
    np.testing.assert_allclose(x_lci2, x_lci, atol=1e-6)


def test_state_lci_to_lfpa_transport_term():
    # Same finite-difference pattern as the Mars module: numerically
    # differentiate R(t)*r and compare with the analytic transport term.
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_inertial = np.array([brahe.R_MOON + 100e3, 1e5, 2e5])
    x = np.array([r_inertial[0], r_inertial[1], r_inertial[2], 0.0, 0.0, 0.0])
    dt = 1.0  # s
    p0 = brahe.position_lci_to_lfpa(epc, r_inertial)
    p1 = brahe.position_lci_to_lfpa(epc + dt, r_inertial)
    v_fd = (p1 - p0) / dt
    v_analytic = brahe.state_lci_to_lfpa(epc, x)[3:6]
    np.testing.assert_allclose(v_analytic, v_fd, atol=1e-2)


def test_lfpa_surface_point_is_stationary():
    # A point rotating with the Moon (in the PA frame) has near-zero
    # LFPA velocity.
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_lfpa = np.array([brahe.R_MOON, 0.0, 0.0])
    x_lci = brahe.state_lfpa_to_lci(
        epc, np.array([r_lfpa[0], r_lfpa[1], r_lfpa[2], 0.0, 0.0, 0.0])
    )
    back = brahe.state_lci_to_lfpa(epc, x_lci)
    np.testing.assert_allclose(back[3:6], 0.0, atol=1e-9)


def test_lci_lfme_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_lci = np.array([brahe.R_MOON + 100e3, 1e5, 2e5, 0.0, 1.6e3, 0.0])
    x_lfme = brahe.state_lci_to_lfme(epc, x_lci)
    x_lci2 = brahe.state_lfme_to_lci(epc, x_lfme)
    np.testing.assert_allclose(x_lci2, x_lci, atol=1e-6)

    p_lci = x_lci[:3]
    p_lfme = brahe.position_lci_to_lfme(epc, p_lci)
    p_lci2 = brahe.position_lfme_to_lci(epc, p_lfme)
    np.testing.assert_allclose(p_lci2, p_lci, atol=1e-6)


def test_lfme_surface_point_is_nearly_stationary():
    # A point rotating with the Moon (in the LFME frame) has near-zero
    # LFME velocity, same as the LFPA case (LFME is rigidly offset from
    # LFPA by a constant rotation).
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r_lfme = np.array([brahe.R_MOON, 0.0, 0.0])
    x_lci = brahe.state_lfme_to_lci(
        epc, np.array([r_lfme[0], r_lfme[1], r_lfme[2], 0.0, 0.0, 0.0])
    )
    back = brahe.state_lci_to_lfme(epc, x_lci)
    np.testing.assert_allclose(back[3:6], 0.0, atol=1e-9)


def test_state_eci_to_lci_matches_spk():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, 2e8, 3e8, 1.0, 2.0, 3.0])
    offset = brahe.spk_state(brahe.NAIFId.MOON, brahe.NAIFId.EARTH, epc)
    expected = x - offset
    got = brahe.state_eci_to_lci(epc, x)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_state_eci_to_lci_roundtrip():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_eci = np.array([1e8, 2e8, 3e8, 1.0, 2.0, 3.0])
    x_lci = brahe.state_eci_to_lci(epc, x_eci)
    x_eci2 = brahe.state_lci_to_eci(epc, x_lci)
    np.testing.assert_allclose(x_eci2, x_eci, atol=1e-6)

    p_eci = x_eci[:3]
    p_lci = brahe.position_eci_to_lci(epc, p_eci)
    p_eci2 = brahe.position_lci_to_eci(epc, p_lci)
    np.testing.assert_allclose(p_eci2, p_eci, atol=1e-6)


# ---------------------------------------------------------------------------
# Synodic frames (EMR, SER, GSE)
# ---------------------------------------------------------------------------


def test_reference_frame_synodic_attrs():
    assert brahe.ReferenceFrame.from_string("EMR") == brahe.ReferenceFrame.EMR
    assert brahe.ReferenceFrame.from_string("ser") == brahe.ReferenceFrame.SER
    assert brahe.ReferenceFrame.from_string("GSE") == brahe.ReferenceFrame.GSE
    assert str(brahe.ReferenceFrame.EMR) == "EMR"


def test_state_gcrf_to_emr_moon_on_x_axis():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_moon_gcrf = brahe.spk_state(brahe.NAIFId.MOON, brahe.NAIFId.EARTH, epc)
    x_moon_emr = brahe.state_gcrf_to_emr(epc, x_moon_gcrf)
    assert 3.4e8 < x_moon_emr[0] < 4.1e8
    assert x_moon_emr[1] == approx(0.0, abs=1e-3)
    assert x_moon_emr[2] == approx(0.0, abs=1e-3)
    assert x_moon_emr[4] == approx(0.0, abs=1e-6)
    assert x_moon_emr[5] == approx(0.0, abs=1e-6)

    # Earth sits on -x_hat at the EMB offset (~4.7e6 m).
    x_earth_emr = brahe.state_gcrf_to_emr(epc, np.zeros(6))
    assert -5.5e6 < x_earth_emr[0] < -4.0e6
    assert x_earth_emr[1] == approx(0.0, abs=1e-3)
    assert x_earth_emr[2] == approx(0.0, abs=1e-3)


def test_state_gcrf_to_ser_earth_position():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_earth_ser = brahe.state_gcrf_to_ser(epc, np.zeros(6))
    d = np.linalg.norm(brahe.spk_state(brahe.NAIFId.EARTH, brahe.NAIFId.SUN, epc)[:3])
    expected_x = d * brahe.GM_SUN / (brahe.GM_SUN + brahe.GM_EARTH)
    assert x_earth_ser[0] == approx(expected_x, abs=1.0)
    assert x_earth_ser[1] == approx(0.0, abs=1e-2)
    assert x_earth_ser[2] == approx(0.0, abs=1e-2)

    # The Sun sits on -x_hat at the small SEB offset (~4.5e5 m).
    x_sun_gcrf = brahe.spk_state(brahe.NAIFId.SUN, brahe.NAIFId.EARTH, epc)
    x_sun_ser = brahe.state_gcrf_to_ser(epc, x_sun_gcrf)
    assert -6.0e5 < x_sun_ser[0] < -3.0e5
    assert x_sun_ser[1] == approx(0.0, abs=1e-2)
    assert x_sun_ser[2] == approx(0.0, abs=1e-2)


def test_state_gcrf_to_gse_sun_on_x_axis():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_sun_gcrf = brahe.spk_state(brahe.NAIFId.SUN, brahe.NAIFId.EARTH, epc)
    x_sun_gse = brahe.state_gcrf_to_gse(epc, x_sun_gcrf)
    d = np.linalg.norm(x_sun_gcrf[:3])
    assert x_sun_gse[0] == approx(d, abs=1e-3)
    assert x_sun_gse[1] == approx(0.0, abs=1e-2)
    assert x_sun_gse[2] == approx(0.0, abs=1e-2)


def test_gse_z_axis_near_ecliptic_pole():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    s = brahe.rotation_gcrf_to_gse(epc)
    angle_deg = np.degrees(np.arccos(s[2, 2]))
    assert abs(angle_deg - 23.439) < 0.5


def test_synodic_pairwise_roundtrips():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    for to_frame, from_frame in [
        (brahe.state_gcrf_to_emr, brahe.state_emr_to_gcrf),
        (brahe.state_gcrf_to_ser, brahe.state_ser_to_gcrf),
        (brahe.state_gcrf_to_gse, brahe.state_gse_to_gcrf),
    ]:
        x_back = from_frame(epc, to_frame(epc, x))
        np.testing.assert_allclose(x_back[:3], x[:3], atol=1e-2)
        np.testing.assert_allclose(x_back[3:], x[3:], atol=1e-7)

    x3 = np.array([1e8, -2e8, 5e7])
    for to_frame, from_frame in [
        (brahe.position_gcrf_to_emr, brahe.position_emr_to_gcrf),
        (brahe.position_gcrf_to_ser, brahe.position_ser_to_gcrf),
        (brahe.position_gcrf_to_gse, brahe.position_gse_to_gcrf),
    ]:
        np.testing.assert_allclose(from_frame(epc, to_frame(epc, x3)), x3, atol=1e-2)


def test_synodic_router_matches_pairwise():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    for frame, pairwise in [
        (brahe.ReferenceFrame.EMR, brahe.state_gcrf_to_emr(epc, x)),
        (brahe.ReferenceFrame.SER, brahe.state_gcrf_to_ser(epc, x)),
        (brahe.ReferenceFrame.GSE, brahe.state_gcrf_to_gse(epc, x)),
    ]:
        via_router = brahe.state_frame_to_frame(
            brahe.ReferenceFrame.GCRF, frame, epc, x
        )
        np.testing.assert_allclose(via_router, pairwise, atol=1e-9)


def test_router_lci_to_emr():
    # TP §4.6.3: Moon-centered inertial to EMR. The Moon (LCI origin) must
    # land on EMR's +x_hat axis with zero transverse velocity.
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x_moon_emr = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.LCI, brahe.ReferenceFrame.EMR, epc, np.zeros(6)
    )
    assert 3.4e8 < x_moon_emr[0] < 4.1e8
    assert x_moon_emr[1] == approx(0.0, abs=1e-3)
    assert x_moon_emr[2] == approx(0.0, abs=1e-3)
    assert x_moon_emr[4] == approx(0.0, abs=1e-6)
    assert x_moon_emr[5] == approx(0.0, abs=1e-6)


# ReferenceFrame router tests


def test_reference_frame_from_string_aliases():
    assert brahe.ReferenceFrame.from_string("ECI") == brahe.ReferenceFrame.GCRF
    assert brahe.ReferenceFrame.from_string("ECEF") == brahe.ReferenceFrame.ITRF
    assert brahe.ReferenceFrame.from_string("eci") == brahe.ReferenceFrame.GCRF
    assert brahe.ReferenceFrame.from_string("LFPA") == brahe.ReferenceFrame.LFPA
    with pytest.raises(ValueError):
        brahe.ReferenceFrame.from_string("bogus")


def test_body_fixed_custom_frame_round_trip():
    """Mirrors test_body_fixed_custom_router_round_trip: a Python rotation
    callback routes through the frame router with a numerically derived
    transport term."""
    t0 = brahe.Epoch.from_date(2024, 3, 1, brahe.TimeSystem.TDB)
    rate = 5.0e-4

    def spin(epc):
        theta = rate * (epc - t0)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    brahe.register_custom_frame(1042, spin)
    inertial = brahe.ReferenceFrame.BodyCenteredICRF(-20001)
    fixed = brahe.ReferenceFrame.BodyFixedCustom(-20001, 1042)

    epc = t0 + 600.0
    x = np.array([7.0e5, -2.0e5, 3.0e5, 10.0, 25.0, -5.0])

    x_fixed = brahe.state_frame_to_frame(inertial, fixed, epc, x)

    # Analytic rotating-frame state: r_b = R r, v_b = R v - w x r_b.
    r_mat = spin(epc)
    r_b = r_mat @ x[:3]
    v_b = r_mat @ x[3:] - np.cross(np.array([0.0, 0.0, rate]), r_b)
    np.testing.assert_allclose(x_fixed[:3], r_b, atol=1e-6)
    np.testing.assert_allclose(x_fixed[3:], v_b, atol=1e-4)

    x_back = brahe.state_frame_to_frame(fixed, inertial, epc, x_fixed)
    np.testing.assert_allclose(x_back, x, atol=1e-6)

    assert brahe.unregister_custom_frame(1042)
    assert not brahe.unregister_custom_frame(1042)


def test_body_fixed_custom_frame_explicit_omega():
    """An explicit omega callback is honored (exact transport term)."""
    t0 = brahe.Epoch.from_date(2024, 3, 1, brahe.TimeSystem.TDB)
    rate = 2.0e-4

    def spin(epc):
        theta = rate * (epc - t0)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    brahe.register_custom_frame(1043, spin, lambda epc: np.array([0.0, 0.0, rate]))
    inertial = brahe.ReferenceFrame.BodyCenteredICRF(-20002)
    fixed = brahe.ReferenceFrame.BodyFixedCustom(-20002, 1043)

    epc = t0 + 300.0
    x = np.array([7.0e5, -2.0e5, 3.0e5, 10.0, 25.0, -5.0])
    x_fixed = brahe.state_frame_to_frame(inertial, fixed, epc, x)

    r_mat = spin(epc)
    r_b = r_mat @ x[:3]
    v_b = r_mat @ x[3:] - np.cross(np.array([0.0, 0.0, rate]), r_b)
    np.testing.assert_allclose(x_fixed[:3], r_b, atol=1e-6)
    np.testing.assert_allclose(x_fixed[3:], v_b, atol=1e-9)

    assert brahe.unregister_custom_frame(1043)


def test_reference_frame_class_aliases():
    """ECI/ECEF class attributes alias GCRF/ITRF."""
    assert brahe.ReferenceFrame.ECI == brahe.ReferenceFrame.GCRF
    assert brahe.ReferenceFrame.ECEF == brahe.ReferenceFrame.ITRF


def test_reference_frame_str():
    assert str(brahe.ReferenceFrame.GCRF) == "GCRF"
    assert str(brahe.ReferenceFrame.LFPA) == "LFPA"


def test_reference_frame_generic_variants_equal_named():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    a = brahe.rotation_frame_to_frame(
        brahe.ReferenceFrame.MCI, brahe.ReferenceFrame.MCMF, epc
    )
    b = brahe.rotation_frame_to_frame(
        brahe.ReferenceFrame.BodyCenteredICRF(4),
        brahe.ReferenceFrame.BodyFixedIAU(499),
        epc,
    )
    np.testing.assert_array_equal(a, b)


def test_body_fixed_pck_generic_variant_equals_lfpa():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    # Force the moon_pa_de440 PCK to be loaded before the generic-variant
    # path queries it directly (state_icrf_to_pck_body does not auto-load).
    brahe.rotation_lci_to_lfpa(epc)

    via_lfpa = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.LFPA, epc, x
    )
    via_pck = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.BodyFixedPCK(301, 31008), epc, x
    )
    np.testing.assert_array_equal(via_pck, via_lfpa)


def test_rotation_frame_to_frame_same_frame_is_identity():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r = brahe.rotation_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.GCRF, epc
    )
    np.testing.assert_array_equal(r, np.eye(3))


def test_position_frame_to_frame_same_frame_is_identity():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1.0, 2.0, 3.0])
    out = brahe.position_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.GCRF, epc, x
    )
    np.testing.assert_array_equal(out, x)


def test_router_matches_pairwise_gcrf_itrf(eop):
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    x = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)

    via_router = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.ITRF, epc, x
    )
    pairwise = brahe.state_gcrf_to_itrf(epc, x)
    np.testing.assert_array_equal(via_router, pairwise)


def test_router_matches_pairwise_eci_lci_and_lfpa():
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])

    via_router = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.LCI, epc, x
    )
    pairwise = brahe.state_eci_to_lci(epc, x)
    np.testing.assert_allclose(via_router, pairwise, atol=1e-9)

    x_lci = via_router
    via_router_lfpa = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.LCI, brahe.ReferenceFrame.LFPA, epc, x_lci
    )
    pairwise_lfpa = brahe.state_lci_to_lfpa(epc, x_lci)
    np.testing.assert_array_equal(via_router_lfpa, pairwise_lfpa)  # bit-identical

    via_router_composed = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.LFPA, epc, x
    )
    composed = brahe.state_lci_to_lfpa(epc, brahe.state_eci_to_lci(epc, x))
    np.testing.assert_allclose(via_router_composed, composed, atol=1e-9)


def test_router_roundtrip_all_pairs(eop):
    frames = [
        brahe.ReferenceFrame.GCRF,
        brahe.ReferenceFrame.ITRF,
        brahe.ReferenceFrame.EME2000,
        brahe.ReferenceFrame.LCI,
        brahe.ReferenceFrame.LFPA,
        brahe.ReferenceFrame.LFME,
        brahe.ReferenceFrame.MCI,
        brahe.ReferenceFrame.MCMF,
        brahe.ReferenceFrame.EMBI,
        brahe.ReferenceFrame.SSBI,
    ]
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    # Position tolerance is looser than a same-magnitude round trip would
    # suggest because some pairs (e.g. MCI <-> EME2000) compose the
    # approximate (not exactly orthogonal to machine precision) bias_eme2000
    # rotation with the ~3.3e11 m Earth-Mars SPK center offset; see the
    # matching Rust test for details.
    for a in frames:
        for b in frames:
            there = brahe.state_frame_to_frame(a, b, epc, x)
            back = brahe.state_frame_to_frame(b, a, epc, there)
            np.testing.assert_allclose(back[:3], x[:3], atol=1e-2)
            np.testing.assert_allclose(back[3:6], x[3:6], atol=1e-6)


def test_state_frame_to_frame_roundtrip_lci(eop):
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    y = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.LCI, epc, x
    )
    x2 = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.LCI, brahe.ReferenceFrame.GCRF, epc, y
    )
    np.testing.assert_allclose(x2, x, atol=1e-4)


def test_body_fixed_iau_translation_auto_loads_satellite_kernel():
    """Mirrors test_body_fixed_iau_translation_auto_loads_satellite_kernel.

    BodyFixedIAU(499) is centered on Mars itself (NAIF 499); the mar099s
    satellite ephemeris kernel is auto-loaded for the body-center leg, so the
    translated transform succeeds and agrees with MCMF."""
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
    via_iau = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.BodyFixedIAU(499), epc, x
    )
    via_mcmf = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.MCMF, epc, x
    )
    np.testing.assert_allclose(via_iau, via_mcmf, atol=1e-9)


# ------------------------------------------------------------------------
# Offline router mirrors (no kernels / EOP required)
# ------------------------------------------------------------------------


def test_rotation_frame_to_frame_same_center_eme2000():
    """Mirrors test_rotation_frame_to_frame_same_center_eme2000: GCRF <-> EME2000
    is a same-center, EOP-free constant bias rotation."""
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    r = brahe.rotation_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.EME2000, epc
    )
    np.testing.assert_array_equal(r, brahe.rotation_gcrf_to_eme2000())
    r_inv = brahe.rotation_frame_to_frame(
        brahe.ReferenceFrame.EME2000, brahe.ReferenceFrame.GCRF, epc
    )
    np.testing.assert_allclose(r_inv @ r, np.eye(3), atol=1e-12)


def test_state_frame_to_frame_same_center_eme2000_no_spk():
    """Mirrors test_state_frame_to_frame_same_center_eme2000_no_spk: same-center
    GCRF <-> EME2000 skips translation (no SPK) and needs no EOP."""
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    x = np.array([brahe.R_EARTH + 500e3, 1e5, 2e5, 1.0, 7.5e3, 0.5e3])
    via_router = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.GCRF, brahe.ReferenceFrame.EME2000, epc, x
    )
    np.testing.assert_array_equal(via_router, brahe.state_gcrf_to_eme2000(x))
    back = brahe.state_frame_to_frame(
        brahe.ReferenceFrame.EME2000, brahe.ReferenceFrame.GCRF, epc, via_router
    )
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_router_body_fixed_iau_same_center_no_kernels():
    """Mirrors test_router_body_fixed_iau_same_center_no_kernels:
    BodyCenteredICRF(id) <-> BodyFixedIAU(id) is a same-center, kernel-free
    rotation-only + transport path (IAU analytic model)."""
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    icrf = brahe.ReferenceFrame.BodyCenteredICRF(599)
    fixed = brahe.ReferenceFrame.BodyFixedIAU(599)
    x = np.array([7.0e7, -2.0e7, 3.0e7, 10.0, 25.0, -5.0])

    x_fixed = brahe.state_frame_to_frame(icrf, fixed, epc, x)
    x_back = brahe.state_frame_to_frame(fixed, icrf, epc, x_fixed)
    np.testing.assert_allclose(x_back, x, atol=1e-6)

    p = np.array([7.0e7, -2.0e7, 3.0e7])
    p_fixed = brahe.position_frame_to_frame(icrf, fixed, epc, p)
    p_back = brahe.position_frame_to_frame(fixed, icrf, epc, p_fixed)
    np.testing.assert_allclose(p_back, p, atol=1e-6)


def test_router_errors_on_unsupported_iau_body():
    """Mirrors test_router_errors_on_unsupported_iau_body: a non-identity path
    through an unsupported IAU body surfaces the rotation-model lookup error."""
    epc = brahe.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, brahe.UTC)
    with pytest.raises(RuntimeError):
        brahe.rotation_frame_to_frame(
            brahe.ReferenceFrame.BodyCenteredICRF(999999),
            brahe.ReferenceFrame.BodyFixedIAU(999999),
            epc,
        )
