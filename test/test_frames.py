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
    brahe.set_global_eop_provider_from_static_provider(eop)

def test_bias_precession_nutation(static_eop):
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0, 0.0, "UTC")

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
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, "UTC")

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
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, "UTC")

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
    epc = brahe.Epoch.from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, "UTC")

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
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, "UTC")

    p_eci = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_ecef = brahe.position_eci_to_ecef(epc, p_eci)

    assert p_eci[0] != p_ecef[0]
    assert p_eci[1] != p_ecef[1]
    assert p_eci[2] != p_ecef[2]

def test_position_ecef_to_eci(eop):
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, "UTC")

    p_ecef = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0])

    p_eci = brahe.position_ecef_to_eci(epc, p_ecef)

    assert p_eci[0] != p_ecef[0]
    assert p_eci[1] != p_ecef[1]
    assert p_eci[2] != p_ecef[2]

def test_state_eci_to_ecef_circular(eop):
    epc = brahe.Epoch.from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, "UTC")

    oe = np.array([brahe.R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0])
    eci = brahe.state_osculating_to_cartesian(oe, True)

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