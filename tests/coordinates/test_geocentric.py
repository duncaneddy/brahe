import numpy as np
import pytest
from pytest import approx

import brahe


def test_position_geocentric(eop):
    tol = 1.0e-7

    # Test known position conversions
    geoc1 = np.array([0.0, 0.0, 0.0])
    ecef1 = brahe.position_geocentric_to_ecef(geoc1, brahe.AngleFormat.DEGREES)

    assert ecef1[0] == approx(brahe.WGS84_A, abs=tol)
    assert ecef1[1] == approx(0.0, abs=tol)
    assert ecef1[2] == approx(0.0, abs=tol)

    geoc2 = np.array([90.0, 0.0, 0.0])
    ecef2 = brahe.position_geocentric_to_ecef(geoc2, brahe.AngleFormat.DEGREES)

    assert ecef2[0] == approx(0.0, abs=tol)
    assert ecef2[1] == approx(brahe.WGS84_A, abs=tol)
    assert ecef2[2] == approx(0.0, abs=tol)

    geoc3 = np.array([0.0, 90.0, 0.0])
    ecef3 = brahe.position_geocentric_to_ecef(geoc3, brahe.AngleFormat.DEGREES)

    assert ecef3[0] == approx(0.0, abs=tol)
    assert ecef3[1] == approx(0.0, abs=tol)
    assert ecef3[2] == approx(brahe.WGS84_A, abs=tol)

    # Test two-input format
    geoc = np.array([0.0, 0.0, 0.0])
    ecef = brahe.position_geocentric_to_ecef(geoc, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(brahe.WGS84_A, abs=tol)
    assert ecef[1] == approx(0.0, abs=tol)
    assert ecef[2] == approx(0.0, abs=tol)

    geoc = np.array([90.0, 0.0, 0.0])
    ecef = brahe.position_geocentric_to_ecef(geoc, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(0.0, abs=tol)
    assert ecef[1] == approx(brahe.WGS84_A, abs=tol)
    assert ecef[2] == approx(0.0, abs=tol)

    geoc = np.array([0.0, 90.0, 0.0])
    ecef = brahe.position_geocentric_to_ecef(geoc, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(0.0, abs=tol)
    assert ecef[1] == approx(0.0, abs=tol)
    assert ecef[2] == approx(brahe.WGS84_A, abs=tol)

    # Test circularity
    geoc4 = brahe.position_ecef_to_geocentric(ecef1, brahe.AngleFormat.DEGREES)
    geoc5 = brahe.position_ecef_to_geocentric(ecef2, brahe.AngleFormat.DEGREES)
    geoc6 = brahe.position_ecef_to_geocentric(ecef3, brahe.AngleFormat.DEGREES)

    assert geoc4[0] == approx(geoc1[0], abs=tol)
    assert geoc4[1] == approx(geoc1[1], abs=tol)
    assert geoc4[2] == approx(geoc1[2], abs=tol)

    assert geoc5[0] == approx(geoc2[0], abs=tol)
    assert geoc5[1] == approx(geoc2[1], abs=tol)
    assert geoc5[2] == approx(geoc2[2], abs=tol)

    assert geoc6[0] == approx(geoc3[0], abs=tol)
    assert geoc6[1] == approx(geoc3[1], abs=tol)
    assert geoc6[2] == approx(geoc3[2], abs=tol)

    # Random point circularity
    geoc = np.array([77.875000, 20.975200, 0.000000])
    ecef = brahe.position_geocentric_to_ecef(geoc, brahe.AngleFormat.DEGREES)
    geocc = brahe.position_ecef_to_geocentric(ecef, brahe.AngleFormat.DEGREES)
    assert geoc[0] == approx(geocc[0], abs=tol)
    assert geoc[1] == approx(geocc[1], abs=tol)
    assert geoc[2] == approx(geocc[2], abs=tol)


@pytest.mark.parametrize("lat", [90.1, -90.1])
def test_geocentric_failure(eop, lat):
    # Test Error Condition
    with pytest.raises(ValueError):
        brahe.position_geocentric_to_ecef(
            np.array([0.0, lat, 0.0]), brahe.AngleFormat.DEGREES
        )


def test_batch_geocentric_ecef():
    sites = np.array(
        [
            [-122.4, 37.8, brahe.R_EARTH],
            [151.2, -33.9, brahe.R_EARTH + 100.0],
            [0.0, 0.0, brahe.R_EARTH + 500e3],
        ]
    )
    ecef = brahe.position_geocentric_to_ecef(sites, brahe.AngleFormat.DEGREES)
    back = brahe.position_ecef_to_geocentric(ecef, brahe.AngleFormat.DEGREES)
    assert ecef.shape == (3, 3)
    for i in range(3):
        np.testing.assert_array_equal(
            ecef[i],
            brahe.position_geocentric_to_ecef(sites[i], brahe.AngleFormat.DEGREES),
        )
        np.testing.assert_array_equal(
            back[i],
            brahe.position_ecef_to_geocentric(ecef[i], brahe.AngleFormat.DEGREES),
        )
    np.testing.assert_allclose(back, sites, atol=1e-6)
    with pytest.raises(ValueError):
        brahe.position_geocentric_to_ecef(
            np.array([[0.0, 95.0, 0.0]]), brahe.AngleFormat.DEGREES
        )
