import pytest
import brahe
import numpy as np
from pytest import approx


def test_position_geodetic(eop):
    tol = 1.0e-7

    # Test known position conversions
    geod1 = np.array([0.0, 0.0, 0.0])
    ecef1 = brahe.position_geodetic_to_ecef(geod1, brahe.AngleFormat.DEGREES)

    assert ecef1[0] == approx(brahe.WGS84_A, abs=tol)
    assert ecef1[1] == approx(0.0, abs=tol)
    assert ecef1[2] == approx(0.0, abs=tol)

    geod2 = np.array([90.0, 0.0, 0.0])
    ecef2 = brahe.position_geodetic_to_ecef(geod2, brahe.AngleFormat.DEGREES)

    assert ecef2[0] == approx(0.0, abs=tol)
    assert ecef2[1] == approx(brahe.WGS84_A, abs=tol)
    assert ecef2[2] == approx(0.0, abs=tol)

    geod3 = np.array([0.0, 90.0, 0.0])
    ecef3 = brahe.position_geodetic_to_ecef(geod3, brahe.AngleFormat.DEGREES)

    assert ecef3[0] == approx(0.0, abs=tol)
    assert ecef3[1] == approx(0.0, abs=tol)
    assert ecef3[2] == approx(brahe.WGS84_A * (1.0 - brahe.WGS84_F), abs=tol)

    # Test two input format
    geod = np.array([0.0, 0.0, 0.0])
    ecef = brahe.position_geodetic_to_ecef(geod, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(brahe.WGS84_A, abs=tol)
    assert ecef[1] == approx(0.0, abs=tol)
    assert ecef[2] == approx(0.0, abs=tol)

    geod = np.array([90.0, 0.0, 0.0])
    ecef = brahe.position_geodetic_to_ecef(geod, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(0.0, abs=tol)
    assert ecef[1] == approx(brahe.WGS84_A, abs=tol)
    assert ecef[2] == approx(0.0, abs=tol)

    geod = np.array([0.0, 90.0, 0.0])
    ecef = brahe.position_geodetic_to_ecef(geod, brahe.AngleFormat.DEGREES)

    assert ecef[0] == approx(0.0, abs=tol)
    assert ecef[1] == approx(0.0, abs=tol)
    assert ecef[2] == approx(brahe.WGS84_A * (1.0 - brahe.WGS84_F), abs=tol)

    # Test circularity
    geod4 = brahe.position_ecef_to_geodetic(ecef1, brahe.AngleFormat.DEGREES)
    geod5 = brahe.position_ecef_to_geodetic(ecef2, brahe.AngleFormat.DEGREES)
    geod6 = brahe.position_ecef_to_geodetic(ecef3, brahe.AngleFormat.DEGREES)

    assert geod4[0] == approx(geod1[0], abs=tol)
    assert geod4[1] == approx(geod1[1], abs=tol)
    assert geod4[2] == approx(geod1[2], abs=tol)

    assert geod5[0] == approx(geod2[0], abs=tol)
    assert geod5[1] == approx(geod2[1], abs=tol)
    assert geod5[2] == approx(geod2[2], abs=tol)

    assert geod6[0] == approx(geod3[0], abs=tol)
    assert geod6[1] == approx(geod3[1], abs=tol)
    assert geod6[2] == approx(geod3[2], abs=tol)

    geod = np.array([77.875000, 20.975200, 0.000000])
    ecef = brahe.position_geodetic_to_ecef(geod, brahe.AngleFormat.DEGREES)
    geodc = brahe.position_ecef_to_geodetic(ecef, brahe.AngleFormat.DEGREES)
    assert geod[0] == approx(geodc[0], abs=tol)
    assert geod[1] == approx(geodc[1], abs=tol)
    assert geod[2] == approx(geodc[2], abs=tol)


@pytest.mark.parametrize("lat", [90.1, -90.1])
def test_geodetic_failure(eop, lat):
    # Test Error Condition
    with pytest.raises(ValueError):
        brahe.position_geodetic_to_ecef(
            np.array([0.0, lat, 0.0]), brahe.AngleFormat.DEGREES
        )
