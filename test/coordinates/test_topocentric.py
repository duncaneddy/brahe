import pytest
import math
import brahe
import numpy as np
from pytest import approx
from brahe import AngleFormat

def test_rotation_ellipsoid_to_enz():
    # Epsilon Tolerance
    tol = np.finfo(float).eps

    # Test aligned coordinates
    x_sta = np.array([0.0, 0.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_enz(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is ENZ Z-dir
    assert rot1[0,0] == approx(0.0, abs = tol)
    assert rot1[1,0] == approx(0.0, abs = tol)
    assert rot1[2,0] == approx(1.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is ENZ E-dir
    assert rot1[0,1] == approx(1.0, abs = tol)
    assert rot1[1,1] == approx(0.0, abs = tol)
    assert rot1[2,1] == approx(0.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is ENZ N-dir
    assert rot1[0,2] == approx(0.0, abs = tol)
    assert rot1[1,2] == approx(1.0, abs = tol)
    assert rot1[2,2] == approx(0.0, abs = tol)

    assert np.linalg.det(rot1) == approx(1.0, abs = tol)

    # Test 90 degree longitude
    x_sta = np.array([90.0, 0.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_enz(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is ENZ -E-dir
    assert rot1[0,0] == approx(-1.0, abs = tol)
    assert rot1[1,0] == approx(0.0, abs = tol)
    assert rot1[2,0] == approx(0.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is ENZ Z-dir
    assert rot1[0,1] == approx(0.0, abs = tol)
    assert rot1[1,1] == approx(0.0, abs = tol)
    assert rot1[2,1] == approx(1.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is ENZ N-dir
    assert rot1[0,2] == approx(0.0, abs = tol)
    assert rot1[1,2] == approx(1.0, abs = tol)
    assert rot1[2,2] == approx(0.0, abs = tol)

    # assert rot1.de ==(rminant(), 1.0, abs = tol)

    # Test 90 degree latitude
    x_sta = np.array([00.0, 90.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_enz(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is ENZ -N-dir
    assert rot1[0,0] == approx(0.0, abs = tol)
    assert rot1[1,0] == approx(-1.0, abs = tol)
    assert rot1[2,0] == approx(0.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is ENZ E-dir
    assert rot1[0,1] == approx(1.0, abs = tol)
    assert rot1[1,1] == approx(0.0, abs = tol)
    assert rot1[2,1] == approx(0.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is ENZ Z-dir
    assert rot1[0,2] == approx(0.0, abs = tol)
    assert rot1[1,2] == approx(0.0, abs = tol)
    assert rot1[2,2] == approx(1.0, abs = tol)

    assert np.linalg.det(rot1) == approx(1.0, abs = tol)

def test_rotation_enz_to_ellipsoid():
    tol = np.finfo(float).eps

    x_sta = np.array([42.1, 53.9, 100.0])
    rot = brahe.rotation_ellipsoid_to_enz(x_sta, AngleFormat.DEGREES)
    rot_t = brahe.rotation_enz_to_ellipsoid(x_sta, AngleFormat.DEGREES)

    r = rot @ rot_t

    # Confirm identity
    assert r[0,0] == approx(1.0, abs = tol)
    assert r[0,1] == approx(0.0, abs = tol)
    assert r[0,2] == approx(0.0, abs = tol)
    assert r[1,0] == approx(0.0, abs = tol)
    assert r[1,1] == approx(1.0, abs = tol)
    assert r[1,2] == approx(0.0, abs = tol)
    assert r[2,0] == approx(0.0, abs = tol)
    assert r[2,1] == approx(0.0, abs = tol)
    assert r[2,2] == approx(1.0, abs = tol)

def test_relative_position_ecef_to_enz():
    tol = np.finfo(float).eps

    # 100m Overhead
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH + 100.0, 0.0, 0.0])

    r_enz = brahe.relative_position_ecef_to_enz(x_sta, r_ecef, "Geocentric")

    assert r_enz[0] == approx(0.0, abs = tol)
    assert r_enz[1] == approx(0.0, abs = tol)
    assert r_enz[2] == approx(100.0, abs = tol)

    # 100m North
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH, 0.0, 100.0])

    r_enz = brahe.relative_position_ecef_to_enz(x_sta, r_ecef, "Geocentric")

    assert r_enz[0] == approx(0.0, abs = tol)
    assert r_enz[1] == approx(100.0, abs = tol)
    assert r_enz[2] == approx(0.0, abs = tol)

    # 100m East
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH, 100.0, 0.0])

    r_enz = brahe.relative_position_ecef_to_enz(x_sta, r_ecef, "Geocentric")

    assert r_enz[0] == approx(100.0, abs = tol)
    assert r_enz[1] == approx(0.0, abs = tol)
    assert r_enz[2] == approx(0.0, abs = tol)

    # Confirm higher latitude and longitude is (+E, +N, -Z)
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    x_geoc = np.array([0.5, 0.5, 0.0])
    r_ecef = brahe.position_geocentric_to_ecef(x_geoc, brahe.AngleFormat.DEGREES)

    r_enz_geoc = brahe.relative_position_ecef_to_enz(x_sta, r_ecef, "Geocentric")

    assert r_enz_geoc[0] > 0.0
    assert r_enz_geoc[1] > 0.0
    assert r_enz_geoc[2] < 0.0

    # Confirm difference in geocentric and geodetic conversions
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    x_geod = np.array([0.5, 0.5, 0.0])
    r_ecef = brahe.position_geodetic_to_ecef(x_geod, brahe.AngleFormat.DEGREES)

    r_enz_geod = brahe.relative_position_ecef_to_enz(x_sta, r_ecef, "Geodetic")

    assert r_enz_geod[0] > 0.0
    assert r_enz_geod[1] > 0.0
    assert r_enz_geod[2] < 0.0

    for i in range(0,3):
        assert r_enz_geoc[i] != r_enz_geod[i]

def test_relative_position_enz_to_ecef():
    tol = np.finfo(float).eps

    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_enz = np.array([0.0, 0.0, 100.0])

    r_ecef = brahe.relative_position_enz_to_ecef(x_sta, r_enz, "Geodetic")

    assert r_ecef[0] == approx(brahe.R_EARTH + 100.0, abs = tol)
    assert r_ecef[1] == approx(0.0, abs = tol)
    assert r_ecef[2] == approx(0.0, abs = tol)

def test_rotation_ellipsoid_to_sez():
                                    # Epsilon Tolerance
    tol = np.finfo(float).eps

    # Test aligned coordinates
    x_sta = np.array([0.0, 0.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_sez(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is SEZ Z-dir
    assert rot1[0,0] == approx(0.0, abs = tol)
    assert rot1[1,0] == approx(0.0, abs = tol)
    assert rot1[2,0] == approx(1.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is SEZ E-dir
    assert rot1[0,1] == approx(0.0, abs = tol)
    assert rot1[1,1] == approx(1.0, abs = tol)
    assert rot1[2,1] == approx(0.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is SEZ -S-dir
    assert rot1[0,2] == approx(-1.0, abs = tol)
    assert rot1[1,2] == approx(0.0, abs = tol)
    assert rot1[2,2] == approx(0.0, abs = tol)

    assert np.linalg.det(rot1) == approx(1.0, abs = tol)

    # Test 90 degree longitude
    x_sta = np.array([90.0, 0.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_sez(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is SEZ -E-dir
    assert rot1[0,0] == approx(0.0, abs = tol)
    assert rot1[1,0] == approx(-1.0, abs = tol)
    assert rot1[2,0] == approx(0.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is SEZ Z-dir
    assert rot1[0,1] == approx(0.0, abs = tol)
    assert rot1[1,1] == approx(0.0, abs = tol)
    assert rot1[2,1] == approx(1.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is SEZ -S-dir
    assert rot1[0,2] == approx(-1.0, abs = tol)
    assert rot1[1,2] == approx(0.0, abs = tol)
    assert rot1[2,2] == approx(0.0, abs = tol)

    assert np.linalg.det(rot1) == approx(1.0, abs = tol)

    # Test 90 degree latitude
    x_sta = np.array([00.0, 90.0, 0.0])
    rot1 = brahe.rotation_ellipsoid_to_sez(x_sta, AngleFormat.DEGREES)

    # ECEF input X - [1, 0, 0] - Expected output is SEZ S-dir
    assert rot1[0,0] == approx(1.0, abs = tol)
    assert rot1[1,0] == approx(0.0, abs = tol)
    assert rot1[2,0] == approx(0.0, abs = tol)

    # ECEF input Y - [0, 1, 0] - Expected output is SEZ E-dir
    assert rot1[0,1] == approx(0.0, abs = tol)
    assert rot1[1,1] == approx(1.0, abs = tol)
    assert rot1[2,1] == approx(0.0, abs = tol)

    # ECEF input Z - [0, 0, 1] - Expected output is SEZ Z-dir
    assert rot1[0,2] == approx(0.0, abs = tol)
    assert rot1[1,2] == approx(0.0, abs = tol)
    assert rot1[2,2] == approx(1.0, abs = tol)

    assert np.linalg.det(rot1) == approx(1.0, abs = tol)

def test_rotation_sez_to_ellipsoid():
    tol = np.finfo(float).eps

    x_sta = np.array([42.1, 53.9, 100.0])
    rot = brahe.rotation_ellipsoid_to_sez(x_sta, AngleFormat.DEGREES)
    rot_t = brahe.rotation_sez_to_ellipsoid(x_sta, AngleFormat.DEGREES)

    r = rot @ rot_t

    # Confirm identity
    assert r[0,0] == approx(1.0, abs = tol)
    assert r[0,1] == approx(0.0, abs = tol)
    assert r[0,2] == approx(0.0, abs = tol)
    assert r[1,0] == approx(0.0, abs = tol)
    assert r[1,1] == approx(1.0, abs = tol)
    assert r[1,2] == approx(0.0, abs = tol)
    assert r[2,0] == approx(0.0, abs = tol)
    assert r[2,1] == approx(0.0, abs = tol)
    assert r[2,2] == approx(1.0, abs = tol)

def test_relative_position_ecef_to_sez():
    tol = np.finfo(float).eps

    # 100m Overhead
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH + 100.0, 0.0, 0.0])

    r_sez = brahe.relative_position_ecef_to_sez(x_sta, r_ecef, "Geocentric")

    assert r_sez[0] == approx(0.0, abs = tol)
    assert r_sez[1] == approx(0.0, abs = tol)
    assert r_sez[2] == approx(100.0, abs = tol)

    # 100m North
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH, 0.0, 100.0])

    r_sez = brahe.relative_position_ecef_to_sez(x_sta, r_ecef, "Geocentric")

    assert r_sez[0] == approx(-100.0, abs = tol)
    assert r_sez[1] == approx(0.0, abs = tol)
    assert r_sez[2] == approx(0.0, abs = tol)

    # 100m East
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_ecef = np.array([brahe.R_EARTH, 100.0, 0.0])

    r_sez = brahe.relative_position_ecef_to_sez(x_sta, r_ecef, "Geocentric")

    assert r_sez[0] == approx(0.0, abs = tol)
    assert r_sez[1] == approx(100.0, abs = tol)
    assert r_sez[2] == approx(0.0, abs = tol)

    # Confirm higher latitude and longitude is (+E, +N, -Z)
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    x_geoc = np.array([0.5, 0.5, 0.0])
    r_ecef = brahe.position_geocentric_to_ecef(x_geoc, brahe.AngleFormat.DEGREES)

    r_sez_geoc = brahe.relative_position_ecef_to_sez(x_sta, r_ecef, "Geocentric")

    assert r_sez_geoc[0] < 0.0
    assert r_sez_geoc[1] > 0.0
    assert r_sez_geoc[2] < 0.0

    # Confirm difference in geocentric and geodetic conversions
    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    x_geod = np.array([0.5, 0.5, 0.0])
    r_ecef = brahe.position_geodetic_to_ecef(x_geod, brahe.AngleFormat.DEGREES)

    r_sez_geod = brahe.relative_position_ecef_to_sez(x_sta, r_ecef, "Geodetic")

    assert r_sez_geod[0] < 0.0
    assert r_sez_geod[1] > 0.0
    assert r_sez_geod[2] < 0.0

    for i in range(0, 3):
        assert r_sez_geoc[i] != r_sez_geod[i]

def test_relative_position_sez_to_ecef():
    tol = np.finfo(float).eps

    x_sta = np.array([brahe.R_EARTH, 0.0, 0.0])
    r_sez = np.array([0.0, 0.0, 100.0])

    r_ecef = brahe.relative_position_sez_to_ecef(x_sta, r_sez, "Geodetic")

    assert r_ecef[0] == approx(brahe.R_EARTH + 100.0, abs = tol)
    assert r_ecef[1] == approx(0.0, abs = tol)
    assert r_ecef[2] == approx(0.0, abs = tol)

def test_position_enz_to_azel():
    tol = np.finfo(float).eps

    # Directly above
    r_enz = np.array([0.0, 0.0, 100.0])
    x_azel = brahe.position_enz_to_azel(r_enz, AngleFormat.DEGREES)

    assert x_azel[0] == approx(0.0, abs = tol)
    assert x_azel[1] == approx(90.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # North
    r_enz = np.array([0.0, 100.0, 0.0])
    x_azel = brahe.position_enz_to_azel(r_enz, AngleFormat.DEGREES)

    assert x_azel[0] == approx(0.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # East
    r_enz = np.array([100.0, 0.0, 0.0])
    x_azel = brahe.position_enz_to_azel(r_enz, AngleFormat.DEGREES)

    assert x_azel[0] == approx(90.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # North-West
    r_enz = np.array([-100.0, 100.0, 0.0])
    x_azel = brahe.position_enz_to_azel(r_enz, AngleFormat.DEGREES)

    assert x_azel[0] == approx(315.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0 * math.sqrt(2.0), abs = tol)

def test_position_sez_to_azel():
    tol = np.finfo(float).eps

    # Directly above
    r_sez = np.array([0.0, 0.0, 100.0])
    x_azel = brahe.position_sez_to_azel(r_sez, AngleFormat.DEGREES)

    assert x_azel[0] == approx(0.0, abs = tol)
    assert x_azel[1] == approx(90.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # North
    r_sez = np.array([-100.0, 0.0, 0.0])
    x_azel = brahe.position_sez_to_azel(r_sez, AngleFormat.DEGREES)

    assert x_azel[0] == approx(0.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # East
    r_sez = np.array([0.0, 100.0, 0.0])
    x_azel = brahe.position_sez_to_azel(r_sez, AngleFormat.DEGREES)

    assert x_azel[0] == approx(90.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0, abs = tol)

    # North-West
    r_sez = np.array([-100.0, -100.0, 0.0])
    x_azel = brahe.position_sez_to_azel(r_sez, AngleFormat.DEGREES)

    assert x_azel[0] == approx(315.0, abs = tol)
    assert x_azel[1] == approx(0.0, abs = tol)
    assert x_azel[2] == approx(100.0 * math.sqrt(2.0), abs = tol)