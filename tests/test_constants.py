import brahe
import math

import pytest


def test_deg2rad():
    assert brahe.DEG2RAD == math.pi / 180.0


def test_DEG2RAD():
    assert brahe.DEG2RAD == math.pi / 180.0


def test_RAD2DEG():
    assert brahe.RAD2DEG == 180.0 / math.pi


def test_AS2RAD():
    assert brahe.AS2RAD == brahe.DEG2RAD / 3600.0


def test_RAD2AS():
    assert brahe.RAD2AS == brahe.RAD2DEG * 3600.0


def test_MJD_ZERO():
    assert brahe.MJD_ZERO == 2400000.5


def test_MJD_J2000():
    assert brahe.MJD_J2000 == 51544.5


def test_JD_J2000():
    assert brahe.JD_J2000 == 2451545.0


def test_GPS_TAI():
    assert brahe.GPS_TAI == -19.0


def test_TAI_GPS():
    assert brahe.TAI_GPS == -brahe.GPS_TAI


def test_TT_TAI():
    assert brahe.TT_TAI == 32.184


def test_TAI_TT():
    assert brahe.TAI_TT == -brahe.TT_TAI


def test_GPS_TT():
    assert brahe.GPS_TT == brahe.GPS_TAI + brahe.TAI_TT


def test_TT_GPS():
    assert brahe.TT_GPS == -brahe.GPS_TT


def test_GPS_ZERO():
    assert brahe.GPS_ZERO == 44244.0


def test_C_LIGHT():
    assert brahe.C_LIGHT == 299792458.0


def test_AU():
    assert brahe.AU == 1.49597870700e11


def test_R_EARTH():
    assert brahe.R_EARTH == 6.378136300e6


def test_WGS84_A():
    assert brahe.WGS84_A == 6378137.0


def test_WGS84_F():
    assert brahe.WGS84_F == 1.0 / 298.257223563


def test_GM_EARTH():
    assert brahe.GM_EARTH == 3.986004415e14


def test_ECC_EARTH():
    assert brahe.ECC_EARTH == 8.1819190842622e-2


# EGM2008 fully-normalized Stokes coefficients C_n,0
# Source: data/gravity_models/EGM2008_360.gfc (degree n, order m=0 entries)
EGM2008_C_2_0 = -0.484165143790815e-03
EGM2008_C_3_0 = 0.957161207093473e-06
EGM2008_C_4_0 = 0.539965866638991e-06
EGM2008_C_5_0 = 0.686702913736681e-07
EGM2008_C_6_0 = -0.149953927978527e-06


def _unnormalize_zonal(c_n0: float, n: int) -> float:
    """Convert a fully-normalized zonal Stokes coefficient C_n,0 to the
    conventional unnormalized zonal J_n via J_n = -C_n,0 * sqrt(2n + 1)."""
    return -c_n0 * math.sqrt(2.0 * n + 1.0)


def test_J2_EARTH():
    assert brahe.J2_EARTH == pytest.approx(
        _unnormalize_zonal(EGM2008_C_2_0, 2), abs=1e-18
    )


def test_J3_EARTH():
    assert brahe.J3_EARTH == pytest.approx(
        _unnormalize_zonal(EGM2008_C_3_0, 3), abs=1e-21
    )


def test_J4_EARTH():
    assert brahe.J4_EARTH == pytest.approx(
        _unnormalize_zonal(EGM2008_C_4_0, 4), abs=1e-21
    )


def test_J5_EARTH():
    assert brahe.J5_EARTH == pytest.approx(
        _unnormalize_zonal(EGM2008_C_5_0, 5), abs=1e-22
    )


def test_J6_EARTH():
    assert brahe.J6_EARTH == pytest.approx(
        _unnormalize_zonal(EGM2008_C_6_0, 6), abs=1e-21
    )


def test_OMEGA_EARTH():
    assert brahe.OMEGA_EARTH == 7.292115146706979e-5


def test_GM_SUN():
    assert brahe.GM_SUN == 132712440041.939400 * 1e9


def test_R_SUN():
    assert brahe.R_SUN == 6.957 * 1e8


def test_P_SUN():
    assert brahe.P_SUN == 4.560e-6


def test_R_MOON():
    assert brahe.R_MOON == 1738 * 1e3


def test_GM_MOON():
    assert brahe.GM_MOON == 4902.800066 * 1e9


def test_GM_MERCURY():
    assert brahe.GM_MERCURY == 22031.780000 * 1e9


def test_GM_VENUS():
    assert brahe.GM_VENUS == 324858.592000 * 1e9


def test_GM_MARS():
    assert brahe.GM_MARS == 42828.37362069909 * 1e9


def test_GM_MARS_SYSTEM():
    assert brahe.GM_MARS_SYSTEM == 42828.375815756102 * 1e9


def test_GM_JUPITER():
    assert brahe.GM_JUPITER == 126686531.9003704 * 1e9


def test_GM_JUPITER_SYSTEM():
    assert brahe.GM_JUPITER_SYSTEM == 126712764.09999998 * 1e9


def test_GM_SATURN():
    assert brahe.GM_SATURN == 37931206.23436167 * 1e9


def test_GM_SATURN_SYSTEM():
    assert brahe.GM_SATURN_SYSTEM == 37940584.841799997 * 1e9


def test_GM_URANUS():
    assert brahe.GM_URANUS == 5793951.256527211 * 1e9


def test_GM_URANUS_SYSTEM():
    assert brahe.GM_URANUS_SYSTEM == 5794556.3999999985 * 1e9


def test_GM_NEPTUNE():
    assert brahe.GM_NEPTUNE == 6835103.145462294 * 1e9


def test_GM_NEPTUNE_SYSTEM():
    assert brahe.GM_NEPTUNE_SYSTEM == 6836527.1005803989 * 1e9


def test_GM_PLUTO():
    assert brahe.GM_PLUTO == 869.6138177608748 * 1e9


def test_GM_PLUTO_SYSTEM():
    assert brahe.GM_PLUTO_SYSTEM == 975.5 * 1e9


def test_R_MARS():
    assert brahe.R_MARS == pytest.approx(3.39619e6, abs=1.0)


def test_OMEGA_MARS():
    # OMEGA_MARS derives from the WGCCRE 2015 prime-meridian rate (350.891982443297 deg/day)
    assert brahe.OMEGA_MARS == pytest.approx(
        math.radians(350.891982443297) / 86400.0, abs=1e-15
    )


def test_OMEGA_MOON():
    # OMEGA_MOON derives from the IAU W1 rate for the Moon (13.17635815 deg/day)
    assert brahe.OMEGA_MOON == pytest.approx(
        math.radians(13.17635815) / 86400.0, abs=1e-15
    )


def test_GM_PHOBOS():
    assert brahe.GM_PHOBOS == pytest.approx(7.087546066894452e5, abs=1e-3)


def test_GM_DEIMOS():
    assert brahe.GM_DEIMOS == pytest.approx(9.615569648120313e4, abs=1e-3)


def test_mars_system_gm_is_not_the_component_sum():
    """gm_de440.tpc combines the DE440 planetary-solution barycenter GM with
    the older Horizons satellite-solution body GMs; their sum deliberately
    does not reproduce the system value (mirrors the Rust guard test)."""
    component_sum = brahe.GM_MARS + brahe.GM_PHOBOS + brahe.GM_DEIMOS
    diff = brahe.GM_MARS_SYSTEM - component_sum
    assert 1.0e6 < abs(diff) < 2.0e6
