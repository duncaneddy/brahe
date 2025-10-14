import brahe
import math

def test_deg2rad():
    assert brahe.DEG2RAD == math.pi/180.0

def test_DEG2RAD():
    assert brahe.DEG2RAD == math.pi/180.0

def test_RAD2DEG():
    assert brahe.RAD2DEG == 180.0/math.pi

def test_AS2RAD():
    assert brahe.AS2RAD == brahe.DEG2RAD / 3600.0

def test_RAD2AS():
    assert brahe.RAD2AS == brahe.RAD2DEG * 3600.0

def test_MJD_ZERO():
    assert brahe.MJD_ZERO == 2400000.5

def test_MJD2000():
    assert brahe.MJD2000 == 51544.5

def test_GPS_TAI():
    assert brahe.GPS_TAI == -19.0

def test_TAI_GPS ():
    assert brahe.TAI_GPS  == -brahe.GPS_TAI

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
    assert brahe.WGS84_F == 1.0/298.257223563

def test_GM_EARTH():
    assert brahe.GM_EARTH == 3.986004415e14

def test_ECC_EARTH():
    assert brahe.ECC_EARTH == 8.1819190842622e-2

def test_J2_EARTH():
    assert brahe.J2_EARTH == 0.0010826358191967

def test_OMEGA_EARTH():
    assert brahe.OMEGA_EARTH == 7.292115146706979e-5

def test_GM_SUN():
    assert brahe.GM_SUN == 132712440041.939400*1e9

def test_R_SUN():
    assert brahe.R_SUN == 6.957*1e8

def test_P_SUN():
    assert brahe.P_SUN == 4.560E-6

def test_R_MOON():
    assert brahe.R_MOON == 1738*1e3

def test_GM_MOON():
    assert brahe.GM_MOON == 4902.800066*1e9

def test_GM_MERCURY():
    assert brahe.GM_MERCURY == 22031.780000*1e9

def test_GM_VENUS():
    assert brahe.GM_VENUS == 324858.592000*1e9

def test_GM_MARS():
    assert brahe.GM_MARS == 42828.37521*1e9

def test_GM_JUPITER():
    assert brahe.GM_JUPITER == 126712764.8*1e9

def test_GM_SATURN():
    assert brahe.GM_SATURN == 37940585.2*1e9

def test_GM_URANUS():
    assert brahe.GM_URANUS == 5794548.6*1e9

def test_GM_NEPTUNE():
    assert brahe.GM_NEPTUNE == 6836527.100580*1e9

def test_GM_PLUTO():
    assert brahe.GM_PLUTO == 977.000000*1e9
