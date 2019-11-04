# Test Imports
from pytest import approx

# Modules Under Test
from brahe.eop import *

# Other imports
import math

def test_load():
    EOP.load(IERS_AB_EOP_DATA)
    EOP.load(IERS_C04_DATA)
    EOP.load(DEFAULT_EOP_DATA)

# Test Data valutes
def test_data():
    # EOP
    ut1_utc, xp, yp =  EOP.eop(58483)
    assert approx(ut1_utc, -0.0351914, abs=1e-6)
    assert approx(xp, 0.088501*AS2RAD, abs=1e-9)
    assert approx(yp, 0.270752*AS2RAD, abs=1e-9)

    # UTC - UT1 Offset
    assert approx(EOP.ut1_utc(58483), -0.0351914, abs=1e-6)
    assert approx(EOP.utc_ut1(58483),  0.0351914, abs=1e-6)
    
    # Pole Locator
    xp, yp =  EOP.pole_locator(58483)
    assert approx(xp, 0.088501*AS2RAD, abs=1e-9)
    assert approx(yp, 0.270752*AS2RAD, abs=1e-9)

    # Pole Components
    assert approx(EOP.xp(58483), 0.088501*AS2RAD, abs=1e-9)
    assert approx(EOP.yp(58483), 0.270752*AS2RAD, abs=1e-9)

# Test Set values
def test_set():
    EOP.set(58747, -0.2, 0.225, 0.3)

    assert EOP.ut1_utc(58747) == -0.2
    xp, yp =  EOP.pole_locator(58747)
    assert xp == 0.225*AS2RAD
    assert yp == 0.3*AS2RAD
    assert EOP.xp(58747) == 0.225*AS2RAD
    assert EOP.yp(58747) == 0.3*AS2RAD

    EOP.set(58747,-0.1897929, 0.230292, 0.332704)

    assert EOP.ut1_utc(58747) == -0.1897929
    xp, yp =  EOP.pole_locator(58747)
    assert xp == 0.230292*AS2RAD
    assert yp == 0.332704*AS2RAD
    assert EOP.xp(58747) == 0.230292*AS2RAD
    assert EOP.yp(58747) == 0.332704*AS2RAD

# Test interpolation values
def test_interp():
    ut1_utc = (EOP.ut1_utc(58748) + EOP.ut1_utc(58747))/2.0
    assert EOP.ut1_utc(58747.5, interp=True) == ut1_utc

    x1, y1 = EOP.pole_locator(58747)
    x2, y2 = EOP.pole_locator(58748)
    pole_locator = ((x2+x1)/2.0, (y2+y1)/2.0)
    assert EOP.pole_locator(58747.5, interp=True)[0] == pole_locator[0]
    assert EOP.pole_locator(58747.5, interp=True)[1] == pole_locator[1]

    xp = (EOP.xp(58748) + EOP.xp(58747))/2.0
    assert EOP.xp(58747.5, interp=True) == xp

    yp = (EOP.yp(58748) + EOP.yp(58747))/2.0
    assert EOP.yp(58747.5, interp=True) == yp

if __name__ == '__main__':
    test_load()
    test_data()
    test_set()
    test_interp()