import pytest
from pytest import approx
import uuid
import numpy as np

from brahe.epoch import Epoch

import brahe

@pytest.fixture
def access_geometry_left():
    # Time - Set a vernal equinox  so OE line up as intented
    epc = Epoch(2020, 3, 21, 0, 0, 0, time_system='UTC')

    # Sat Location
    sat_oe = np.array([brahe.R_EARTH + 550e3, 0.0, 90, 0, 0, 0])
    sat_eci = brahe.sOSCtoCART(sat_oe, use_degrees=True)
    sat_ecef = brahe.sECItoECEF(epc, sat_eci)

    # Station Location
    loc_oe = np.array([brahe.R_EARTH + 550e3, 0.0, 90, 358, 0, 0])
    loc_eci = brahe.sOSCtoCART(loc_oe, use_degrees=True)
    loc_ecef = brahe.sECItoECEF(epc, loc_eci)
    loc_geod = brahe.sECEFtoGEOD(loc_ecef[0:3])
    sub_loc_geod = np.array([loc_geod[0], loc_geod[1], 0.0])
    loc_ecef = brahe.sGEODtoECEF(sub_loc_geod)

    yield epc, sat_ecef, loc_ecef

@pytest.fixture
def access_geometry_right():
    # Time - Set a vernal equinox  so OE line up as intented
    epc = Epoch(2020, 3, 21, 0, 0, 0, time_system='UTC')

    # Sat Location
    sat_oe = np.array([brahe.R_EARTH + 550e3, 0.0, 90, 180, 0, 180])
    sat_eci = brahe.sOSCtoCART(sat_oe, use_degrees=True)
    sat_ecef = brahe.sECItoECEF(epc, sat_eci)

    # Station Location
    loc_oe = np.array([brahe.R_EARTH + 550e3, 0.0, 90, 358, 0, 0])
    loc_eci = brahe.sOSCtoCART(loc_oe, use_degrees=True)
    loc_ecef = brahe.sECItoECEF(epc, loc_eci)
    loc_geod = brahe.sECEFtoGEOD(loc_ecef[0:3])
    sub_loc_geod = np.array([loc_geod[0], loc_geod[1], 0.0])
    loc_ecef = brahe.sGEODtoECEF(sub_loc_geod)

    yield epc, sat_ecef, loc_ecef