# Imports
import pytest
import sys
import os
import math
import numpy as np

import brahe.constants as constants
import brahe.astrodynamics as astro
import brahe.coordinates as coords

# Add folder root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# State Test Fixtures

@pytest.fixture
def state_keplerian_deg():
    '''Intial Keplerian State. Near-Circular, Polar
    '''
    
    yield np.array([constants.R_EARTH + 500e3, 0.001, 90.0, 45.0, 30.0, 15.0])

@pytest.fixture
def state_keplerian_equatorial():
    '''Intial Keplerian State. Near-Circular, Equatorial
    '''
    yield np.array([constants.R_EARTH + 500e3, 0.001, 0.0, 45.0, 30.0, 15.0])

@pytest.fixture
def state_keplerian_circular():
    '''Intial Keplerian State. Circular, Polar
    '''
    
    yield np.array([constants.R_EARTH + 500e3, 0.0, 90.0, 45.0, 30.0, 15.0])

@pytest.fixture
def state_gcrf():
    '''Intial Equatorial ECEF State. At Equator, Near-Circular, Polar.
    '''
    
    # Get Semi-major Axis
    a = constants.R_EARTH + 500e3
    yield np.array([a, 0, 0, 0, 0, math.sqrt(constants.GM_EARTH/a)])

@pytest.fixture
def state_itrf():
    '''Intial Equatorial ECEF State. At Equator, Near-Circular, Polar.
    '''
    
    # Get Semi-major Axis
    a = constants.R_EARTH + 500e3
    yield np.array([0, -a, 0, 0, 0, math.sqrt(constants.GM_EARTH/a)])