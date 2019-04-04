# -*- coding: utf-8 -*-
"""
constants module provides common astrodynamics constants 
"""

# Module imports
from math import pi as _pi

# Mathematical Constants
AS2RAD = 2.0*_pi/360.0/3600.0
"""
Constant to convert arcseconds to radians. Equal to 2pi/(360*3600). [rad/as]
"""

RAD2AS = 360.0*3600.0/_pi/2.0
"""
Constant to convert radians to arcseconds. Equal to 2pi/(360*3600) [as/ras]
"""

# Physical Constants
C_LIGHT     = 299792458.0                 # [m/s]Exact definition Vallado
"""
Speed of light in vacuum. [m/s]

D. Vallado, Fundamentals of Astrodynamics and Applications (4th Ed.), 2010
"""

AU          = 1.49597870700e11            # [m] Astronomical Unit IAU 2010
"""
Astronomical Unit. Equal to the mean distance of the Earth from the sun.
TDB-compatible value. [m]

P. Gérard and B. Luzum, IERS Technical Note 36, 2010
"""

# Time Consants
MJD_ZERO = 2400000.5
MJD2000  = 51544.0
GPS_TAI  = -19.0
TAI_GPS  = -GPS_TAI
TT_TAI   = 32.184
TAI_TT   = -TT_TAI
GPS_TT   = GPS_TAI + TAI_TT
TT_GPS   = -GPS_TT
GPS_ZERO = 44244.0

# Earth Constants
R_EARTH     = 6.378136300e6               # [m] GGM05s Value
WGS84_a     = 6378137.0                   # WGS-84 semi-major axis
WGS84_f     = 1.0/298.257223563           # WGS-84 flattening
GM_EARTH    = 3.986004415e14              # [m^3/s^2] GGM05s Value
e_EARTH     = 8.1819190842622e-2          # [] First Eccentricity WGS84 Value
J2_EARTH    = 0.0010826358191967          # [] GGM05s value
OMEGA_EARTH = 7.292115146706979e-5        # [rad/s] Taken from Vallado 4th Ed page 222

# Sun Constants
GM_SUN      = 132712440041.939400*1e9     # Gravitational constant of the Sun
R_SUN       = 6.957*1e8                   # Nominal solar radius corresponding to photospheric radius
P_SUN       = 4.560E-6                    # [N/m^2] (~1367 W/m^2) Solar radiation pressure at 1 AU

# Celestial Constants - from JPL DE430 Ephemerides
GM_MOON     = 4902.800066*1e9
GM_MERCURY  = 22031.780000*1e9
GM_VENUS    = 324858.592000*1e9
GM_MARS     = 42828.37521*1e9
GM_JUPITER  = 126712764.8*1e9
GM_SATURN   = 37940585.2*1e9
GM_URANUS   = 5794548.6*1e9
GM_NEPTUNE  = 6836527.100580*1e9
GM_PLUTO    = 977.000000*1e9