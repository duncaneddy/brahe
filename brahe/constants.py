# -*- coding: utf-8 -*-
"""
constants module provides common astrodynamics constants 
"""

# Module imports
from math import pi as _pi
import pathlib as pathlib

# Module Constatns
DATA_PATH = pathlib.Path(__file__).parent / 'data'
"""
Path to internal module data. Path is defined relative to the package installation
directory.
"""

# Mathematical Constants
AS2RAD = 2.0*_pi/360.0/3600.0
"""
Constant to convert arcseconds to radians. Equal to 2pi/(360*3600). Units: *rad/as*
"""

RAD2AS = 360.0*3600.0/_pi/2.0
"""
Constant to convert radians to arcseconds. Equal to 2pi/(360*3600). Units: *as/ras*
"""

# Physical Constants
C_LIGHT     = 299792458.0                 # [m/s]Exact definition Vallado
"""
Speed of light in vacuum. Units: *m/s*

References:

1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010
"""

AU          = 1.49597870700e11            # [m] Astronomical Unit IAU 2010
"""
Astronomical Unit. Equal to the mean distance of the Earth from the sun.
TDB-compatible value. Units: *m*

References:
    
1. P. Gérard and B. Luzum, *IERS Technical Note 36*, 2010
"""

# Time Consants
MJD_ZERO = 2400000.5
"""
Offset of Modified Julian Days representation with respect to Julian Days. For 
a time, t, MJD_ZERO is equal to:

``MJD_ZERO = t_jd - t_mjd``

Where t_jd is the epoch represented in Julian Days, and t_mjd is the epoch in
Modified Julian Days.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012.
"""

MJD2000  = 51544.5
"""
Modified Julian Date of January 1, 2000 12:00:00. Value is independent of time
scale.

References:
TODO: Fix Reference
1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012. 
"""

GPS_TAI  = -19.0
"""
Offset of GPS time system with respect to TAI time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

TAI_GPS  = -GPS_TAI
"""
Offset of TAI time system with respect to GPS time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

TT_TAI   = 32.184
"""
Offset of TT time system with respect to TAI time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

TAI_TT   = -TT_TAI
"""
Offset of TAI time system with respect to TT time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GPS_TT   = GPS_TAI + TAI_TT
"""
Offset of GPS time system with respect to TT time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

TT_GPS   = -GPS_TT
"""
Offset of TT time system with respect to GPS time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GPS_ZERO = 44244.0
"""
Modified Julian Date of the start of the GPS time system in the GPS time system.
This date was January 6, 1980 0H as reckond in the UTC time system.

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

# Earth Constants
R_EARTH     = 6.378136300e6               # [m] GGM05s Value
"""
Earth's equatorial radius. [m]

References:

1. GGM05s Gravity Model
"""

WGS84_a     = 6378137.0                   # WGS-84 semi-major axis
"""
Earth's semi-major axis as defined by the WGS84 geodetic system. [m]

References:

1. NIMA Technical Report TR8350.2
"""

WGS84_f     = 1.0/298.257223563           # WGS-84 flattening
"""
Earth's ellipsoidal flattening.  WGS84 Value.

References:

1. NIMA Technical Report TR8350.2
"""

GM_EARTH    = 3.986004415e14              # [m^3/s^2] GGM05s Value
"""
Earth's Gravitational constant [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

ECC_EARTH   = 8.1819190842622e-2          # [] First Eccentricity WGS84 Value
"""
Earth's first eccentricity. WGS84 Value. [dimensionless]

References:

1. NIMA Technical Report TR8350.2
"""

J2_EARTH    = 0.0010826358191967          # [] GGM05s value
"""
Earth's first zonal harmonic. [dimensionless]

References:

1. GGM05s Gravity Model.
"""

OMEGA_EARTH = 7.292115146706979e-5        # [rad/s] Taken from Vallado 4th Ed page 222
"""
Earth axial rotation rate. [rad/s]

References:

1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, p. 222, 2010
"""

# Sun Constants
GM_SUN      = 132712440041.939400*1e9     # Gravitational constant of the Sun
"""
Gravitational constant of the Sun. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

R_SUN       = 6.957*1e8                   # Nominal solar radius corresponding to photospheric radius
"""
Nominal solar photospheric radius. [m]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

P_SUN       = 4.560E-6                    # [N/m^2] (~1367 W/m^2) Solar radiation pressure at 1 AU
"""
Nominal solar radiation pressure at 1 AU. [N/m^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

# Celestial Constants - from JPL DE430 Ephemerides
GM_MOON     = 4902.800066*1e9
"""
Gravitational constant of the Moon. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_MERCURY  = 22031.780000*1e9
"""
Gravitational constant of the Mercury. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_VENUS    = 324858.592000*1e9
"""
Gravitational constant of the Venus. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_MARS     = 42828.37521*1e9
"""
Gravitational constant of the Mars. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_JUPITER  = 126712764.8*1e9
"""
Gravitational constant of the Jupiter. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_SATURN   = 37940585.2*1e9
"""
Gravitational constant of the Saturn. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_URANUS   = 5794548.6*1e9
"""
Gravitational constant of the Uranus. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_NEPTUNE  = 6836527.100580*1e9
"""
Gravitational constant of the Neptune. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""

GM_PLUTO    = 977.000000*1e9
"""
Gravitational constant of the Pluto. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and 
Applications*, 2012.
"""