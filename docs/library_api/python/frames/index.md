# Reference Frames Module

Reference frame transformations between inertial (ECI) and Earth-fixed (ECEF) coordinate systems.

## Transformation Categories

Brahe provides three categories of reference frame transformations:

### [ECI ↔ ECEF](eci_ecef.md)

Generic transformations using common "Earth-Centered Inertial" and "Earth-Centered Earth-Fixed" naming convention. Currently maps to GCRF ↔ ITRF transformations.

### [GCRF ↔ ITRF](gcrf_itrf.md)

Explicit transformations between Geocentric Celestial Reference Frame (inertial) and International Terrestrial Reference Frame (Earth-fixed). Uses IAU 2006/2000A CIO-based theory with classical angles.

### [EME2000 ↔ GCRF](eme2000_gcrf.md)

Constant frame bias transformations between Earth Mean Equator and Equinox of J2000.0 (classical inertial) and GCRF (modern inertial).
