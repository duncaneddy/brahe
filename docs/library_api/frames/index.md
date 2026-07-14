# Reference Frames Module

Reference frame transformations between inertial and body-fixed coordinate systems for Earth, the Moon, and Mars, synodic (two-body rotating) frames, plus a generic router for any two supported frames.

## Transformation Categories

### [ECI ↔ ECEF](eci_ecef.md)

Generic transformations using common "Earth-Centered Inertial" and "Earth-Centered Earth-Fixed" naming convention. Currently maps to GCRF ↔ ITRF transformations.

### [GCRF ↔ ITRF](gcrf_itrf.md)

Explicit transformations between Geocentric Celestial Reference Frame (inertial) and International Terrestrial Reference Frame (Earth-fixed). Uses IAU 2006/2000A CIO-based theory with classical angles.

### [EME2000 ↔ GCRF](eme2000_gcrf.md)

Constant frame bias transformations between Earth Mean Equator and Equinox of J2000.0 (classical inertial) and GCRF (modern inertial).

### [Lunar Frames](lunar.md)

Transformations between Lunar-Centered Inertial (LCI) and the Moon-fixed LFPA/LFME frames.

### [Mars Frames](mars.md)

Transformations between Mars-Centered Inertial (MCI) and the Mars-fixed MCMF frame.

### [Synodic Frames](synodic.md)

Transformations between GCRF and the synodic EMR, SER, and GSE frames.

### [Reference Frame Router](router.md)

`ReferenceFrame` and the generic `rotation_frame_to_frame`/`position_frame_to_frame`/`state_frame_to_frame` functions, which convert between any two supported frames, including generic NAIF-ID variants for bodies without a dedicated named frame.
