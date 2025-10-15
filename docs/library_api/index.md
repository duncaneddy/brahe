# Python API Reference

This section provides comprehensive documentation for the Brahe Python API. All classes, functions, and modules are documented with detailed descriptions, parameters, return values, and usage examples.

## Core Modules

### Time
The time module provides precise time representation and conversion between different time systems (UTC, TAI, GPS, TT, UT1). The `Epoch` class is the foundational time representation used throughout Brahe.

**Key Components:**
- `Epoch`: High-precision time representation with nanosecond accuracy
- Time system conversions and utilities

### Coordinates
Coordinate transformation functions for converting between various coordinate systems used in astrodynamics.

**Key Components:**
- Cartesian coordinates (position and velocity vectors)
- Geodetic coordinates (latitude, longitude, altitude)
- Geocentric coordinates
- Topocentric coordinates (East-North-Up, South-East-Zenith)

### Frames
Reference frame transformations between Earth-Centered Inertial (ECI) and Earth-Centered Earth-Fixed (ECEF) coordinate frames using IAU 2006/2000A models.

### Orbits
Orbital mechanics representations and propagators for satellite trajectory computation.

**Key Components:**
- Keplerian orbital elements
- Two-Line Element (TLE) format parsing
- Analytical propagators (Keplerian, SGP4/SDP4)

### Attitude
Attitude representation and conversion between different rotation parameterizations.

**Key Components:**
- Quaternions
- Rotation matrices (Direction Cosine Matrices)
- Euler angles (various sequences)
- Euler axis-angle representation

### Trajectories
High-level trajectory containers with interpolation support for storing and querying orbital states over time.

### Earth Orientation Parameters (EOP)
Earth orientation parameter data management for high-precision coordinate frame transformations.

### Constants
Mathematical, physical, and time-related constants used throughout the library.

---

## Navigation

Use the sidebar to navigate through the detailed API documentation for each module. Each page includes:

- Complete class and function signatures
- Detailed parameter descriptions
- Return value specifications
- Usage examples
- References to related functionality

All documentation is automatically generated from the source code docstrings to ensure accuracy and consistency.
