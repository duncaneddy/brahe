# Coordinate Transformations

Coordinate systems and transformations are fundamental to astrodynamics. Different coordinate systems are used depending on the application - some are better for orbital mechanics, others for ground station tracking, and still others for describing positions on Earth's surface.

Brahe provides coordinate transformation functions for converting between different coordinate representations and reference frames.

## Coordinate Systems in Brahe

### Cartesian State Vectors

Cartesian coordinates represent positions and velocities as vectors in three-dimensional space: `[x, y, z, vx, vy, vz]`. In astrodynamics, these are typically used to represent the state in Earth-Centered Inertial (ECI) frames or Earth-Centered Earth-Fixed (ECEF) frames. 

While Cartesian coordinates are mathematically convenient for numerical integration and propagation, orbital elements are often more intuitive for understanding orbits. Therefore, Brahe provides functions to convert between ECI Cartesian states and Keplerian orbital elements.

Learn more: [Cartesian Transformations](cartesian_transformations.md)

### Geocentric Coordinates

Geocentric coordinates use a spherical Earth model to represent positions using longitude, latitude, and altitude from Earth's center: `[lon, lat, alt]`. Applications generally convert between ECEF Cartesian coordinates and geocentric spherical coordinates.

Learn more: [Geocentric Transformations](geocentric_transformations.md)

### Geodetic Coordinates

Geodetic coordinates use the WGS84 ellipsoid model to represent positions: `[lon, lat, alt]`. Unlike geocentric coordinates, geodetic coordinates account for Earth's equatorial bulge (flattening), providing much more accurate resulting ECEF Cartesian positions for points on or near Earth's surface. Similar to geocentric coordinates, applications typically convert between ECEF Cartesian coordinates and geodetic coordinates.

Learn more: [Geodetic Transformations](geodetic_transformations.md)

### Topocentric Coordinates

Topocentric coordinate systems are local horizon-based systems centered on an observer (like a ground station on the Earth). They represent the position of objects relative to the local horizon, tagent to the body's surface.

Two of the most common topocentric coodinate systems are the East-North-Zenith (ENZ) or South-East-Zenith (SEZ) systems. These systems are essential for computing satellite visibility, tracking angles (azimuth and elevation), and determining when satellites are observable from a specific location.

Learn more: [Topocentric Transformations](topocentric_transformations.md)

## Common Transformation Patterns

### Orbital Mechanics

To work with orbital elements:

1. Define or receive Keplerian elements `[a, e, i, Ω, ω, M]`
1. Convert to Cartesian state `[x, y, z, vx, vy, vz]` in ECI frame


### Ground Station Observations

To compute satellite location from a ground station:

1. Start with station location in geodetic coordinates `[lon, lat, alt]`
1. Convert the Satellite ECI positions from to the ECEF frame at the observation time
1. Convert the station location to ECEF coordinates
1. Transform the satellite and location ECEF coordinates to the local ENZ
1. Convert ENZ to azimuth-elevation-range

---

## See Also

- [Cartesian Transformations](cartesian_transformations.md) - Orbital elements and Cartesian states
- [Geocentric Transformations](geocentric_transformations.md) - Spherical Earth coordinates
- [Geodetic Transformations](geodetic_transformations.md) - WGS84 ellipsoid coordinates
- [Topocentric Transformations](topocentric_transformations.md) - Local horizon systems
- [Coordinates API Reference](../../library_api/coordinates/index.md) - Complete API documentation
