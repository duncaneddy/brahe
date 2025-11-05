# Orbital Elements

In orbital mechanics, describing the motion of a satellite requires representing its position and velocity at any given time. There are two primary ways to represent an orbit:

1. **Cartesian state vectors** - Position and velocity in three-dimensional space: `[x, y, z, vx, vy, vz]`
2. **Keplerian orbital elements** - Six parameters that describe the orbit's shape, orientation, and the satellite's position within it: `[a, e, i, Ω, ω, ν]`

Both representations contain the same information, but each has advantages for different applications. Cartesian states are ideal for numerical propagation and reference frame transformations, while Keplerian elements provide intuitive understanding of orbital characteristics like size, shape, and orientation.

The `brahe.orbits` module provides tools for working with Keplerian orbital elements, computing orbital properties, and handling the Two-Line Element (TLE) format used for distributing satellite orbit information.

## Orbital Representations

### Keplerian Elements

Keplerian orbital elements describe an orbit using six parameters:

- **Semi-major axis (a)** - Defines the size of the orbit [meters]
- **Eccentricity (e)** - Defines the shape (0 = circular, 0 < e < 1 = elliptical) [dimensionless]
- **Inclination (i)** - Angle between orbital plane and equator [radians or degrees]
- **Right Ascension of Ascending Node (Ω)** - Orientation of the orbital plane [radians or degrees]
- **Argument of Periapsis (ω)** - Orientation of the orbit within its plane [radians or degrees]
- **Anomaly** - Satellite's position along the orbit [radians or degrees]

In Brahe, Keplerian elements are represented as arrays: `[a, e, i, Ω, ω, anomaly]` where the semi-major axis is in meters and angles are in radians (unless using the `AngleFormat` enum to specify degrees).

The anomaly can be expressed in three forms - true, eccentric, or mean anomaly - each useful for different calculations. See [Anomaly Conversions](anomalies.md) for details.

!!! tip
    For all functions in the `brahe.orbits` module, the anomaly is assumed to be the **mean anomaly** unless otherwise specified.

### Cartesian States

Cartesian state vectors represent position and velocity in three-dimensional space: `[x, y, z, vx, vy, vz]`. In Brahe, position components are in meters and velocity components are in meters per second.

Brahe provides functions to convert between Keplerian elements and Cartesian states:

- `state_osculating_to_cartesian()` - Convert orbital elements to Cartesian state
- `state_cartesian_to_osculating()` - Convert Cartesian state to orbital elements

These functions are found in the [coordinates module](../coordinates/cartesian_transformations.md) but are essential for working with orbits.

## Topics in This Section

### [Orbital Properties](properties.md)

Learn about computing fundamental orbital properties including:

- Orbital period and mean motion
- Semi-major axis from period or mean motion
- Periapsis and apoapsis distances, altitudes, and velocities
- Sun-synchronous orbit inclination

### [Anomaly Conversions](anomalies.md)

Understand the three types of orbital anomaly and how to convert between them:

- **True anomaly** - Actual angular position from periapsis
- **Eccentric anomaly** - Auxiliary angle used in elliptical orbit calculations
- **Mean anomaly** - Linearly increasing angle representing average motion

### [Two-Line Elements](two_line_elements.md)

Work with the TLE format for distributing satellite orbital information:

- TLE structure and parsing
- TLE validation and checksums
- Extracting orbital elements from TLEs
- Creating TLEs from Keplerian elements
- NORAD satellite catalog number conversions

---

## See Also

- [Orbits API Reference](../../library_api/orbits/index.md) - Complete orbital functions documentation
- [Coordinates](../coordinates/index.md) - Cartesian and Keplerian conversions
- [Orbit Propagation](../orbit_propagation/index.md) - Propagating orbits over time
- [Physical Constants](../../library_api/constants/physical.md) - Gravitational parameters and radii
