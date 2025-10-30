# Topocentric Coordinate Transformations

Topocentric coordinate systems are local horizon-based reference frames centered on an observer, such as a ground station or radar site. These coordinate systems are essential for satellite tracking, visibility analysis, and determining where to point antennas or telescopes.

Unlike global coordinate systems (ECEF, ECI), topocentric systems define positions relative to a specific location on Earth, making it easy to determine whether a satellite is visible and where to look in the sky.

For complete API details, see the [Topocentric Coordinates API Reference](../../library_api/coordinates/topocentric.md).

## Topocentric Coordinate Systems

Brahe supports two local horizon coordinate systems:

### ENZ (East-North-Zenith)

- **East** (E): Positive toward geographic east
- **North** (N): Positive toward geographic north
- **Zenith** (Z): Positive upward (toward the sky)

This is the most common topocentric system for satellite tracking and is aligned with geographic directions.

### SEZ (South-East-Zenith)

- **South** (S): Positive toward geographic south
- **East** (E): Positive toward geographic east
- **Zenith** (Z): Positive upward (toward the sky)

The SEZ system is sometimes used in radar and missile tracking applications. The main difference from ENZ is that the first two axes are rotated 180° around the zenith axis.

!!! info
    Both ENZ and SEZ use a right-handed coordinate system with the zenith axis pointing up. The choice between them is typically driven by convention in your specific field or application.

## Station Location Interpretation

When specifying the observer (ground station) location, you must choose whether the coordinates represent:

- **Geodetic** (`EllipsoidalConversionType.GEODETIC`): Station coordinates use WGS84 ellipsoid (recommended for accuracy)
- **Geocentric** (`EllipsoidalConversionType.GEOCENTRIC`): Station coordinates use spherical Earth model

For ground stations, geodetic interpretation is almost always preferred for accuracy.

## ENZ Transformations

### Converting ECEF to ENZ

To get the position of an object relative to a location, you need to convert the object's ECEF position to the local ENZ frame centered on the location:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/ecef_to_enz.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/ecef_to_enz.rs:4"
    ```

### Converting ENZ to ECEF

The reverse transformation converts a relative ENZ position back to an absolute ECEF position:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/enz_to_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/enz_to_ecef.rs:4"
    ```

## SEZ Transformations

### Converting ECEF to SEZ

Similar to ENZ, you can convert ECEF positions to the SEZ frame:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/ecef_to_sez.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/ecef_to_sez.rs:4"
    ```

### Converting SEZ to ECEF

The reverse transformation converts a relative SEZ position back to an absolute ECEF position:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/sez_to_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/sez_to_ecef.rs:4"
    ```

## Azimuth and Elevation from Topocentric Coordinates

For object tracking, it's often more intuitive to work with azimuth (compass direction) and elevation (angle above the horizon) rather than Cartesian ENZ or SEZ coordinates. Both ENZ and SEZ topocentric systems can be converted to azimuth-elevation-range format.

### From ENZ Coordinates

Convert ENZ positions to azimuth (measured clockwise from North), elevation (angle above horizon), and range:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/enz_to_azel.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/enz_to_azel.rs:4"
    ```

!!! info
    Azimuth is measured clockwise from North (0° = North, 90° = East, 180° = South, 270° = West). Elevation is the angle above the horizon (0° = horizon, 90° = directly overhead).

### From SEZ Coordinates

The same conversion is available from SEZ coordinates:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/sez_to_azel.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/sez_to_azel.rs:4"
    ```

!!! info
    Both ENZ and SEZ produce identical azimuth-elevation-range results for the same physical position. The choice between them is purely a matter of intermediate representation.

## See Also

- [Topocentric Coordinates API Reference](../../library_api/coordinates/topocentric.md) - Complete function documentation
- [Geodetic Transformations](geodetic_transformations.md) - Converting station locations to ECEF
- [Frame Transformations](../../library_api/frames.md) - Converting satellite positions from ECI to ECEF
- Access Analysis - Higher-level tools for computing satellite visibility windows
