# Geocentric Transformations

Geocentric longitude, latitude, altitude coordinates represent positions relative to a spherical Earth's surface. These coordinates can be converted to and from Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates. This coordinate system is simpler and computationally faster than the geodetic system, but less accurate for near-surface applications because it assumes Earth is a perfect sphere.

For complete API details, see the [Geocentric Coordinates API Reference](../../library_api/coordinates/geodetic.md#geocentric-conversions).

## Geocentric Coordinate System

Geocentric coordinates represent a position using:

- **Longitude** ($\lambda$): East-west angle from the prime meridian, in degrees [-180°, +180°] or radians $[-\pi, +\pi]$
- **Latitude** ($\varphi$): North-south angle from the equatorial plane, in degrees [-90°, +90°] or radians $[-\frac{\pi}{2}, +\frac{\pi}{2}]$
- **Altitude** ($h$): Height above the spherical Earth surface, in meters

Combined as: `[longitude, latitude, altitude]`, often abbreviated as `[lon, lat, alt]`.

!!! info
    The spherical Earth model uses an Earth radius of `6378137.0` meters, which is the WGS84 semi-major axis. This means the geocentric "surface" is a sphere with Earth's equatorial radius.

### Spherical vs Ellipsoidal Earth

The key difference between geocentric and geodetic coordinates is the Earth model:

- **Geocentric**: Earth is a perfect sphere of radius `WGS84_A`
- **Geodetic**: Earth is an ellipsoid (oblate spheroid) with equatorial bulge

## Converting Geocentric to ECEF

Earth-Centered Earth-Fixed (ECEF) is a Cartesian coordinate system with:

- Origin at Earth's center of mass
- X-axis through the intersection of the prime meridian and equator
- Z-axis through the North Pole
- Y-axis completing a right-handed system

You can convert geocentric spherical coordinates to ECEF Cartesian coordinates using following:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/geocentric_to_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/geocentric_to_ecef.rs:4"
    ```

## Converting ECEF to Geocentric

The reverse transformation converts Cartesian ECEF coordinates back to geocentric spherical coordinates:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/ecef_to_geocentric.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/ecef_to_geocentric.rs:4"
    ```

!!! info
    Latitude values are automatically constrained to the valid range [-90°, +90°] or [$-\frac{\pi}{2}$, $+\frac{\pi}{2}$] during conversion.

## See Also

- [Geocentric Coordinates API Reference](../../library_api/coordinates/geodetic.md#geocentric-conversions) - Complete function documentation
- [Geodetic Transformations](geodetic_transformations.md) - More accurate WGS84 ellipsoid model
- [Topocentric Transformations](topocentric_transformations.md) - Local horizon coordinate systems
