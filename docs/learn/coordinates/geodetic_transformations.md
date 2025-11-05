# Geodetic Transformations

Geodetic longitude, latitude, altitude coordinates represent positions relative to the WGS84 ellipsoidal Earth model. These coordinates can be converted to and from Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates. This coordinate system is more accurate than the geocentric system for near-surface applications because it accounts for Earth's equatorial bulge.

For complete API details, see the [Geodetic Coordinates API Reference](../../library_api/coordinates/geodetic.md).

## Geodetic Coordinate System

Geodetic coordinates represent a position using:

- **Longitude** ($\lambda$): East-west angle from the prime meridian, in degrees [-180°, +180°] or radians $[-\pi, +\pi]$
- **Latitude** ($\varphi$): North-south angle from the equatorial plane, measured perpendicular to the ellipsoid surface, in degrees [-90°, +90°] or radians [$-\frac{\pi}{2}$, $+\frac{\pi}{2}$]
- **Altitude** ($h$): Height above the WGS84 ellipsoid surface, in meters

Combined as: `[longitude, latitude, altitude]`, often abbreviated as `[lon, lat, alt]`.

!!! info
    Geodetic latitude is measured perpendicular to the ellipsoid surface, not from Earth's center. This differs from geocentric latitude, which is measured from the center. For a point on the surface, these can differ by up to 11 arcminutes (about 0.2°).

### WGS84 Ellipsoid Model

The key difference between geodetic and geocentric coordinates is the Earth model:

- **Geodetic**: Earth is an ellipsoid (oblate spheroid) with parameters:
    - Semi-major axis: `WGS84_A = 6378137.0` meters (equatorial radius)
    - Flattening: `WGS84_F = 1/298.257223563`
- **Geocentric**: Earth is a perfect sphere of radius `WGS84_A`

The difference between equatorial and polar radii is approximately 21 km, which significantly affects position calculations near Earth's surface.

## Converting Geodetic to ECEF

Earth-Centered Earth-Fixed (ECEF) is a Cartesian coordinate system with:

- Origin at Earth's center of mass
- X-axis through the intersection of the prime meridian and equator
- Z-axis through the North Pole
- Y-axis completing a right-handed system

You can convert geodetic coordinates to ECEF Cartesian coordinates using the following:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/geodetic_to_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/geodetic_to_ecef.rs:4"
    ```

!!! info
    The conversion from geodetic to ECEF accounts for the ellipsoidal shape using the radius of curvature in the prime vertical and the first eccentricity of the ellipsoid.

## Converting ECEF to Geodetic

The reverse transformation converts Cartesian ECEF coordinates back to geodetic coordinates. This requires an iterative algorithm due to the ellipsoidal geometry:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/ecef_to_geodetic.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/ecef_to_geodetic.rs:4"
    ```

!!! info
    The ECEF to geodetic conversion uses an iterative algorithm that typically converges in 3-5 iterations to sub-millimeter precision.

## Geodetic vs Geocentric Accuracy

For the same longitude, latitude, and altitude values, geodetic and geocentric coordinates produce different ECEF positions. The difference is smallest near the equator and largest near the poles.

For most applications, it's best to use geodetic coordinates since any computational overhead is negligible compared to the improved accuracy near Earth's surface.

---

## See Also

- [Geodetic Coordinates API Reference](../../library_api/coordinates/geodetic.md) - Complete function documentation
- [Geocentric Transformations](geocentric_transformations.md) - Simpler spherical Earth model
- [Topocentric Transformations](topocentric_transformations.md) - Local horizon coordinate systems
