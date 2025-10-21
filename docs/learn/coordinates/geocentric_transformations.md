# Geocentric Coordinate Transformations

Geocentric coordinates represent positions using spherical coordinates centered at Earth's center.

## Overview

Geocentric coordinates describe a position using:

- **Radius** (r): Distance from Earth's center in meters
- **Latitude** (φ): Angle north/south of equator in radians
- **Longitude** (λ): Angle east of prime meridian in radians

Format: `[radius, latitude, longitude]`

## Geocentric vs Geodetic

!!! warning "Important Distinction"
    Geocentric coordinates use a spherical Earth model, while geodetic coordinates use the WGS84 ellipsoid. The latitude values differ between the two systems.

**Geocentric latitude**: Angle from equatorial plane to point
**Geodetic latitude**: Angle from equatorial plane to surface normal

## Conversions

### Geocentric to ECEF (Cartesian)

```python
import brahe as bh

# Geocentric coordinates [radius, lat, lon] in meters and radians
geocentric = [bh.R_EARTH + 500e3, bh.DEG2RAD * 45.0, bh.DEG2RAD * -122.0]

# Convert to ECEF Cartesian
ecef = bh.position_geocentric_to_ecef(geocentric)
```

### ECEF to Geocentric

```python
import brahe as bh

# ECEF Cartesian position
ecef = [3194.469e3, -3194.469e3, 4487.348e3]

# Convert to geocentric
geocentric = bh.position_ecef_to_geocentric(ecef)
radius, lat, lon = geocentric
```

## Use Cases

Geocentric coordinates are useful for:

- Quick distance calculations from Earth's center
- Simplified orbital computations
- When Earth oblateness can be ignored

## See Also

- [Geodetic Coordinates](../../library_api/coordinates/geodetic.md)
- [Geocentric API Reference](../../library_api/coordinates/geodetic.md)
