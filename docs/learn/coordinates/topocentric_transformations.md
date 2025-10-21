# Topocentric Coordinate Transformations

Topocentric coordinates represent positions relative to a location on Earth's surface.

## Overview

Topocentric coordinates describe a position relative to an observer on Earth using local horizontal coordinates:

- **East (E)**: Eastward component in meters
- **North (N)**: Northward component in meters
- **Zenith (Z)**: Upward component in meters

Format: `[east, north, zenith]` (ENZ coordinates)

## Alternative: SEZ Coordinates

Some applications use SEZ (South-East-Zenith) coordinates instead:

- **South (S)**: Southward component
- **East (E)**: Eastward component
- **Zenith (Z)**: Upward component

## Azimuth-Elevation

Topocentric positions can also be expressed in spherical coordinates:

- **Azimuth (Az)**: Angle from north (clockwise) in radians
- **Elevation (El)**: Angle above horizon in radians
- **Range (R)**: Distance from observer in meters

## Conversions

### Relative Position to ENZ

```python
import brahe as bh

# Observer location (geodetic)
observer_lat = bh.DEG2RAD * 40.0
observer_lon = bh.DEG2RAD * -105.0
observer_alt = 1655.0  # meters

# Satellite position in ECEF
satellite_ecef = [...]

# Observer position in ECEF
observer_ecef = bh.position_geodetic_to_ecef([observer_lat, observer_lon, observer_alt])

# Relative position
relative_ecef = satellite_ecef - observer_ecef

# Convert to ENZ
enz = bh.relative_position_ecef_to_enz(observer_lat, observer_lon, relative_ecef)
```

### ENZ to Azimuth-Elevation-Range

```python
import brahe as bh

# ENZ position
enz = [1000.0, 2000.0, 500000.0]

# Convert to azimuth, elevation, range
azel = bh.position_enz_to_azel(enz)
azimuth, elevation, range = azel
```

## Use Cases

Topocentric coordinates are essential for:

- Ground station tracking
- Satellite visibility analysis
- Antenna pointing calculations
- Local horizon constraints

## See Also

- [Topocentric API Reference](../../library_api/coordinates/topocentric.md)
- [Access Computation](../access_computation/index.md)
