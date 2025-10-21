# Coordinate Transformations

Coordinate systems are fundamental to astrodynamics, allowing us to represent positions and velocities in different reference frames optimized for specific calculations. Brahe provides comprehensive coordinate transformation capabilities between multiple systems.

## Overview

Brahe supports three primary coordinate systems for representing positions on or near Earth:

1. **Cartesian Coordinates** - 3D rectangular coordinates `[x, y, z]`
2. **Geocentric Coordinates** - Spherical coordinates `[radius, latitude, longitude]`
3. **Geodetic Coordinates** - Ellipsoidal coordinates accounting for Earth's shape `[latitude, longitude, altitude]`
4. **Topocentric Coordinates** - Local horizontal coordinates relative to an observer `[East, North, Zenith]` or `[Azimuth, Elevation, Range]`

Each system has specific use cases and advantages depending on the problem being solved.

## Coordinate Systems

### Cartesian Coordinates

Cartesian coordinates represent positions as 3D vectors with orthogonal axes:

- **ECEF (Earth-Centered Earth-Fixed)**: Rotates with Earth, Z-axis through North Pole
- **ECI (Earth-Centered Inertial)**: Inertial frame fixed to celestial sphere

**Best for**:
- Orbital mechanics (Newton's laws in inertial frames)
- Vector operations (dot products, cross products)
- Frame transformations (rotation matrices)

See: [Cartesian Transformations](cartesean_transformations.md)

### Geocentric Coordinates

Geocentric coordinates use spherical representation `[r, φ, λ]`:

- **r**: Distance from Earth's center (meters)
- **φ**: Geocentric latitude (radians)
- **λ**: Longitude (radians)

**Best for**:
- Spherical Earth approximations
- Quick distance calculations
- When Earth's oblateness can be ignored

See: [Geocentric Transformations](geocentric_transformations.md)

### Geodetic Coordinates

Geodetic coordinates account for Earth's ellipsoidal shape using WGS84:

- **φ**: Geodetic latitude - angle from equatorial plane to surface normal (radians)
- **λ**: Longitude (radians)
- **h**: Height above WGS84 ellipsoid (meters)

**Best for**:
- Ground station locations
- GPS coordinates
- Precise Earth surface positioning
- Terrain-relative calculations

See: [Geodetic & Geocentric Transformations](geocentric_transformations.md)

### Topocentric Coordinates

Topocentric coordinates are local horizontal systems centered at an observer:

**ENZ (East-North-Zenith)**:
- Local Cartesian with origin at observer
- E: East direction, N: North direction, Z: Zenith (up)

**SEZ (South-East-Zenith)**:
- Alternative convention with S: South, E: East, Z: Zenith

**Azimuth-Elevation-Range**:
- Spherical representation of observer-relative position
- Az: Compass bearing (0° = North, 90° = East)
- El: Elevation angle above horizon (degrees)
- Range: Distance to object (meters)

**Best for**:
- Ground station tracking
- Antenna pointing
- Visibility calculations
- Access computation

See: [Topocentric Transformations](topocentric_transformations.md)

## Common Transformation Chains

### Satellite State to Ground Observer

```
ECI State → ECEF State → Topocentric Position → Az/El/Range
```

Used for ground station tracking and visibility.

### Ground Location to Orbit Frame

```
Geodetic (lat/lon/alt) → ECEF Position → ECI Position
```

Used for determining when satellites can view ground locations.

### Orbital Elements to Cartesian

```
Keplerian Elements → ECI Cartesian State
```

Used for orbit propagation and trajectory analysis.

## Key Concepts

### Reference Frames

- **Inertial Frames (ECI)**: Non-rotating, fixed to celestial sphere. Required for applying Newton's laws.
- **Rotating Frames (ECEF)**: Rotate with Earth. Natural for ground-based observations.
- **Topocentric Frames (ENZ/SEZ)**: Local observer-centered. Natural for antenna pointing.

### Frame Transformations

Transformations between ECI and ECEF require:
- Precession and nutation models (IAU 2006/2000A)
- Earth Orientation Parameters (EOP) for high precision
- Time system conversions (UTC, UT1, TT)

See: [Frame Transformations](../frame_transformations.md)

### Coordinate System Properties

| System | Origin | Axes | Rotating | Best Use Case |
|--------|--------|------|----------|---------------|
| ECI | Earth center | Celestial | No | Orbit propagation |
| ECEF | Earth center | Earth-fixed | Yes | Ground locations |
| Geodetic | Earth surface | Ellipsoid normal | Yes | GPS, ground stations |
| Geocentric | Earth center | Spherical | Yes | Simplified calculations |
| Topocentric | Observer | Local horizontal | Yes | Tracking, visibility |

## Usage Patterns

### Converting Satellite State to Observer View

```python
import brahe as bh
import numpy as np

# Satellite state in ECI
sat_state_eci = np.array([...])  # [x, y, z, vx, vy, vz]

# Convert to ECEF
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
sat_state_ecef = bh.state_eci_to_ecef(sat_state_eci, epoch)

# Observer location (geodetic)
observer_lat = np.radians(40.0)  # 40° N
observer_lon = np.radians(-75.0)  # 75° W
observer_alt = 100.0  # meters

# Convert observer to ECEF
observer_ecef = bh.position_geodetic_to_ecef(observer_lat, observer_lon, observer_alt)

# Compute topocentric position
relative_ecef = sat_state_ecef[:3] - observer_ecef
enz = bh.relative_position_ecef_to_enz(relative_ecef, observer_lat, observer_lon)

# Convert to azimuth/elevation
az, el, range_m = bh.position_enz_to_azel(enz)

print(f"Azimuth: {np.degrees(az):.1f}°")
print(f"Elevation: {np.degrees(el):.1f}°")
print(f"Range: {range_m/1000:.1f} km")
```

### Ground Station Network in Multiple Systems

```python
# Define stations in geodetic (natural input)
stations_geodetic = [
    (np.radians(15.4), np.radians(78.2), 0.0),  # Svalbard
    (np.radians(-64.5), np.radians(-31.5), 0.0),  # Malargue
    (np.radians(-117.2), np.radians(34.1), 500.0),  # Goldstone
]

# Convert to ECEF for computations
stations_ecef = [
    bh.position_geodetic_to_ecef(lat, lon, alt)
    for lat, lon, alt in stations_geodetic
]

# Also compute geocentric for reference
stations_geocentric = [
    bh.position_ecef_to_geocentric(ecef)
    for ecef in stations_ecef
]
```

## Performance Considerations

### Transformation Cost

Coordinate transformations have different computational costs:

- **Cartesian ↔ Geocentric**: Fast (trigonometric functions)
- **Cartesian ↔ Geodetic**: Moderate (iterative algorithm for inverse)
- **ECI ↔ ECEF**: Moderate to expensive (precession/nutation models)
- **ECEF → Topocentric**: Fast (rotation matrix)

### When to Cache

For repeated transformations at the same epoch:
- Cache rotation matrices for ECI ↔ ECEF
- Precompute observer ECEF positions
- Store frequently-used station locations

### Batch Operations

When transforming multiple points:
```python
# Single rotation matrix for all points at same epoch
R = bh.rotation_matrix_ecef_to_eci(epoch)

# Apply to all states
states_eci = [R @ state_ecef[:3] for state_ecef in states_ecef]
```

## Common Pitfalls

### Latitude Types

**Geodetic vs Geocentric latitude**:
- Geodetic: Perpendicular to ellipsoid surface (GPS uses this)
- Geocentric: Angle from Earth center

These differ by up to 0.2° at mid-latitudes!

### Longitude Wrapping

Ensure consistent longitude ranges:
- Use `-180° to +180°` (preferred in Brahe)
- Not `0° to 360°`

### Angle Units

All Brahe functions use radians internally:
```python
# Use AngleFormat enum for clarity
lat_deg = 40.0
lat_rad = np.radians(lat_deg)
pos = bh.position_geodetic_to_ecef(lat_rad, lon_rad, alt)
```

### Frame Consistency

Never mix coordinates from different frames:
```python
# WRONG: Mixing ECI and ECEF
range_vec = sat_state_eci[:3] - observer_ecef  # Error!

# CORRECT: Convert to same frame first
sat_state_ecef = bh.state_eci_to_ecef(sat_state_eci, epoch)
range_vec = sat_state_ecef[:3] - observer_ecef  # OK
```

## See Also

- [Cartesian Transformations](cartesean_transformations.md) - ECI and ECEF Cartesian coordinates
- [Geocentric & Geodetic Transformations](geocentric_transformations.md) - Spherical and ellipsoidal Earth models
- [Topocentric Transformations](topocentric_transformations.md) - Local horizontal coordinate systems
- [Frame Transformations](../frame_transformations.md) - Transformations between reference frames
- [Coordinates API Reference](../../library_api/coordinates/index.md) - Complete coordinate function documentation
