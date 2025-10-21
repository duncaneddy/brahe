# Cartesian Coordinate Transformations

Cartesian coordinates represent positions and velocities in 3D space using orthogonal axes.

## Overview

Cartesian coordinates are the foundation for orbital mechanics computations in Brahe. Positions and velocities are represented as:

- **Position**: `[x, y, z]` in meters
- **Velocity**: `[vx, vy, vz]` in meters/second
- **State**: `[x, y, z, vx, vy, vz]`

## Coordinate Frames

Brahe works with Cartesian coordinates in different reference frames:

- **ECI (Earth-Centered Inertial)**: Inertial frame fixed to stars
- **ECEF (Earth-Centered Earth-Fixed)**: Rotates with Earth
- **Local frames**: Relative to ground stations or spacecraft

## Converting Between Frames

Use the frame transformation functions to convert between ECI and ECEF:

```python
import brahe as bh

# Convert from ECI to ECEF
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

# Convert from ECEF to ECI
state_eci = bh.state_ecef_to_eci(epoch, state_ecef)
```

## See Also

- [Frame Transformations](../frame_transformations.md)
- [Cartesian API Reference](../../library_api/coordinates/cartesian.md)
- [Frames API Reference](../../library_api/frames.md)
