# Trajectories

**Module**: `brahe.trajectories`

Trajectory containers for storing, managing, and interpolating time-series state data.

## Overview

Brahe provides several trajectory container types for storing sequences of states (positions, velocities, or other data) over time with automatic interpolation capabilities.

## Trajectory Types

### [DTrajectory](dtrajectory.md)
**Dynamic-dimension** trajectory container where dimension is set at runtime. Flexible for storing any N-dimensional state data.

### [STrajectory6](strajectory6.md)
**Static 6-dimensional** trajectory optimized for orbital state vectors [x, y, z, vx, vy, vz]. Faster than DTrajectory for fixed-size data.

### [OrbitTrajectory](orbit_trajectory.md)
**Specialized orbital** trajectory with frame-aware storage and automatic coordinate transformations.

## Key Features

- **Time-ordered storage**: States automatically sorted by epoch
- **Interpolation**: Linear or Lagrange interpolation between states
- **Eviction policies**: Automatic state removal based on age or count
- **Query methods**: Get states before/after/at specific times
- **Batch operations**: Add and query multiple states efficiently

## Quick Comparison

| Feature | DTrajectory | STrajectory6 | OrbitTrajectory |
|---------|-------------|--------------|-----------------|
| Dimension | Runtime (any N) | Compile-time (6) | Compile-time (6) |
| Performance | Good | Better | Better |
| Use Case | General data | Orbital states | Frame-aware orbits |
| Frames | Not frame-aware | Not frame-aware | ECI/ECEF support |

## Usage Example

```python
import brahe as bh
import numpy as np

# Create epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Option 1: Dynamic trajectory (any dimension)
traj_dyn = bh.DTrajectory(dimension=6)

# Option 2: Static 6D trajectory (faster for orbital states)
traj_static = bh.STrajectory6()

# Option 3: Orbit trajectory (frame-aware)
traj_orbit = bh.OrbitTrajectory(frame=bh.OrbitFrame.ECI)

# Add states
state = np.array([7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])  # [x,y,z,vx,vy,vz]
traj_static.add(epoch, state)

# Propagate orbit and store trajectory
prop = bh.KeplerianPropagator(...)
times = np.linspace(0, 86400, 100)
for dt in times:
    future_epoch = epoch + dt
    state = prop.propagate(future_epoch)
    traj_static.add(future_epoch, state)

# Query with interpolation
query_epoch = epoch + 43200.0  # 12 hours later
interpolated_state = traj_static.interpolate(query_epoch)
```

---

## See Also

- [InterpolationMethod](../orbits/enums.md#interpolationmethod) - Interpolation options
- [OrbitFrame](../orbits/enums.md#orbitframe) - Frame specifications
- [KeplerianPropagator](../propagators/keplerian_propagator.md) - Analytical orbit propagation
- [SGPPropagator](../propagators/sgp_propagator.md) - SGP4/SDP4 orbit propagation
