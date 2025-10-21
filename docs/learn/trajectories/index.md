# Trajectories

Brahe provides trajectory containers for storing and managing time-series state data.

## Overview

Trajectories store sequences of states (positions and velocities) indexed by time. They support:

- Interpolation between time points
- Efficient storage
- Automatic eviction policies
- Multiple representations (Cartesian, Keplerian)

## Trajectory Types

Brahe offers three trajectory implementations:

### DTrajectory

**Dynamic-dimension trajectory** for arbitrary state sizes:

- Runtime-configurable dimension
- Flexible for various applications
- Slightly more overhead than static types

**Use when**: State dimension varies or is determined at runtime.

### STrajectory6

**Static 6-dimensional trajectory** optimized for orbital states:

- Fixed 6D states `[x, y, z, vx, vy, vz]`
- Compile-time optimization
- Lower memory overhead
- Fastest performance

**Use when**: Storing standard Cartesian orbital states.

### OrbitTrajectory

**Specialized trajectory for orbital mechanics**:

- Stores states in multiple representations (Cartesian, Keplerian)
- Automatic frame tracking (ECI/ECEF)
- Orbit-specific interpolation
- Ideal for orbit analysis

**Use when**: Performing orbital mechanics analysis and visualization.

## Common Features

All trajectory types support:

- **Interpolation**: Linear, cubic, or Lagrange
- **Eviction policies**: None, LRU (least recently used), time-based
- **Time indexing**: Fast lookup by epoch
- **State queries**: Get state at specific times

## Choosing a Trajectory Type

```
┌─────────────────────────────────────┐
│ Need orbit-specific features?      │
│ (Keplerian elements, frame info)   │
└───────────┬─────────────────────────┘
            │
    ┌───────┴────────┐
    │ Yes            │ No
    │                │
    ▼                ▼
OrbitTrajectory   ┌────────────────────────┐
                  │ Fixed 6D state?        │
                  └───────┬────────────────┘
                          │
                  ┌───────┴────────┐
                  │ Yes            │ No
                  │                │
                  ▼                ▼
              STrajectory6    DTrajectory
```

## See Also

- [DTrajectory](dtrajectory.md)
- [OrbitTrajectory](orbit_trajectory.md)
- [API Reference](../../library_api/trajectories/index.md)
