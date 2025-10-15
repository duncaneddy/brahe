# OrbitTrajectory

Frame-aware orbital trajectory with automatic coordinate transformations.

::: brahe.OrbitTrajectory
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

`OrbitTrajectory` is a specialized trajectory container for orbital mechanics that stores states in a specific reference frame (ECI or ECEF) and can automatically transform between frames when querying.

**Module**: `brahe.trajectories`

**Key Features**:
- Frame-aware storage (ECI or ECEF)
- Automatic frame transformations on query
- Built on STrajectory6 (6D states only)
- Same performance as STrajectory6

## Creating with Frame

```python
import brahe as bh

# ECI frame trajectory
traj_eci = bh.OrbitTrajectory(frame=bh.OrbitFrame.ECI)

# ECEF frame trajectory
traj_ecef = bh.OrbitTrajectory(frame=bh.OrbitFrame.ECEF)
```

## Example Usage

```python
import brahe as bh
import numpy as np

# Create ECI trajectory
traj = bh.OrbitTrajectory(frame=bh.OrbitFrame.ECI)

# Propagate and store states in ECI
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([7000e3, 0.001, 98*bh.DEG2RAD, 0, 0, 0])
prop = bh.KeplerianPropagator(
    epoch=epoch,
    elements=elements,
    frame=bh.OrbitFrame.ECI
)

# Add states
for i in range(100):
    t = epoch + i * 60.0
    state_eci = prop.propagate(t)
    traj.add(t, state_eci)

# States are stored and retrieved in ECI
query_epoch = epoch + 1800.0
state = traj.interpolate(query_epoch)  # ECI frame
```

## Frame Information

```python
# Get trajectory frame
frame = traj.frame()  # Returns OrbitFrame.ECI or OrbitFrame.ECEF
```

## API

OrbitTrajectory has the same API as [STrajectory6](strajectory6.md) and [DTrajectory](dtrajectory.md), plus frame awareness.

See [STrajectory6](strajectory6.md) or [DTrajectory](dtrajectory.md) for full API documentation.

## See Also

- [STrajectory6](strajectory6.md) - Non-frame-aware 6D trajectory
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory
- [OrbitFrame](../orbits/enums.md#orbitframe) - Frame specifications
