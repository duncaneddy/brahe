# STrajectory6

Static 6-dimensional trajectory container optimized for orbital state vectors.

::: brahe.STrajectory6
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

`STrajectory6` is a trajectory container optimized for 6-dimensional state vectors [x, y, z, vx, vy, vz]. The dimension is fixed at compile time, providing better performance than DTrajectory for orbital mechanics applications.

**Module**: `brahe.trajectories`

**Use When**:
- Storing orbital state vectors (position + velocity)
- Performance is critical
- Dimension is always 6

**Advantages over DTrajectory**:
- Faster operations (no runtime dimension checks)
- More memory efficient
- Same API for state management

## Example Usage

```python
import brahe as bh
import numpy as np

# Create trajectory
traj = bh.STrajectory6(
    interpolation_method=bh.InterpolationMethod.LINEAR
)

# Add states from propagation
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
prop = bh.KeplerianPropagator(...)

for i in range(1000):
    t = epoch + i * 60.0  # Every minute
    state = prop.propagate(t)  # Returns [x,y,z,vx,vy,vz]
    traj.add(t, state)

# Query with interpolation
query_time = epoch + 1800.0
interp_state = traj.interpolate(query_time)

print(f"Position: {interp_state[:3]} m")
print(f"Velocity: {interp_state[3:]} m/s")
```

## API

STrajectory6 has the same methods as [DTrajectory](dtrajectory.md):
- `add()`, `interpolate()`, `state()`, `epoch()`
- `first()`, `last()`, `len()`, `is_empty()`
- `set_eviction_policy_*()`, `set_interpolation_method()`
- `to_matrix()`, `to_epochs()`, `clear()`

See [DTrajectory](dtrajectory.md) for detailed API documentation.

## See Also

- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory
- [OrbitTrajectory](orbit_trajectory.md) - Frame-aware orbital trajectory
