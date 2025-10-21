# DTrajectory

Dynamic-dimension trajectory for variable-size state vectors.

## Overview

`DTrajectory` stores time-series state data where the state dimension is determined at runtime. It provides flexibility for applications with varying state sizes.

## Creating a DTrajectory

```python
import brahe as bh
import numpy as np

# Create trajectory with 6D states
traj = bh.DTrajectory(
    dimension=6,
    interpolation=bh.InterpolationMethod.LINEAR,
    eviction=bh.TrajectoryEvictionPolicy.NONE
)
```

## Adding States

```python
import brahe as bh
import numpy as np

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

traj.add_state(epoch, state)

# Add multiple states
for i in range(10):
    epoch_i = epoch + i * 60.0  # Every minute
    state_i = propagate_state(state, i * 60.0)  # Your propagation function
    traj.add_state(epoch_i, state_i)
```

## Querying States

```python
import brahe as bh

# Get state at specific epoch
query_epoch = epoch + 300.0  # 5 minutes later
interpolated_state = traj.get_state(query_epoch)

# Get state without interpolation (nearest)
nearest_state = traj.get_state_nearest(query_epoch)
```

## Interpolation Methods

- **None**: Return exact matches only
- **Linear**: Linear interpolation between points
- **Cubic**: Cubic spline interpolation
- **Lagrange**: Lagrange polynomial interpolation

```python
import brahe as bh

traj = bh.DTrajectory(
    dimension=6,
    interpolation=bh.InterpolationMethod.CUBIC,
    eviction=bh.TrajectoryEvictionPolicy.NONE
)
```

## Eviction Policies

Control memory usage by automatically removing old states:

```python
import brahe as bh

# Keep all states
traj = bh.DTrajectory(6, bh.InterpolationMethod.LINEAR, bh.TrajectoryEvictionPolicy.NONE)

# Keep only most recent N states
traj = bh.DTrajectory(6, bh.InterpolationMethod.LINEAR, bh.TrajectoryEvictionPolicy.LRU)
traj.set_max_size(1000)  # Keep last 1000 states
```

## Use Cases

`DTrajectory` is ideal for:

- Extended state vectors (state + covariance)
- Multi-spacecraft formations
- Custom dynamics models
- Variable-dimension applications

## See Also

- [DTrajectory API Reference](../../library_api/trajectories/dtrajectory.md)
- [Trajectories Overview](index.md)
