# DTrajectory

Dynamic-dimension trajectory container for N-dimensional state data.

::: brahe.DTrajectory
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

`DTrajectory` is a flexible trajectory container where the state dimension is determined at runtime. Use this when you need to store state vectors of varying sizes or when the dimension isn't known at compile time.

**Module**: `brahe.trajectories`

**Key Features**:
- Runtime dimension specification
- Automatic time-ordering of states
- Configurable interpolation (Linear or Lagrange)
- Eviction policies for memory management
- Efficient state queries and interpolation

## Creating a Trajectory

```python
import brahe as bh

# Create with specified dimension
traj = bh.DTrajectory(dimension=6)  # For orbital states

# Create with specific interpolation method
traj = bh.DTrajectory(
    dimension=3,
    interpolation_method=bh.InterpolationMethod.LAGRANGE
)

# Create from existing data
import numpy as np
epochs = [epoch1, epoch2, epoch3]
states = np.array([[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]])
traj = bh.DTrajectory.from_data(
    epochs,
    states,
    interpolation_method=bh.InterpolationMethod.LINEAR
)
```

## Adding States

```python
import brahe as bh
import numpy as np

traj = bh.DTrajectory(dimension=6)

# Add single state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state = np.array([7e6, 0, 0, 0, 7.5e3, 0])
traj.add(epoch, state)

# Add multiple states from propagation
prop = bh.KeplerianPropagator(...)
for i in range(100):
    future_epoch = epoch + i * 60.0  # Every minute
    state = prop.propagate(future_epoch)
    traj.add(future_epoch, state)
```

## Querying States

```python
# Interpolate at specific epoch
query_epoch = epoch + 1800.0  # 30 minutes later
interp_state = traj.interpolate(query_epoch)

# Get state at specific index
state_10 = traj.state(10)
epoch_10 = traj.epoch(10)

# Get first and last states
first_epoch, first_state = traj.first()
last_epoch, last_state = traj.last()

# Get state before/after epoch
before_epoch, before_state = traj.state_before_epoch(query_epoch)
after_epoch, after_state = traj.state_after_epoch(query_epoch)

# Get all data
all_states = traj.to_matrix()  # Returns numpy array (n_states, dimension)
all_epochs = traj.to_epochs()  # Returns list of Epochs
```

## Eviction Policies

Control memory usage by automatically removing old states:

```python
# Maximum age: keep only states within 1 hour of newest
traj.set_eviction_policy_max_age(3600.0)

# Maximum size: keep only last 1000 states
traj.set_eviction_policy_max_size(1000)

# No eviction (default)
traj.set_eviction_policy_no_eviction()

# Builder pattern (method chaining)
traj = bh.DTrajectory(dimension=6) \
    .with_eviction_policy_max_age(3600.0) \
    .with_interpolation_method(bh.InterpolationMethod.LAGRANGE)
```

## Interpolation Methods

```python
# Linear interpolation (faster, less accurate)
traj.set_interpolation_method(bh.InterpolationMethod.LINEAR)

# Lagrange interpolation (slower, more accurate)
traj.set_interpolation_method(bh.InterpolationMethod.LAGRANGE)
```

## Trajectory Information

```python
# Get dimension
dim = traj.dimension()  # Returns dimension of state vectors

# Get size
n_states = traj.len()  # Number of states stored

# Check if empty
is_empty = traj.is_empty()

# Get time span
span = traj.timespan()  # Duration in seconds from first to last

# Get start/end epochs
start = traj.start_epoch()
end = traj.end_epoch()
```

## Clearing and Removing States

```python
# Clear all states
traj.clear()

# Remove state at specific epoch
removed_state = traj.remove_epoch(epoch)

# Remove state at index
removed_epoch, removed_state = traj.remove_at(index)
```

## Complete Example

```python
import brahe as bh
import numpy as np
import matplotlib.pyplot as plt

# Set up propagator
epoch_start = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([7000e3, 0.001, 98*bh.DEG2RAD, 0, 0, 0])
prop = bh.KeplerianPropagator(
    epoch=epoch_start,
    elements=elements,
    element_type=bh.OrbitRepresentation.MEAN_ELEMENTS,
    frame=bh.OrbitFrame.ECI
)

# Create trajectory with eviction policy
traj = bh.DTrajectory(dimension=6) \
    .with_interpolation_method(bh.InterpolationMethod.LINEAR) \
    .with_eviction_policy_max_size(1000)

# Propagate and store states
times = np.linspace(0, 86400, 1440)  # 1 day, 1-minute steps
for dt in times:
    epoch = epoch_start + dt
    state = prop.propagate(epoch)
    traj.add(epoch, state)

print(f"Stored {traj.len()} states")
print(f"Time span: {traj.timespan()/3600:.1f} hours")

# Interpolate at arbitrary times
query_times = np.linspace(0, 86400, 100)
altitudes = []
for dt in query_times:
    query_epoch = epoch_start + dt
    state = traj.interpolate(query_epoch)
    altitude = (np.linalg.norm(state[:3]) - bh.R_EARTH) / 1000  # km
    altitudes.append(altitude)

# Plot altitude profile
plt.plot(query_times/3600, altitudes)
plt.xlabel('Time (hours)')
plt.ylabel('Altitude (km)')
plt.title('Orbit Altitude Over 1 Day')
plt.grid(True)
plt.show()
```

## See Also

- [STrajectory6](strajectory6.md) - Fixed 6D trajectory (faster)
- [OrbitTrajectory](orbit_trajectory.md) - Frame-aware orbital trajectory
- [InterpolationMethod](../orbits/enums.md#interpolationmethod)
