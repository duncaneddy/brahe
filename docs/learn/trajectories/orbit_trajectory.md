# OrbitTrajectory

Specialized trajectory for orbital mechanics with multiple state representations.

## Overview

`OrbitTrajectory` provides orbit-specific features beyond basic state storage:

- Automatic Cartesian ↔ Keplerian conversions
- Reference frame tracking (ECI/ECEF)
- Orbit-aware interpolation
- Built-in orbit analysis

## Creating an OrbitTrajectory

```python
import brahe as bh

traj = bh.OrbitTrajectory(
    frame=bh.OrbitFrame.ECI,
    representation=bh.OrbitRepresentation.CARTESIAN,
    interpolation=bh.InterpolationMethod.LINEAR,
    eviction=bh.TrajectoryEvictionPolicy.NONE
)
```

## Reference Frames

Specify which reference frame states are stored in:

- **ECI**: Earth-Centered Inertial (inertial frame)
- **ECEF**: Earth-Centered Earth-Fixed (rotating with Earth)

```python
import brahe as bh

# ECI frame (most common for orbit propagation)
traj_eci = bh.OrbitTrajectory(
    frame=bh.OrbitFrame.ECI,
    representation=bh.OrbitRepresentation.CARTESIAN
)

# ECEF frame (useful for ground tracking)
traj_ecef = bh.OrbitTrajectory(
    frame=bh.OrbitFrame.ECEF,
    representation=bh.OrbitRepresentation.CARTESIAN
)
```

## State Representations

Choose how orbital states are stored:

- **Cartesian**: `[x, y, z, vx, vy, vz]` in meters and m/s
- **Keplerian**: `[a, e, i, Ω, ω, ν]` orbital elements

```python
import brahe as bh

# Store as Cartesian states
traj_cart = bh.OrbitTrajectory(
    frame=bh.OrbitFrame.ECI,
    representation=bh.OrbitRepresentation.CARTESIAN
)

# Store as Keplerian elements
traj_kep = bh.OrbitTrajectory(
    frame=bh.OrbitFrame.ECI,
    representation=bh.OrbitRepresentation.KEPLERIAN
)
```

## Adding Orbital States

```python
import brahe as bh
import numpy as np

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)

# Cartesian state [x, y, z, vx, vy, vz]
state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
traj.add_state(epoch, state)
```

## Querying in Different Representations

```python
import brahe as bh

query_epoch = epoch + 600.0  # 10 minutes later

# Get Cartesian state
cart_state = traj.get_state_cartesian(query_epoch)

# Get Keplerian elements
kep_elements = traj.get_state_keplerian(query_epoch)
a, e, i, raan, argp, nu = kep_elements
```

## Orbit-Specific Features

### Ground Track

```python
import brahe as bh

# Get ground track (lat, lon, alt) over time
ground_track = traj.compute_ground_track(
    start_epoch=epoch,
    end_epoch=epoch + 5400.0,  # 90 minutes
    step=10.0  # Every 10 seconds
)
```

### Orbit Statistics

```python
import brahe as bh

# Compute orbit statistics
stats = traj.get_orbit_statistics()
print(f"Mean altitude: {stats.mean_altitude / 1000:.1f} km")
print(f"Eccentricity: {stats.mean_eccentricity:.6f}")
```

## Use Cases

`OrbitTrajectory` is ideal for:

- Orbit visualization and analysis
- Multi-representation workflows
- Ground track computation
- Orbit determination

## See Also

- [OrbitTrajectory API Reference](../../library_api/trajectories/orbit_trajectory.md)
- [Trajectories Overview](index.md)
- [Orbit Propagation](../orbit_propagation/index.md)
