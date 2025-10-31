# Trajectories

Brahe provides trajectory containers for storing and managing time-series state data. A trajectory is a sequence of state vectors (positions, velocities, or other multi-dimensional data) indexed by time epochs. Trajectories are essential for storing propagation results, analyzing orbital history, performing interpolation, and converting between different reference frames.

## Trajectory Traits

Brahe's trajectory system is built on a hierarchy of traits that define core functionality. This design allows you to choose the right level of functionality for your application while maintaining a consistent interface.

### `Trait`: Trajectory

The `Trajectory` trait is the foundation of all trajectory implementations. It defines the core interface for storing, accessing, and managing time-series state data.

**Purpose**: Provides basic storage and retrieval of state vectors indexed by time epochs.

**Key Methods**:

**Creation**:

- `from_data(epochs, states)` - Create trajectory from vectors of epochs and states
- `add(epoch, state)` - Add a single state at a specific epoch

**Access**:

- `epoch_at_idx(index)` - Get epoch at a specific index
- `state_at_idx(index)` - Get state at a specific index
- `get(epoch)` - Get exact state if epoch exists
- `nearest_state(epoch)` - Get state at nearest epoch to query time

**Querying**:

- `len()` - Number of states in trajectory
- `is_empty()` - Check if trajectory contains no states
- `start_epoch()` - First epoch in trajectory
- `end_epoch()` - Last epoch in trajectory
- `timespan()` - Duration between first and last epochs (seconds)
- `first()` - Get first epoch-state pair
- `last()` - Get last epoch-state pair

**Modification**:

- `clear()` - Remove all states
- `remove_epoch(epoch)` - Remove state at specific epoch
- `remove(index)` - Remove state at specific index

**Temporal Indexing**:

- `index_before_epoch(epoch)` - Find index of state before query epoch
- `index_after_epoch(epoch)` - Find index of state after query epoch
- `state_before_epoch(epoch)` - Get state before query epoch
- `state_after_epoch(epoch)` - Get state after query epoch

**Memory Management**:

- `set_eviction_policy_max_size(size)` - Limit trajectory to N most recent states
- `set_eviction_policy_max_age(duration_seconds)` - Keep only states within time window
- `get_eviction_policy()` - Get current eviction policy

**When to Use**: Implement this trait when you need basic trajectory storage without interpolation or orbital-specific features.

### `Trait`: Interpolatable

The `Interpolatable` trait adds interpolation capabilities to trajectories, allowing retrieval of state estimates at arbitrary epochs between stored data points.

**Purpose**: Enables interpolation of states at arbitrary times, not just at stored epochs.

**Key Methods**:

- `interpolate(epoch)` - Get interpolated state at arbitrary epoch
- `set_interpolation_method(method)` - Configure interpolation algorithm
- `get_interpolation_method()` - Get current interpolation method

**Supported Methods** (via `InterpolationMethod` enum):

- `Linear` - Linear interpolation between adjacent states (default)

**When to Use**: Implement this trait when you need to query trajectory states at arbitrary times, not just at stored epochs.

**Requires**: Must also implement `Trajectory` trait.

### `Trait`: OrbitalTrajectory

The `OrbitalTrajectory` trait specializes trajectories for orbital mechanics applications. It adds awareness of reference frames (ECI/ECEF) and orbital representations (Cartesian/Keplerian), enabling automatic conversions.

**Purpose**: Provides orbital mechanics-specific conversions between reference frames and orbital element representations.

**Key Methods**:

**Creation**:

- `from_orbital_data(epochs, states, frame, representation, angle_format)` - Create from orbital data with frame/representation metadata

**Frame Conversions**:

- `to_eci()` - Convert all states to Earth-Centered Inertial frame
- `to_ecef()` - Convert all states to Earth-Centered Earth-Fixed frame

**Representation Conversions**:

- `to_keplerian(angle_format)` - Convert Cartesian states to Keplerian orbital elements

**When to Use**: Use this trait when working with orbital mechanics and need to convert between ECI/ECEF frames or Cartesian/Keplerian representations.

**Requires**: Must also implement `Trajectory` and `Interpolatable` traits. States must be 6-dimensional (position + velocity).

## Supporting Types

### InterpolationMethod

Defines interpolation algorithms available for computing states at arbitrary epochs:

- `Linear` - Linear interpolation between adjacent state vectors (default)

### TrajectoryEvictionPolicy

Controls automatic memory management for long-running applications:

- `None` - Keep all states indefinitely (default)
- `KeepCount(n)` - Keep only the N most recent states, removing older ones
- `KeepWithinDuration(seconds)` - Keep only states within time window from most recent epoch

Eviction policies are useful for real-time applications where memory must be bounded, such as satellite ground station passes or long-term simulations.

### OrbitFrame

Specifies the reference frame for orbital states:

- `ECI` - Earth-Centered Inertial frame (GCRF/J2000)
- `ECEF` - Earth-Centered Earth-Fixed frame

### OrbitRepresentation

Specifies how orbital states are represented:

- `Cartesian` - Position and velocity vectors `[x, y, z, vx, vy, vz]` in meters and m/s
- `Keplerian` - Classical orbital elements `[a, e, i, Ω, ω, ν]` where:
    - `a` - Semi-major axis (meters)
    - `e` - Eccentricity (dimensionless)
    - `i` - Inclination (radians or degrees)
    - `Ω` - Right ascension of ascending node (radians or degrees)
    - `ω` - Argument of periapsis (radians or degrees)
    - `ν` - True anomaly (radians or degrees)

## Choosing a Trajectory Implementation

Brahe provides three trajectory implementations, each optimized for different use cases:

### DTrajectory - Dynamic Dimensions

**Use when**: State dimension is not known at compile time, or varies between different trajectories.

**Features**:

- Runtime-sized state vectors (any dimension)
- Frame-agnostic storage
- Flexible for arbitrary state data
- Implements: `Trajectory`, `Interpolatable`

**Best for**: Applications with varying state dimensions, non-standard state vectors, or when flexibility is prioritized over performance.

### STrajectory<R> - Static Dimensions

**Use when**: State dimension is known at compile time (common case for orbital mechanics).

**Features**:

- Compile-time sized state vectors (maximum performance)
- Type-safe dimensions
- Frame-agnostic storage
- Common type aliases: `STrajectory3`, `STrajectory4`, `STrajectory6`
- Implements: `Trajectory`, `Interpolatable`

**Best for**: Performance-critical applications where state dimension is fixed. Use `STrajectory6` for orbital position/velocity states.

### OrbitTrajectory - Orbital Mechanics

**Use when**: Working with orbital mechanics and need frame/representation conversions.

**Features**:

- Always 6-dimensional (position + velocity)
- Tracks reference frame (ECI/ECEF)
- Tracks representation (Cartesian/Keplerian)
- Frame conversions: ECI ↔ ECEF
- Representation conversions: Cartesian ↔ Keplerian
- Implements: `Trajectory`, `Interpolatable`, `OrbitalTrajectory`

**Best for**: Orbital mechanics applications requiring frame conversions, propagation result storage, or working with both Cartesian and Keplerian representations.

**Quick Decision Guide**:

```
Need frame conversions (ECI/ECEF)?
├─ Yes → OrbitTrajectory
└─ No
   ├─ State dimension known at compile time?
   │  ├─ Yes → STrajectory<R> (e.g., STrajectory6 for 6D states)
   │  └─ No → DTrajectory
   └─ Need maximum performance with fixed dimension?
      └─ Yes → STrajectory<R>
```

## See Also

- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory implementation
- [STrajectory6](strajectory6.md) - Static 6D trajectory implementation
- [OrbitTrajectory](orbit_trajectory.md) - Orbital mechanics trajectory with frame conversions
- [API Reference](../../library_api/trajectories/index.md)
