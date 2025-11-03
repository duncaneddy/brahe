# Trajectories

Brahe provides trajectory containers for storing and managing time-series state data. A trajectory is a sequence of state vectors (positions, velocities, or other multi-dimensional data) indexed by time epochs. Trajectories store the dynamic state evolution and provide a number of convenience methods for accessing, querying, and manipulating the data.

## Trajectory Traits

Brahe's trajectory system is built on a set of [Rust traits](https://doc.rust-lang.org/book/ch10-02-traits.html) that define common functionality. This design allows for common access patterns across different trajectory implementations, while enabling specialized behavior for specific use cases.

Generally, a "state" is a vector of floating-point numbers representing some dynamic quantity. For most applications in Brahe, states are 6-dimensional vectors representing the satellite position and velocity in 3D space. However, the trajectory system is flexible enough to handle arbitrary state definitions.

### `Trajectory` Trait

The `Trajectory` trait is the foundation of all trajectory implementations. It defines the core interface for storing, accessing, and managing time-series state data. Any Trajectory implementation must implement this trait which requires the implementation of the following methods:

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

### `Interpolatable` Trait

Since trajectories often store states at discrete epochs, the `Interpolatable` trait provides methods for interpolating states at arbitrary times between stored data points. This is useful for applications that require continuous state estimates.

**Methods**:

- `interpolate(epoch)` - Get interpolated state at arbitrary epoch
- `set_interpolation_method(method)` - Configure interpolation algorithm
- `get_interpolation_method()` - Get current interpolation method

**Supported Interpolation Methods** (via `InterpolationMethod` enum):

- `Linear` - Linear interpolation between adjacent states (default)

### `OrbitalTrajectory` Trait

The `OrbitalTrajectory` trait specializes trajectories for orbital mechanics applications. It adds awareness of reference frames (ECI/ECEF) and orbital representations (Cartesian/Keplerian), enabling automatic conversions of the stored states to different frames or representations.

**Creation**:

- `from_orbital_data(epochs, states, frame, representation, angle_format)` - Create from orbital data with frame/representation metadata

**Frame Conversions**:

- `to_eci()` - Convert all states to Earth-Centered Inertial frame
- `to_ecef()` - Convert all states to Earth-Centered Earth-Fixed frame

**Representation Conversions**:

- `to_keplerian(angle_format)` - Convert Cartesian states to Keplerian orbital elements

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

- `Cartesian` - Position and velocity vectors $\[p_x, p_y, p_z, v_x, v_y, v_z\]$ in meters and m/s
- `Keplerian` - Classical orbital elements $\[a, e, i, \Omega, \omega, M\]$ where:
    - $a$ - Semi-major axis (meters)
    - $e$ - Eccentricity (dimensionless)
    - $i$ - Inclination (radians or degrees)
    - $\Omega$ - Right ascension of ascending node (radians or degrees)
    - $\omega$ - Argument of periapsis (radians or degrees)
    - $M$ - Mean anomaly (radians or degrees)

## Choosing a Trajectory Implementation

Brahe provides three trajectory implementations, each optimized for different use cases:

### DTrajectory - Dynamic Dimensions

The `DTrajectory` implementation supports runtime-sized state vectors, allowing for arbitrary state dimensions. This makes it able to accomodate applications where users may want to augment the state vector with additional parameters beyond standard position/velocity.

**Features**:

- Runtime-sized state vectors (any dimension)
- Frame-agnostic storage
- Flexible for arbitrary state data
- Implements traits: `Trajectory`, `Interpolatable`

### STrajectory<R> - Static Dimensions

The `STrajectory<R>` implementation uses compile-time sized state vectors, providing maximum performance for applications where the state dimension is known ahead of time. The generic parameter `R` specifies the number of state dimensions.

**Features**:

- Compile-time sized state vectors (maximum performance)
- Type-safe dimensions
- Common type aliases: `STrajectory3`, `STrajectory4`, `STrajectory6`
- Implements traits: `Trajectory`, `Interpolatable`

!!! tip
    Because `STrajectory` uses compile-time dimensions python bindings are only provided for common sizes of `STrajectory3`, `STrajectory4`, and `STrajectory6`.

    Rust users can create `STrajectory` instances with any dimension using the generic type.

### OrbitTrajectory - Orbital Mechanics

The `OrbitTrajectory` implementation is specialized for orbital mechanics applications. It always 6-dimensional state vectors (position + velocity or orbital elements) and tracks the reference frame (ECI/ECEF) and representation (Cartesian/Keplerian). It provides built-in methods for converting between frames and representations. The `OrbitTrajectory` is ideal for satellite orbit propagation and analysis where you expect to need frame conversions.

**Features**:

- Always 6-dimensional (position + velocity)
- Tracks reference frame (ECI/ECEF)
- Tracks representation (Cartesian/Keplerian)
- Frame conversions: ECI ↔ ECEF
- Representation conversions: Cartesian ↔ Keplerian
- Implements traits: `Trajectory`, `Interpolatable`, `OrbitalTrajectory`

## See Also

- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory implementation
- [STrajectory6](strajectory6.md) - Static 6D trajectory implementation
- [OrbitTrajectory](orbit_trajectory.md) - Orbital mechanics trajectory with frame conversions
- [API Reference](../../library_api/trajectories/index.md)
