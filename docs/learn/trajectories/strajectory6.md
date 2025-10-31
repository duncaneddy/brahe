# STrajectory6

`STrajectory6` is a static, compile-time sized trajectory container optimized for 6-dimensional state vectors (position + velocity). Unlike `DTrajectory` which determines dimension at runtime, `STrajectory6` uses compile-time sizing for maximum performance and type safety.

Use `STrajectory6` when:

- Working with 6D orbital states (position + velocity)
- State dimension is always fixed at 6
- Performance is critical
- You want compile-time dimension checking

`STrajectory6` is part of the `STrajectory<R>` family which includes `STrajectory3` (3D, position only) and `STrajectory4` (4D, quaternions). For applications requiring frame conversions or Keplerian elements, consider [`OrbitTrajectory`](orbit_trajectory.md).

## Initialization

### Empty Trajectory

Create an empty 6D trajectory:

=== "Python"

    ```python
    import brahe as bh

    # Create empty 6D trajectory
    traj = bh.STrajectory6()
    print(f"Trajectory length: {len(traj)}")  # Output: 0
    print(f"Is empty: {traj.is_empty()}")  # Output: True
    ```

=== "Rust"

    ```rust
    use brahe::trajectories::{STrajectory6, Trajectory};

    // Create empty 6D trajectory
    let traj = STrajectory6::new();
    println!("Trajectory length: {}", traj.len());  // Output: 0
    println!("Is empty: {}", traj.is_empty());  // Output: true
    ```

### From Existing Data

Create a trajectory from existing epochs and 6D states:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create epochs
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch1 = epoch0 + 60.0  # 1 minute later
    epoch2 = epoch0 + 120.0  # 2 minutes later

    # Create 6D states (position + velocity in meters and m/s)
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
    state2 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0])

    # Create trajectory from data
    epochs = [epoch0, epoch1, epoch2]
    states = np.array([state0, state1, state2])
    traj = bh.STrajectory6.from_data(epochs, states)

    print(f"Trajectory length: {len(traj)}")  # Output: 3
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create epochs
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;
    let epoch2 = epoch0 + 120.0;

    // Create 6D states
    let state0 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    let state1 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    );
    let state2 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0
    );

    // Create trajectory from data
    let epochs = vec![epoch0, epoch1, epoch2];
    let states = vec![state0, state1, state2];
    let traj = STrajectory6::from_data(epochs, states).unwrap();

    println!("Trajectory length: {}", traj.len());  // Output: 3
    ```

## Adding and Accessing States

### Adding States

Add 6D states to a trajectory:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create empty trajectory
    traj = bh.STrajectory6()

    # Add states
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch0, state0)

    epoch1 = epoch0 + 60.0
    state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
    traj.add(epoch1, state1)

    print(f"Trajectory length: {len(traj)}")  # Output: 2
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create empty trajectory
    let mut traj = STrajectory6::new();

    // Add states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let state0 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    traj.add(epoch0, state0);

    let epoch1 = epoch0 + 60.0;
    let state1 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    );
    traj.add(epoch1, state1);

    println!("Trajectory length: {}", traj.len());  // Output: 2
    ```

### Accessing by Index

Retrieve states and epochs by their index:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create and populate trajectory
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch0, state0)

    # Access by index
    retrieved_epoch = traj.epoch_at_idx(0)
    retrieved_state = traj.state_at_idx(0)

    print(f"Epoch: {retrieved_epoch}")
    print(f"Position: [{retrieved_state[0]:.2f}, {retrieved_state[1]:.2f}, {retrieved_state[2]:.2f}] m")
    print(f"Velocity: [{retrieved_state[3]:.2f}, {retrieved_state[4]:.2f}, {retrieved_state[5]:.2f}] m/s")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create and populate trajectory
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let state0 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    traj.add(epoch0, state0);

    // Access by index
    let retrieved_epoch = traj.epoch_at_idx(0);
    let retrieved_state = traj.state_at_idx(0);

    println!("Epoch: {}", retrieved_epoch);
    println!("Position: [{:.2}, {:.2}, {:.2}] m",
        retrieved_state[0], retrieved_state[1], retrieved_state[2]);
    println!("Velocity: [{:.2}, {:.2}, {:.2}] m/s",
        retrieved_state[3], retrieved_state[4], retrieved_state[5]);
    ```

### Accessing by Epoch

Get states at or near specific epochs:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with multiple states
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(5):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Get exact match (if exists)
    query_epoch = epoch0 + 120.0  # 2 minutes after start
    state = traj.get(query_epoch)
    if state is not None:
        print(f"Exact match found at altitude: {(state[0] - bh.R_EARTH) / 1e3:.2f} km")

    # Get nearest state
    query_epoch = epoch0 + 125.0  # Between stored epochs
    nearest = traj.nearest_state(query_epoch)
    print(f"Nearest state altitude: {(nearest[0] - bh.R_EARTH) / 1e3:.2f} km")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with multiple states
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Get exact match (if exists)
    let query_epoch = epoch0 + 120.0;
    if let Some(state) = traj.get(query_epoch) {
        println!("Exact match found at altitude: {:.2} km",
            (state[0] - R_EARTH) / 1e3);
    }

    // Get nearest state
    let query_epoch = epoch0 + 125.0;
    let nearest = traj.nearest_state(query_epoch);
    println!("Nearest state altitude: {:.2} km",
        (nearest[0] - R_EARTH) / 1e3);
    ```

## Querying Trajectory Properties

Query the temporal extent and properties of a trajectory:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory spanning 5 minutes
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(6):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Query properties
    print(f"Number of states: {len(traj)}")  # Output: 6
    print(f"Start epoch: {traj.start_epoch()}")
    print(f"End epoch: {traj.end_epoch()}")
    print(f"Timespan: {traj.timespan():.1f} seconds")  # Output: 300.0
    print(f"Is empty: {traj.is_empty()}")  # Output: False

    # Access first and last states
    first_epoch, first_state = traj.first()
    last_epoch, last_state = traj.last()
    print(f"First epoch: {first_epoch}")
    print(f"Last epoch: {last_epoch}")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory spanning 5 minutes
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..6 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Query properties
    println!("Number of states: {}", traj.len());  // Output: 6
    println!("Start epoch: {}", traj.start_epoch());
    println!("End epoch: {}", traj.end_epoch());
    println!("Timespan: {:.1} seconds", traj.timespan());  // Output: 300.0
    println!("Is empty: {}", traj.is_empty());  // Output: false

    // Access first and last states
    let (first_epoch, first_state) = traj.first();
    let (last_epoch, last_state) = traj.last();
    println!("First epoch: {}", first_epoch);
    println!("Last epoch: {}", last_epoch);
    ```

## Interpolation

STrajectory6 supports linear interpolation to estimate states at arbitrary epochs:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with sparse data
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Add states every 60 seconds
    for i in range(3):
        epoch = epoch0 + i * 60.0
        # Simplified motion: position changes linearly with time
        state = np.array([bh.R_EARTH + 500e3 + i * 10000, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Interpolate state at intermediate time
    query_epoch = epoch0 + 30.0  # Halfway between first two states
    interpolated_state = traj.interpolate(query_epoch)

    print(f"Interpolated altitude: {(interpolated_state[0] - bh.R_EARTH) / 1e3:.2f} km")
    # Expected: approximately 505 km (halfway between 500 and 510 km)
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory, Interpolatable};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with sparse data
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add states every 60 seconds
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 10000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Interpolate state at intermediate time
    let query_epoch = epoch0 + 30.0;
    let interpolated_state = traj.interpolate(query_epoch);

    println!("Interpolated altitude: {:.2} km",
        (interpolated_state[0] - R_EARTH) / 1e3);
    // Expected: approximately 505 km (halfway between 500 and 510 km)
    ```

## Memory Management

STrajectory6 supports eviction policies for automatic memory management:

### Maximum Size Policy

Keep only the N most recent states:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with max size limit
    traj = bh.STrajectory6().with_eviction_policy_max_size(3)

    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Add 5 states
    for i in range(5):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Only the 3 most recent states are kept
    print(f"Trajectory length: {len(traj)}")  # Output: 3
    print(f"Start altitude: {(traj.state_at_idx(0)[0] - bh.R_EARTH) / 1e3:.2f} km")
    # Output: ~502 km (states 0 and 1 were evicted)
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with max size limit
    let mut traj = STrajectory6::new()
        .with_eviction_policy_max_size(3);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add 5 states
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Only the 3 most recent states are kept
    println!("Trajectory length: {}", traj.len());  // Output: 3
    println!("Start altitude: {:.2} km",
        (traj.state_at_idx(0)[0] - R_EARTH) / 1e3);
    // Output: ~502 km (states 0 and 1 were evicted)
    ```

### Maximum Age Policy

Keep only states within a time window:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Keep only states within last 2 minutes (120 seconds)
    traj = bh.STrajectory6().with_eviction_policy_max_age(120.0)

    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Add states spanning 4 minutes
    for i in range(5):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Only states within 120 seconds of the most recent are kept
    print(f"Trajectory length: {len(traj)}")  # Output: 3
    print(f"Timespan: {traj.timespan():.1f} seconds")  # Output: ~120.0
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Keep only states within last 2 minutes (120 seconds)
    let mut traj = STrajectory6::new()
        .with_eviction_policy_max_age(120.0);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add states spanning 4 minutes
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Only states within 120 seconds of the most recent are kept
    println!("Trajectory length: {}", traj.len());  // Output: 3
    println!("Timespan: {:.1} seconds", traj.timespan());  // Output: ~120.0
    ```

## Iteration

Trajectories can be iterated to process all epoch-state pairs:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create and populate trajectory
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(3):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Iterate over all states
    for epoch, state in traj:
        altitude = (state[0] - bh.R_EARTH) / 1e3
        velocity = np.linalg.norm(state[3:6])
        print(f"Epoch: {epoch}, Altitude: {altitude:.2f} km, Speed: {velocity:.0f} m/s")
    # Output:
    # Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km, Speed: 7600 m/s
    # Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km, Speed: 7600 m/s
    # Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km, Speed: 7600 m/s
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create and populate trajectory
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Iterate over all states
    for (epoch, state) in &traj {
        let altitude = (state[0] - R_EARTH) / 1e3;
        let velocity = state.fixed_rows::<3>(3).norm();
        println!("Epoch: {}, Altitude: {:.2} km, Speed: {:.0} m/s",
            epoch, altitude, velocity);
    }
    // Output:
    // Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km, Speed: 7600 m/s
    // Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km, Speed: 7600 m/s
    // Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km, Speed: 7600 m/s
    ```

## Matrix Export

Convert trajectory data to matrix format for analysis or export:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory
    traj = bh.STrajectory6()
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(3):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + i * 10, 0.0])
        traj.add(epoch, state)

    # Convert to matrix (rows are states, columns are dimensions)
    matrix = traj.to_matrix()
    print(f"Matrix shape: {matrix.shape}")  # Output: (3, 6)
    print(f"First state velocity: {matrix[0, 4]:.1f} m/s")  # Output: 7600.0
    print(f"Last state velocity: {matrix[2, 4]:.1f} m/s")  # Output: 7620.0
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{STrajectory6, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory
    let mut traj = STrajectory6::new();
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + (i as f64) * 10.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Convert to matrix (rows are states, columns are dimensions)
    let matrix = traj.to_matrix();
    println!("Matrix shape: ({}, {})", matrix.nrows(), matrix.ncols());
    // Output: (3, 6)
    println!("First state velocity: {:.1} m/s", matrix[(0, 4)]);
    // Output: 7600.0
    println!("Last state velocity: {:.1} m/s", matrix[(2, 4)]);
    // Output: 7620.0
    ```

## Performance Benefits

`STrajectory6` uses compile-time sized vectors (`SVector<f64, 6>` in Rust, fixed-size arrays in Python) which provide several benefits:

**Memory Layout**: Contiguous memory allocation enables better cache utilization and SIMD optimizations.

**Type Safety**: Dimension mismatches are caught at compile time in Rust, preventing runtime errors.

**Optimization**: Compilers can optimize operations on fixed-size arrays more aggressively than dynamic allocations.

**Stack Allocation**: Small fixed-size vectors can be allocated on the stack, avoiding heap allocations.

For most orbital mechanics applications with 6D states, `STrajectory6` provides the best balance of performance and ease of use.

## Other Static Trajectory Types

The `STrajectory<R>` family includes other compile-time sized variants:

- **`STrajectory3`**: 3D states (position only, or other 3D data)
- **`STrajectory4`**: 4D states (quaternions for attitude)
- **`STrajectory6`**: 6D states (position + velocity, most common)

All variants share the same API and support the same operations.

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory for variable sizes
- [OrbitTrajectory](orbit_trajectory.md) - Orbital trajectory with frame conversions
- [STrajectory6 API Reference](../../library_api/trajectories/strajectory6.md)
