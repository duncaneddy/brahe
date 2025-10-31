# DTrajectory

`DTrajectory` is a dynamic-dimension trajectory container that stores time-series state data with runtime-determined dimensions. Unlike static trajectory types, `DTrajectory` allows you to specify the state vector dimension at creation time, making it ideal for applications where the dimension varies or is not known at compile time.

Use `DTrajectory` when:

- State dimension is determined at runtime
- You need flexibility to work with different dimensions in the same codebase
- State vectors are non-standard (not 3D or 6D)
- Flexibility is prioritized over maximum performance

For fixed-dimension orbital mechanics applications, consider using [`STrajectory6`](strajectory6.md) or [`OrbitTrajectory`](orbit_trajectory.md) instead for better performance.

## Initialization

### Empty Trajectory

Create an empty trajectory by specifying the state dimension. The default dimension is 6 (suitable for position + velocity states):

=== "Python"

    ```python
    import brahe as bh

    # Create 6D trajectory (default)
    traj = bh.DTrajectory()
    print(f"Dimension: {traj.dimension}")  # Output: 6

    # Create 3D trajectory (position only)
    traj_3d = bh.DTrajectory(3)
    print(f"Dimension: {traj_3d.dimension}")  # Output: 3

    # Create 12D trajectory (custom)
    traj_12d = bh.DTrajectory(12)
    print(f"Dimension: {traj_12d.dimension}")  # Output: 12
    ```

=== "Rust"

    ```rust
    use brahe::trajectories::DTrajectory;

    // Create 6D trajectory (default)
    let traj = DTrajectory::default();
    println!("Dimension: {}", traj.dimension);  // Output: 6

    // Create 3D trajectory
    let traj_3d = DTrajectory::new(3);
    println!("Dimension: {}", traj_3d.dimension);  // Output: 3

    // Create 12D trajectory
    let traj_12d = DTrajectory::new(12);
    println!("Dimension: {}", traj_12d.dimension);  // Output: 12
    ```

### From Existing Data

Create a trajectory from existing epochs and states:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create epochs
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch1 = epoch0 + 60.0  # 1 minute later
    epoch2 = epoch0 + 120.0  # 2 minutes later

    # Create states (6D: position + velocity)
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
    state2 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0])

    # Create trajectory from data
    epochs = [epoch0, epoch1, epoch2]
    states = np.array([state0, state1, state2])
    traj = bh.DTrajectory.from_data(epochs, states)

    print(f"Trajectory length: {len(traj)}")  # Output: 3
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create epochs
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;  // 1 minute later
    let epoch2 = epoch0 + 120.0;  // 2 minutes later

    // Create states
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    let state1 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    ]);
    let state2 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0
    ]);

    // Create trajectory from data
    let epochs = vec![epoch0, epoch1, epoch2];
    let states = vec![state0, state1, state2];
    let traj = DTrajectory::from_data(epochs, states);

    println!("Trajectory length: {}", traj.len());  // Output: 3
    ```

## Adding and Accessing States

### Adding States

Add states to a trajectory one at a time:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create empty trajectory
    traj = bh.DTrajectory(6)

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
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create empty trajectory
    let mut traj = DTrajectory::new(6);

    // Add states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    traj.add(epoch0, state0);

    let epoch1 = epoch0 + 60.0;
    let state1 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    ]);
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
    traj = bh.DTrajectory(6)
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj.add(epoch0, state0)

    # Access by index
    retrieved_epoch = traj.epoch_at_idx(0)
    retrieved_state = traj.state_at_idx(0)

    print(f"Epoch: {retrieved_epoch}")
    print(f"Position: {retrieved_state[0]:.2f} m")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create and populate trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    traj.add(epoch0, state0);

    // Access by index
    let retrieved_epoch = traj.epoch_at_idx(0);
    let retrieved_state = traj.state_at_idx(0);

    println!("Epoch: {}", retrieved_epoch);
    println!("Position: {:.2} m", retrieved_state[0]);
    ```

### Accessing by Epoch

Get states at or near specific epochs:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with multiple states
    traj = bh.DTrajectory(6)
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
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with multiple states
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
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

### Time Span and Bounds

Query the temporal extent of a trajectory:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory spanning 5 minutes
    traj = bh.DTrajectory(6)
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
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory spanning 5 minutes
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..6 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
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

DTrajectory supports linear interpolation to estimate states at arbitrary epochs between stored data points:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with sparse data
    traj = bh.DTrajectory(6)
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
    use brahe::trajectories::{DTrajectory, Trajectory, Interpolatable};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with sparse data
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add states every 60 seconds
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 10000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Interpolate state at intermediate time
    let query_epoch = epoch0 + 30.0;  // Halfway between first two states
    let interpolated_state = traj.interpolate(query_epoch);

    println!("Interpolated altitude: {:.2} km",
        (interpolated_state[0] - R_EARTH) / 1e3);
    // Expected: approximately 505 km (halfway between 500 and 510 km)
    ```

## Memory Management

DTrajectory supports eviction policies to automatically manage memory in long-running applications:

### Maximum Size Policy

Keep only the N most recent states:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory with max size limit
    traj = bh.DTrajectory(6).with_eviction_policy_max_size(3)

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
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory with max size limit
    let mut traj = DTrajectory::new(6)
        .with_eviction_policy_max_size(3);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add 5 states
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
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
    traj = bh.DTrajectory(6).with_eviction_policy_max_age(120.0)

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
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Keep only states within last 2 minutes (120 seconds)
    let mut traj = DTrajectory::new(6)
        .with_eviction_policy_max_age(120.0);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    // Add states spanning 4 minutes
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
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
    traj = bh.DTrajectory(6)
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(3):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3 + i * 1000, 0.0, 0.0, 0.0, 7600.0, 0.0])
        traj.add(epoch, state)

    # Iterate over all states
    for epoch, state in traj:
        altitude = (state[0] - bh.R_EARTH) / 1e3
        print(f"Epoch: {epoch}, Altitude: {altitude:.2f} km")
    # Output:
    # Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km
    # Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km
    # Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create and populate trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Iterate over all states
    for (epoch, state) in &traj {
        let altitude = (state[0] - R_EARTH) / 1e3;
        println!("Epoch: {}, Altitude: {:.2} km", epoch, altitude);
    }
    // Output:
    // Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km
    // Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km
    // Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km
    ```

## Matrix Export

Convert trajectory data to matrix format for analysis or export:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory
    traj = bh.DTrajectory(6)
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    for i in range(3):
        epoch = epoch0 + i * 60.0
        state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + i * 10, 0.0])
        traj.add(epoch, state)

    # Convert to matrix (rows are states, columns are dimensions)
    matrix = traj.to_matrix()
    print(f"Matrix shape: {matrix.shape}")  # Output: (3, 6)
    print(f"First state velocity: {matrix[0, 4]:.1f} m/s")  # Output: 7600.0
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{DTrajectory, Trajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + (i as f64) * 10.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Convert to matrix (rows are states, columns are dimensions)
    let matrix = traj.to_matrix();
    println!("Matrix shape: ({}, {})", matrix.nrows(), matrix.ncols());
    // Output: (3, 6)
    println!("First state velocity: {:.1} m/s", matrix[(0, 4)]);
    // Output: 7600.0
    ```

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [STrajectory6](strajectory6.md) - Fixed 6D trajectory for better performance
- [OrbitTrajectory](orbit_trajectory.md) - Orbital trajectory with frame conversions
- [DTrajectory API Reference](../../library_api/trajectories/dtrajectory.md)
