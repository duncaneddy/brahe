# Keplerian Propagation

The `KeplerianPropagator` provides fast, analytical two-body orbital propagation using Kepler's equations. It assumes only gravitational attraction from a central body (Earth) with no perturbations, making it ideal for rapid trajectory generation, high-altitude orbits, or when perturbations are negligible.

For complete API documentation, see the [KeplerianPropagator API Reference](../../library_api/propagators/keplerian_propagator.md).

## Initialization

The `KeplerianPropagator` can be initialized from several state representations. The initialization method determines the input format, but all methods support configurable output formats.

### From Keplerian Elements

The most direct initialization method uses classical Keplerian orbital elements.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Define initial epoch
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Define Keplerian elements [a, e, i, Ω, ω, M]
    elements = np.array([
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.001,               # Eccentricity
        97.8,                # Inclination (degrees)
        15.0,                # RAAN (degrees)
        30.0,                # Argument of perigee (degrees)
        45.0                 # Mean anomaly (degrees)
    ])

    # Create propagator with 60-second step size
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    print(f"Orbital period: {bh.orbital_period(elements[0]):.1f} seconds")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    // Define initial epoch
    let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Define Keplerian elements [a, e, i, Ω, ω, M]
    let elements = Vector6::new(
        R_EARTH + 500e3,  // Semi-major axis (m)
        0.001,            // Eccentricity
        97.8,             // Inclination (degrees)
        15.0,             // RAAN (degrees)
        30.0,             // Argument of perigee (degrees)
        45.0              // Mean anomaly (degrees)
    );

    // Create propagator with 60-second step size
    let prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    println!("Orbital period: {:.1} seconds", orbital_period(elements[0]));
    ```

### From ECI Cartesian State

Initialize from position and velocity vectors in the Earth-Centered Inertial (ECI) frame.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Define Cartesian state in ECI frame [x, y, z, vx, vy, vz]
    # Convert from Keplerian elements for this example
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    state_eci = bh.state_osculating_to_cartesian(elements, bh.AngleFormat.DEGREES)

    # Create propagator from ECI state
    prop = bh.KeplerianPropagator.from_eci(epoch, state_eci, 60.0)

    print(f"Initial position magnitude: {np.linalg.norm(state_eci[:3]) / 1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Define Cartesian state in ECI frame [x, y, z, vx, vy, vz]
    // Convert from Keplerian elements for this example
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state_eci = state_osculating_to_cartesian(elements, AngleFormat::Degrees);

    // Create propagator from ECI state
    let prop = KeplerianPropagator::from_eci(epoch, state_eci, 60.0);

    println!("Initial position magnitude: {:.1} km",
             state_eci.fixed_rows::<3>(0).norm() / 1e3);
    ```

### From ECEF Cartesian State

Initialize from position and velocity vectors in the Earth-Centered Earth-Fixed (ECEF) frame. The propagator will automatically convert to ECI internally.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    bh.initialize_eop()  # Required for ECEF ↔ ECI transformations

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Get state in ECI, then convert to ECEF for demonstration
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    state_eci = bh.state_osculating_to_cartesian(elements, bh.AngleFormat.DEGREES)
    state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

    # Create propagator from ECEF state
    prop = bh.KeplerianPropagator.from_ecef(epoch, state_ecef, 60.0)
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    initialize_eop().unwrap();  // Required for ECEF ↔ ECI transformations

    let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Get state in ECI, then convert to ECEF for demonstration
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state_eci = state_osculating_to_cartesian(elements, AngleFormat::Degrees);
    let state_ecef = state_eci_to_ecef(epoch, state_eci);

    // Create propagator from ECEF state
    let prop = KeplerianPropagator::from_ecef(epoch, state_ecef, 60.0);
    ```

## Stepping Through Time

The propagator accumulates a trajectory as you step forward in time. Each stepping operation adds new state(s) to the internal trajectory.

### Single Steps

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create propagator
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Take one step (60 seconds)
    prop.step()
    print(f"After 1 step: {prop.current_epoch}")

    # Step by custom duration (120 seconds)
    prop.step_by(120.0)
    print(f"After custom step: {prop.current_epoch}")

    # Trajectory now contains 3 states (initial + 2 steps)
    print(f"Trajectory length: {len(prop.trajectory)}")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    // Create propagator
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Take one step (60 seconds)
    prop.step();
    println!("After 1 step: {}", prop.current_epoch());

    // Step by custom duration (120 seconds)
    prop.step_by(120.0);
    println!("After custom step: {}", prop.current_epoch());

    // Trajectory now contains 3 states (initial + 2 steps)
    println!("Trajectory length: {}", prop.trajectory.len());
    ```

### Multiple Steps

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Take 10 steps (10 × 60 = 600 seconds)
    prop.propagate_steps(10)
    print(f"After 10 steps: {(prop.current_epoch - epoch):.1f} seconds elapsed")
    print(f"Trajectory length: {len(prop.trajectory)}")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Take 10 steps (10 × 60 = 600 seconds)
    prop.propagate_steps(10);
    println!("After 10 steps: {:.1} seconds elapsed",
             prop.current_epoch() - epoch);
    println!("Trajectory length: {}", prop.trajectory.len());
    ```

### Propagate to Target Epoch

For precise time targeting, use `propagate_to()` which adjusts the final step size to exactly reach the target epoch.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Propagate exactly 500 seconds (not evenly divisible by step size)
    target = epoch + 500.0
    prop.propagate_to(target)

    print(f"Target epoch: {target}")
    print(f"Current epoch: {prop.current_epoch}")
    print(f"Difference: {abs(prop.current_epoch - target):.10f} seconds")
    # Output shows machine precision agreement
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Propagate exactly 500 seconds (not evenly divisible by step size)
    let target = epoch + 500.0;
    prop.propagate_to(target);

    println!("Target epoch: {}", target);
    println!("Current epoch: {}", prop.current_epoch());
    println!("Difference: {:.10} seconds",
             (prop.current_epoch() - target).abs());
    // Output shows machine precision agreement
    ```

## Direct State Queries

The `StateProvider` trait allows computing states at arbitrary epochs without building a trajectory. This is useful for sparse sampling or parallel batch computation.

### Single Epoch Queries

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    bh.initialize_eop()  # Required for frame transformations

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Query state 1 hour later (doesn't add to trajectory)
    query_epoch = epoch + 3600.0
    state_native = prop.state(query_epoch)           # Native format (Keplerian)
    state_eci = prop.state_eci(query_epoch)          # ECI Cartesian
    state_ecef = prop.state_ecef(query_epoch)        # ECEF Cartesian
    state_kep = prop.state_as_osculating_elements(
        query_epoch, bh.AngleFormat.DEGREES
    )

    print(f"Native state (Keplerian): a={state_native[0]/1e3:.1f} km")
    print(f"ECI position magnitude: {np.linalg.norm(state_eci[:3])/1e3:.1f} km")
    print(f"ECEF position magnitude: {np.linalg.norm(state_ecef[:3])/1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    initialize_eop().unwrap();  // Required for frame transformations

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Query state 1 hour later (doesn't add to trajectory)
    let query_epoch = epoch + 3600.0;
    let state_native = prop.state(query_epoch);       // Native format (Keplerian)
    let state_eci = prop.state_eci(query_epoch);      // ECI Cartesian
    let state_ecef = prop.state_ecef(query_epoch);    // ECEF Cartesian
    let state_kep = prop.state_as_osculating_elements(
        query_epoch, AngleFormat::Degrees
    );

    println!("Native state (Keplerian): a={:.1} km", state_native[0] / 1e3);
    println!("ECI position magnitude: {:.1} km",
             state_eci.fixed_rows::<3>(0).norm() / 1e3);
    println!("ECEF position magnitude: {:.1} km",
             state_ecef.fixed_rows::<3>(0).norm() / 1e3);
    ```

### Batch Queries

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Generate states at irregular intervals
    query_epochs = [epoch + t for t in [0.0, 100.0, 500.0, 1000.0, 3600.0]]
    states_eci = prop.states_eci(query_epochs)

    print(f"Generated {len(states_eci)} states")
    for i, state in enumerate(states_eci):
        print(f"  Epoch {i}: position magnitude = {np.linalg.norm(state[:3])/1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Generate states at irregular intervals
    let query_epochs = vec![
        epoch, epoch + 100.0, epoch + 500.0, epoch + 1000.0, epoch + 3600.0
    ];
    let states_eci = prop.states_eci(&query_epochs);

    println!("Generated {} states", states_eci.len());
    for (i, state) in states_eci.iter().enumerate() {
        println!("  Epoch {}: position magnitude = {:.1} km",
                 i, state.fixed_rows::<3>(0).norm() / 1e3);
    }
    ```

## Trajectory Management

The propagator stores all stepped states in an internal `OrbitTrajectory`. This trajectory can be accessed, converted, and managed.

### Accessing the Trajectory

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Propagate for several steps
    prop.propagate_steps(5)

    # Access trajectory
    traj = prop.trajectory
    print(f"Trajectory contains {len(traj)} states")

    # Iterate over epoch-state pairs
    for epoch, state in traj:
        print(f"Epoch: {epoch}, semi-major axis: {state[0]/1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Propagate for several steps
    prop.propagate_steps(5);

    // Access trajectory
    let traj = &prop.trajectory;
    println!("Trajectory contains {} states", traj.len());

    // Access by index
    for i in 0..traj.len() {
        let epoch = traj.epoch_at_idx(i).unwrap();
        let state = traj.state_at_idx(i).unwrap();
        println!("Epoch: {}, semi-major axis: {:.1} km", epoch, state[0] / 1e3);
    }
    ```

### Frame Conversions

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    bh.initialize_eop()  # Required for ECEF conversions

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )
    prop.propagate_steps(10)

    # Convert entire trajectory to different frames
    traj_eci = prop.trajectory.to_eci()      # ECI Cartesian
    traj_ecef = prop.trajectory.to_ecef()    # ECEF Cartesian
    traj_kep = prop.trajectory.to_keplerian(bh.AngleFormat.RADIANS)

    print(f"ECI trajectory: {len(traj_eci)} states")
    print(f"ECEF trajectory: {len(traj_ecef)} states")
    print(f"Keplerian trajectory: {len(traj_kep)} states")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    initialize_eop().unwrap();  // Required for ECEF conversions

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );
    prop.propagate_steps(10);

    // Convert entire trajectory to different frames
    let traj_eci = prop.trajectory.to_eci();       // ECI Cartesian
    let traj_ecef = prop.trajectory.to_ecef();     // ECEF Cartesian
    let traj_kep = prop.trajectory.to_keplerian(AngleFormat::Radians);

    println!("ECI trajectory: {} states", traj_eci.len());
    println!("ECEF trajectory: {} states", traj_ecef.len());
    println!("Keplerian trajectory: {} states", traj_kep.len());
    ```

### Memory Management

For long-running applications, control trajectory memory usage with eviction policies.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Keep only 100 most recent states
    prop.set_eviction_policy_max_size(100)

    # Propagate many steps
    prop.propagate_steps(500)
    print(f"Trajectory length: {len(prop.trajectory)}")  # Will be 100

    # Alternative: Keep only states within 1 hour of current time
    prop.reset()
    prop.set_eviction_policy_max_age(3600.0)  # 3600 seconds = 1 hour
    prop.propagate_steps(500)
    print(f"Trajectory length after age policy: {len(prop.trajectory)}")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Keep only 100 most recent states
    prop.set_eviction_policy_max_size(100).unwrap();

    // Propagate many steps
    prop.propagate_steps(500);
    println!("Trajectory length: {}", prop.trajectory.len());  // Will be 100

    // Alternative: Keep only states within 1 hour of current time
    prop.reset();
    prop.set_eviction_policy_max_age(3600.0).unwrap();  // 3600 seconds = 1 hour
    prop.propagate_steps(500);
    println!("Trajectory length after age policy: {}", prop.trajectory.len());
    ```

## Configuration and Control

### Resetting the Propagator

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    # Propagate forward
    prop.propagate_steps(100)
    print(f"After propagation: {len(prop.trajectory)} states")

    # Reset to initial conditions
    prop.reset()
    print(f"After reset: {len(prop.trajectory)} states")
    print(f"Current epoch: {prop.current_epoch}")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    // Propagate forward
    prop.propagate_steps(100);
    println!("After propagation: {} states", prop.trajectory.len());

    // Reset to initial conditions
    prop.reset();
    println!("After reset: {} states", prop.trajectory.len());
    println!("Current epoch: {}", prop.current_epoch());
    ```

### Changing Step Size

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    )

    print(f"Initial step size: {prop.step_size} seconds")

    # Change step size
    prop.set_step_size(120.0)
    print(f"New step size: {prop.step_size} seconds")

    # Subsequent steps use new step size
    prop.step()  # Advances 120 seconds
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    );

    println!("Initial step size: {} seconds", prop.step_size());

    // Change step size
    prop.set_step_size(120.0);
    println!("New step size: {} seconds", prop.step_size());

    // Subsequent steps use new step size
    prop.step();  // Advances 120 seconds
    ```

## Identity Tracking

Track propagators with names, IDs, or UUIDs for multi-satellite scenarios.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])

    # Create propagator with identity (builder pattern)
    prop = bh.KeplerianPropagator.from_keplerian(
        epoch, elements, bh.AngleFormat.DEGREES, 60.0
    ).with_name("Satellite-A").with_id(12345)

    print(f"Name: {prop.get_name()}")
    print(f"ID: {prop.get_id()}")
    ```

=== "Rust"

    ```rust
    use brahe::*;
    use nalgebra::Vector6;

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let elements = Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);

    // Create propagator with identity (builder pattern)
    let prop = KeplerianPropagator::from_keplerian(
        epoch, elements, AngleFormat::Degrees, 60.0
    ).with_name("Satellite-A").with_id(12345);

    println!("Name: {:?}", prop.get_name());
    println!("ID: {:?}", prop.get_id());
    ```

## See Also

- [Orbit Propagation Overview](index.md) - Propagation concepts and trait hierarchy
- [SGP Propagation](sgp_propagation.md) - TLE-based SGP4/SDP4 propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and operations
- [KeplerianPropagator API Reference](../../library_api/propagators/keplerian_propagator.md)
