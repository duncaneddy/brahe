# OrbitTrajectory

`OrbitTrajectory` is a specialized trajectory container for orbital mechanics that tracks reference frames (ECI/ECEF) and orbital representations (Cartesian/Keplerian). Unlike `DTrajectory` and `STrajectory6` which store frame-agnostic data, `OrbitTrajectory` understands orbital mechanics and enables automatic conversions between reference frames and representations.

Use `OrbitTrajectory` when:

- Working with orbital mechanics applications
- Need to convert between ECI and ECEF frames
- Need to convert between Cartesian and Keplerian representations
- Want frame/representation metadata tracked automatically
- Working with propagators that output orbital trajectories

`OrbitTrajectory` implements the `OrbitalTrajectory` trait in addition to `Trajectory` and `Interpolatable`, providing orbital-specific functionality on top of the standard trajectory interface.

## Initialization

### Empty Trajectory - Cartesian

Create an empty trajectory in Cartesian representation:

=== "Python"

    ```python
    import brahe as bh

    # Create trajectory in ECI frame, Cartesian representation
    traj_eci = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None  # No angle format for Cartesian
    )
    print(f"Frame: {traj_eci.frame}")  # Output: OrbitFrame.ECI
    print(f"Representation: {traj_eci.representation}")  # Output: OrbitRepresentation.CARTESIAN

    # Create trajectory in ECEF frame, Cartesian representation
    traj_ecef = bh.OrbitTrajectory(
        bh.OrbitFrame.ECEF,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )
    print(f"Frame: {traj_ecef.frame}")  # Output: OrbitFrame.ECEF
    ```

=== "Rust"

    ```rust
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation};

    // Create trajectory in ECI frame, Cartesian representation
    let traj_eci = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );
    println!("Frame: {:?}", traj_eci.frame);
    println!("Representation: {:?}", traj_eci.representation);

    // Create trajectory in ECEF frame, Cartesian representation
    let traj_ecef = OrbitTrajectory::new(
        OrbitFrame::ECEF,
        OrbitRepresentation::Cartesian,
        None
    );
    ```

### Empty Trajectory - Keplerian

Create an empty trajectory in Keplerian representation:

=== "Python"

    ```python
    import brahe as bh

    # Create trajectory in ECI frame, Keplerian representation with radians
    traj_kep_rad = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.KEPLERIAN,
        bh.AngleFormat.RADIANS  # Required for Keplerian
    )
    print(f"Angle format: {traj_kep_rad.angle_format}")  # Output: AngleFormat.RADIANS

    # Create trajectory in ECI frame, Keplerian representation with degrees
    traj_kep_deg = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.KEPLERIAN,
        bh.AngleFormat.DEGREES
    )
    ```

=== "Rust"

    ```rust
    use brahe::trajectories::OrbitTrajectory;
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation};
    use brahe::AngleFormat;

    // Create trajectory in ECI frame, Keplerian representation with radians
    let traj_kep_rad = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Radians)
    );

    // Create trajectory in ECI frame, Keplerian representation with degrees
    let traj_kep_deg = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Degrees)
    );
    ```

### From Existing Data

Create a trajectory from existing epochs and orbital states:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create epochs
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch1 = epoch0 + 60.0
    epoch2 = epoch0 + 120.0

    # Create Cartesian states in ECI
    state0 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    state1 = np.array([bh.R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0])
    state2 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0])

    # Create trajectory from data
    epochs = [epoch0, epoch1, epoch2]
    states = np.concatenate([state0, state1, state2])  # Flattened array
    traj = bh.OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    print(f"Trajectory length: {len(traj)}")  # Output: 3
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create epochs
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;
    let epoch2 = epoch0 + 120.0;

    // Create Cartesian states
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
    let traj = OrbitTrajectory::from_orbital_data(
        epochs,
        states,
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    println!("Trajectory length: {}", traj.len());  // Output: 3
    ```

### From Propagator

The most common way to create an `OrbitTrajectory` is through orbit propagation:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Define orbital elements for a 500 km circular orbit
    a = bh.R_EARTH + 500e3
    e = 0.001
    i = 97.8  # Sun-synchronous
    raan = 15.0
    argp = 30.0
    M = 0.0
    oe = np.array([a, e, i, raan, argp, M])

    # Create epoch and propagator
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator = bh.KeplerianPropagator.from_keplerian(
        epoch, oe, bh.AngleFormat.DEGREES, 60.0
    )

    # Propagate for several steps
    propagator.propagate_steps(10)

    # Access the trajectory
    traj = propagator.trajectory
    print(f"Trajectory length: {len(traj)}")  # Output: 11 (initial + 10 steps)
    print(f"Frame: {traj.frame}")  # Output: OrbitFrame.ECI
    print(f"Representation: {traj.representation}")  # Output: OrbitRepresentation.CARTESIAN
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::Trajectory;
    use brahe::orbits::propagators::{KeplerianPropagator, Propagator};
    use brahe::constants::R_EARTH;
    use brahe::AngleFormat;
    use nalgebra as na;

    // Define orbital elements
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 97.8_f64.to_radians(),
        15.0_f64.to_radians(), 30.0_f64.to_radians(), 0.0
    );

    // Create epoch and propagator
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch, &oe, AngleFormat::Radians, 60.0
    );

    // Propagate for several steps
    propagator.propagate_steps(10);

    // Access the trajectory
    let traj = &propagator.trajectory;
    println!("Trajectory length: {}", traj.len());  // Output: 11
    ```

## Frame Conversions

The key feature of `OrbitTrajectory` is automatic frame conversion between ECI and ECEF. All conversions account for Earth's rotation using EOP data.

### Converting ECI to ECEF

Convert a trajectory from Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF):

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory in ECI frame
    traj_eci = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add states in ECI
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    for i in range(5):
        epoch = epoch0 + i * 60.0
        # Define state at epoch
        state_eci = np.array([bh.R_EARTH + 500e3, i * 100e3, 0.0, 0.0, 7600.0, 0.0])
        traj_eci.add(epoch, state_eci)

    print(f"Original frame: {traj_eci.frame}")  # Output: OrbitFrame.ECI
    print(f"Original representation: {traj_eci.representation}")  # Output: OrbitRepresentation.CARTESIAN

    # Convert to ECEF
    traj_ecef = traj_eci.to_ecef()

    print(f"\nConverted frame: {traj_ecef.frame}")  # Output: OrbitFrame.ECEF
    print(f"Converted representation: {traj_ecef.representation}")  # Output: OrbitRepresentation.CARTESIAN
    print(f"Same number of states: {len(traj_ecef)}")  # Output: 5

    # Compare first states
    _, state_eci_0 = traj_eci.first()
    _, state_ecef_0 = traj_ecef.first()
    print(f"\nFirst ECI state position: {state_eci_0[0]:.2f} m")
    print(f"First ECEF state position: {state_ecef_0[0]:.2f} m")
    # Values differ due to frame rotation
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory in ECI frame
    let mut traj_eci = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states in ECI
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state_eci = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, (i as f64) * 100e3, 0.0, 0.0, 7600.0, 0.0
        );
        traj_eci.add(epoch, state_eci);
    }

    println!("Original frame: {:?}", traj_eci.frame);
    println!("Original representation: {:?}", traj_eci.representation);

    // Convert to ECEF
    let traj_ecef = traj_eci.to_ecef();

    println!("\nConverted frame: {:?}", traj_ecef.frame);
    println!("Converted representation: {:?}", traj_ecef.representation);
    println!("Same number of states: {}", traj_ecef.len());
    ```

### Converting ECEF to ECI

Convert from ECEF back to ECI:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory in ECEF frame
    traj_ecef = bh.OrbitTrajectory(
        bh.OrbitFrame.ECEF,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add states in ECEF
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    for i in range(3):
        epoch = epoch0 + i * 60.0
        # Define state at epoch
        state_ecef = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 7600.0])
        traj_ecef.add(epoch, state_ecef)

    print(f"Original frame: {traj_ecef.frame}")  # Output: OrbitFrame.ECEF

    # Convert to ECI
    traj_eci = traj_ecef.to_eci()

    print(f"Converted frame: {traj_eci.frame}")  # Output: OrbitFrame.ECI
    print(f"Trajectory length: {len(traj_eci)}")  # Output: 3

    # Iterate over converted states
    for epoch, state_eci in traj_eci:
        pos_mag = np.linalg.norm(state_eci[0:3])
        vel_mag = np.linalg.norm(state_eci[3:6])
        print(f"Epoch: {epoch}")
        print(f"  Position magnitude: {pos_mag / 1e3:.2f} km")
        print(f"  Velocity magnitude: {vel_mag:.2f} m/s")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory in ECEF frame
    let mut traj_ecef = OrbitTrajectory::new(
        OrbitFrame::ECEF,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states in ECEF
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state_ecef = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 7600.0
        );
        traj_ecef.add(epoch, state_ecef);
    }

    println!("Original frame: {:?}", traj_ecef.frame);

    // Convert to ECI
    let traj_eci = traj_ecef.to_eci();

    println!("Converted frame: {:?}", traj_eci.frame);
    println!("Trajectory length: {}", traj_eci.len());

    // Iterate over converted states
    for (epoch, state_eci) in &traj_eci {
        let pos_mag = state_eci.fixed_rows::<3>(0).norm();
        let vel_mag = state_eci.fixed_rows::<3>(3).norm();
        println!("Epoch: {}", epoch);
        println!("  Position magnitude: {:.2} km", pos_mag / 1e3);
        println!("  Velocity magnitude: {:.2} m/s", vel_mag);
    }
    ```

### Round-Trip Frame Conversion

Convert from ECI to ECEF and back to verify consistency:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory in ECI
    traj_eci_original = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add a state
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state_original = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    traj_eci_original.add(epoch, state_original)

    # Convert to ECEF and back to ECI
    traj_ecef = traj_eci_original.to_ecef()
    traj_eci_roundtrip = traj_ecef.to_eci()

    # Compare original and round-trip states
    _, state_roundtrip = traj_eci_roundtrip.first()
    diff = np.abs(state_original - state_roundtrip)

    print(f"Position difference: {np.linalg.norm(diff[0:3]):.6e} m")
    print(f"Velocity difference: {np.linalg.norm(diff[3:6]):.6e} m/s")
    # Expected: Very small differences (numerical precision)
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use nalgebra as na;

    // Create trajectory in ECI
    let mut traj_eci_original = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add a state
    let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let state_original = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    traj_eci_original.add(epoch, state_original);

    // Convert to ECEF and back to ECI
    let traj_ecef = traj_eci_original.to_ecef();
    let traj_eci_roundtrip = traj_ecef.to_eci();

    // Compare original and round-trip states
    let (_, state_roundtrip) = traj_eci_roundtrip.first();
    let diff = state_original - state_roundtrip;

    println!("Position difference: {:.6e} m",
        diff.fixed_rows::<3>(0).norm());
    println!("Velocity difference: {:.6e} m/s",
        diff.fixed_rows::<3>(3).norm());
    // Expected: Very small differences (numerical precision)
    ```

## Representation Conversions

### Converting Cartesian to Keplerian

Convert from Cartesian position/velocity to Keplerian orbital elements:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory in ECI Cartesian
    traj_cart = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add Cartesian states
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    for i in range(3):
        epoch = epoch0 + i * 300.0  # 5-minute intervals
        # Use orbital elements to create realistic Cartesian states
        oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, i * 0.1])
        state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
        traj_cart.add(epoch, state_cart)

    print(f"Original representation: {traj_cart.representation}")
    # Output: OrbitRepresentation.CARTESIAN

    # Convert to Keplerian with degrees
    traj_kep = traj_cart.to_keplerian(bh.AngleFormat.DEGREES)

    print(f"Converted representation: {traj_kep.representation}")
    # Output: OrbitRepresentation.KEPLERIAN
    print(f"Angle format: {traj_kep.angle_format}")
    # Output: AngleFormat.DEGREES

    # Examine Keplerian elements
    for epoch, oe in traj_kep:
        print(f"\nEpoch: {epoch}")
        print(f"  Semi-major axis: {oe[0] / 1e3:.2f} km")
        print(f"  Eccentricity: {oe[1]:.6f}")
        print(f"  Inclination: {oe[2]:.2f}°")
        print(f"  RAAN: {oe[3]:.2f}°")
        print(f"  Arg of perigee: {oe[4]:.2f}°")
        print(f"  True anomaly: {oe[5]:.2f}°")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use brahe::orbits::keplerian::state_osculating_to_cartesian;
    use brahe::AngleFormat;
    use nalgebra as na;

    // Create trajectory in ECI Cartesian
    let mut traj_cart = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add Cartesian states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 300.0;
        let oe = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, (i as f64) * 0.1
        );
        let state_cart = state_osculating_to_cartesian(&oe, AngleFormat::Radians);
        traj_cart.add(epoch, state_cart);
    }

    println!("Original representation: {:?}", traj_cart.representation);

    // Convert to Keplerian with radians
    let traj_kep = traj_cart.to_keplerian(AngleFormat::Radians);

    println!("Converted representation: {:?}", traj_kep.representation);
    println!("Angle format: {:?}", traj_kep.angle_format);

    // Examine Keplerian elements
    for (epoch, oe) in &traj_kep {
        println!("\nEpoch: {}", epoch);
        println!("  Semi-major axis: {:.2} km", oe[0] / 1e3);
        println!("  Eccentricity: {:.6}", oe[1]);
        println!("  Inclination: {:.2} rad", oe[2]);
    }
    ```

### Converting with Different Angle Formats

Convert to Keplerian with different angle formats:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory in ECI Cartesian
    traj_cart = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add a state
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0])
    state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    traj_cart.add(epoch, state_cart)

    # Convert to Keplerian with radians
    traj_kep_rad = traj_cart.to_keplerian(bh.AngleFormat.RADIANS)
    _, oe_rad = traj_kep_rad.first()

    # Convert to Keplerian with degrees
    traj_kep_deg = traj_cart.to_keplerian(bh.AngleFormat.DEGREES)
    _, oe_deg = traj_kep_deg.first()

    print("Radians version:")
    print(f"  Inclination: {oe_rad[2]:.6f} rad = {np.degrees(oe_rad[2]):.2f}°")

    print("\nDegrees version:")
    print(f"  Inclination: {oe_deg[2]:.2f}°")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use brahe::orbits::keplerian::state_osculating_to_cartesian;
    use brahe::AngleFormat;
    use nalgebra as na;

    // Create trajectory in ECI Cartesian
    let mut traj_cart = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add a state
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0
    );
    let state_cart = state_osculating_to_cartesian(&oe, AngleFormat::Radians);
    traj_cart.add(epoch, state_cart);

    // Convert to Keplerian with radians
    let traj_kep_rad = traj_cart.to_keplerian(AngleFormat::Radians);
    let (_, oe_rad) = traj_kep_rad.first();

    // Convert to Keplerian with degrees
    let traj_kep_deg = traj_cart.to_keplerian(AngleFormat::Degrees);
    let (_, oe_deg) = traj_kep_deg.first();

    println!("Radians version:");
    println!("  Inclination: {:.6} rad = {:.2}°", oe_rad[2], oe_rad[2].to_degrees());

    println!("\nDegrees version:");
    println!("  Inclination: {:.2}°", oe_deg[2]);
    ```

## Combined Frame and Representation Conversions

You can chain conversions to transform both frame and representation:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Start with ECI Cartesian trajectory
    traj_eci_cart = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add states
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0])
    state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    traj_eci_cart.add(epoch, state_cart)

    print("Original:")
    print(f"  Frame: {traj_eci_cart.frame}")
    print(f"  Representation: {traj_eci_cart.representation}")

    # Convert to ECEF frame (stays Cartesian)
    traj_ecef_cart = traj_eci_cart.to_ecef()
    print("\nAfter to_ecef():")
    print(f"  Frame: {traj_ecef_cart.frame}")
    print(f"  Representation: {traj_ecef_cart.representation}")

    # Convert back to ECI
    traj_eci_cart2 = traj_ecef_cart.to_eci()
    print("\nAfter to_eci():")
    print(f"  Frame: {traj_eci_cart2.frame}")
    print(f"  Representation: {traj_eci_cart2.representation}")

    # Convert to Keplerian (in ECI frame)
    traj_eci_kep = traj_eci_cart2.to_keplerian(bh.AngleFormat.DEGREES)
    print("\nAfter to_keplerian():")
    print(f"  Frame: {traj_eci_kep.frame}")
    print(f"  Representation: {traj_eci_kep.representation}")
    print(f"  Angle format: {traj_eci_kep.angle_format}")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
    use brahe::constants::R_EARTH;
    use brahe::orbits::keplerian::state_osculating_to_cartesian;
    use brahe::AngleFormat;
    use nalgebra as na;

    // Start with ECI Cartesian trajectory
    let mut traj_eci_cart = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0
    );
    let state_cart = state_osculating_to_cartesian(&oe, AngleFormat::Radians);
    traj_eci_cart.add(epoch, state_cart);

    println!("Original:");
    println!("  Frame: {:?}", traj_eci_cart.frame);
    println!("  Representation: {:?}", traj_eci_cart.representation);

    // Convert to ECEF frame (stays Cartesian)
    let traj_ecef_cart = traj_eci_cart.to_ecef();
    println!("\nAfter to_ecef():");
    println!("  Frame: {:?}", traj_ecef_cart.frame);
    println!("  Representation: {:?}", traj_ecef_cart.representation);

    // Convert back to ECI
    let traj_eci_cart2 = traj_ecef_cart.to_eci();
    println!("\nAfter to_eci():");
    println!("  Frame: {:?}", traj_eci_cart2.frame);
    println!("  Representation: {:?}", traj_eci_cart2.representation);

    // Convert to Keplerian (in ECI frame)
    let traj_eci_kep = traj_eci_cart2.to_keplerian(AngleFormat::Radians);
    println!("\nAfter to_keplerian():");
    println!("  Frame: {:?}", traj_eci_kep.frame);
    println!("  Representation: {:?}", traj_eci_kep.representation);
    ```

## Standard Trajectory Operations

`OrbitTrajectory` supports all standard trajectory operations since it implements the `Trajectory` and `Interpolatable` traits:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # Create trajectory
    traj = bh.OrbitTrajectory(
        bh.OrbitFrame.ECI,
        bh.OrbitRepresentation.CARTESIAN,
        None
    )

    # Add states
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    for i in range(10):
        epoch = epoch0 + i * 60.0
        oe = np.array([bh.R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, i * 0.1])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
        traj.add(epoch, state)

    # Query properties
    print(f"Length: {len(traj)}")
    print(f"Timespan: {traj.timespan():.1f} seconds")
    print(f"Start epoch: {traj.start_epoch()}")
    print(f"End epoch: {traj.end_epoch()}")

    # Interpolate at intermediate time
    interp_epoch = epoch0 + 45.0
    interp_state = traj.interpolate(interp_epoch)
    print(f"\nInterpolated altitude: {(interp_state[0] - bh.R_EARTH) / 1e3:.2f} km")

    # Iterate over states
    for i, (epoch, state) in enumerate(traj):
        if i < 2:  # Just show first two
            print(f"State {i}: Epoch={epoch}, Position magnitude={np.linalg.norm(state[0:3]) / 1e3:.2f} km")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{OrbitTrajectory, Trajectory, Interpolatable};
    use brahe::trajectories::traits::{OrbitFrame, OrbitRepresentation};
    use brahe::constants::R_EARTH;
    use brahe::orbits::keplerian::state_osculating_to_cartesian;
    use brahe::AngleFormat;
    use nalgebra as na;

    // Create trajectory
    let mut traj = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    for i in 0..10 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let oe = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, (i as f64) * 0.1
        );
        let state = state_osculating_to_cartesian(&oe, AngleFormat::Radians);
        traj.add(epoch, state);
    }

    // Query properties
    println!("Length: {}", traj.len());
    println!("Timespan: {:.1} seconds", traj.timespan());
    println!("Start epoch: {}", traj.start_epoch());
    println!("End epoch: {}", traj.end_epoch());

    // Interpolate at intermediate time
    let interp_epoch = epoch0 + 45.0;
    let interp_state = traj.interpolate(interp_epoch);
    println!("\nInterpolated altitude: {:.2} km",
        (interp_state[0] - R_EARTH) / 1e3);

    // Iterate over states
    for (i, (epoch, state)) in traj.into_iter().enumerate().take(2) {
        let pos_mag = state.fixed_rows::<3>(0).norm();
        println!("State {}: Epoch={}, Position magnitude={:.2} km",
            i, epoch, pos_mag / 1e3);
    }
    ```

## Practical Workflow Example

A complete example showing propagation, frame conversion, and analysis:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    # 1. Define orbit and create propagator
    a = bh.R_EARTH + 500e3  # 500 km altitude
    e = 0.001  # Nearly circular
    i = 97.8  # Sun-synchronous
    raan = 15.0
    argp = 30.0
    M = 0.0
    oe = np.array([a, e, i, raan, argp, M])

    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator = bh.KeplerianPropagator.from_keplerian(
        epoch, oe, bh.AngleFormat.DEGREES, 60.0
    )

    # 2. Propagate for one orbit period
    period = bh.orbital_period(a)
    end_epoch = epoch + period
    propagator.propagate_to(end_epoch)

    # 3. Get trajectory in ECI Cartesian
    traj_eci = propagator.trajectory
    print(f"Propagated {len(traj_eci)} states over {traj_eci.timespan() / 60:.1f} minutes")

    # 4. Convert to ECEF to analyze ground track
    traj_ecef = traj_eci.to_ecef()
    print(f"\nGround track in ECEF frame:")
    for i, (epoch, state_ecef) in enumerate(traj_ecef):
        if i % 10 == 0:  # Sample every 10 states
            # Convert ECEF to geodetic for latitude/longitude
            lat, lon, alt = bh.position_ecef_to_geodetic(state_ecef[0:3])
            print(f"  {epoch}: Lat={np.degrees(lat):6.2f}°, Lon={np.degrees(lon):7.2f}°, Alt={alt/1e3:6.2f} km")

    # 5. Convert to Keplerian to analyze orbital evolution
    traj_kep = traj_eci.to_keplerian(bh.AngleFormat.DEGREES)
    first_oe = traj_kep.state_at_idx(0)
    last_oe = traj_kep.state_at_idx(len(traj_kep) - 1)

    print(f"\nOrbital element evolution:")
    print(f"  Semi-major axis: {first_oe[0]/1e3:.2f} km → {last_oe[0]/1e3:.2f} km")
    print(f"  Eccentricity: {first_oe[1]:.6f} → {last_oe[1]:.6f}")
    print(f"  Inclination: {first_oe[2]:.2f}° → {last_oe[2]:.2f}°")
    print(f"  True anomaly: {first_oe[5]:.2f}° → {last_oe[5]:.2f}°")
    ```

=== "Rust"

    ```rust
    use brahe::time::Epoch;
    use brahe::trajectories::{Trajectory, Interpolatable};
    use brahe::trajectories::traits::OrbitalTrajectory;
    use brahe::orbits::propagators::{KeplerianPropagator, Propagator};
    use brahe::orbits::keplerian::orbital_period;
    use brahe::coordinates::geodetic::position_ecef_to_geodetic;
    use brahe::constants::R_EARTH;
    use brahe::AngleFormat;
    use nalgebra as na;

    // 1. Define orbit and create propagator
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 97.8_f64.to_radians(),
        15.0_f64.to_radians(), 30.0_f64.to_radians(), 0.0
    );

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC);
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch, &oe, AngleFormat::Radians, 60.0
    );

    // 2. Propagate for one orbit period
    let period = orbital_period(R_EARTH + 500e3);
    let end_epoch = epoch + period;
    propagator.propagate_to(end_epoch);

    // 3. Get trajectory in ECI Cartesian
    let traj_eci = &propagator.trajectory;
    println!("Propagated {} states over {:.1} minutes",
        traj_eci.len(), traj_eci.timespan() / 60.0);

    // 4. Convert to ECEF
    let traj_ecef = traj_eci.to_ecef();
    println!("\nGround track in ECEF frame:");
    for (i, (epoch, state_ecef)) in traj_ecef.into_iter().enumerate() {
        if i % 10 == 0 {
            let pos_ecef = state_ecef.fixed_rows::<3>(0);
            let (lat, lon, alt) = position_ecef_to_geodetic(&pos_ecef.into());
            println!("  {}: Lat={:6.2}°, Lon={:7.2}°, Alt={:6.2} km",
                epoch, lat.to_degrees(), lon.to_degrees(), alt / 1e3);
        }
    }

    // 5. Convert to Keplerian
    let traj_kep = traj_eci.to_keplerian(AngleFormat::Radians);
    let first_oe = traj_kep.state_at_idx(0);
    let last_oe = traj_kep.state_at_idx(traj_kep.len() - 1);

    println!("\nOrbital element evolution:");
    println!("  Semi-major axis: {:.2} km → {:.2} km",
        first_oe[0] / 1e3, last_oe[0] / 1e3);
    println!("  Eccentricity: {:.6} → {:.6}",
        first_oe[1], last_oe[1]);
    println!("  Inclination: {:.2}° → {:.2}°",
        first_oe[2].to_degrees(), last_oe[2].to_degrees());
    ```

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory
- [STrajectory6](strajectory6.md) - Static 6D trajectory
- [OrbitTrajectory API Reference](../../library_api/trajectories/orbit_trajectory.md)
