# SGP Propagation

The `SGPPropagator` implements the SGP4/SDP4 propagation models for orbital prediction from Two-Line Element (TLE) data. SGP4 is the standard method for satellite tracking and includes simplified perturbations from Earth oblateness and atmospheric drag, making it suitable for operational satellite tracking and near-Earth orbit propagation.

For complete API documentation, see the [SGPPropagator API Reference](../../library_api/propagators/sgp_propagator.md).

## TLE Format Support

SGP4 propagation is based on Two-Line Element (TLE) sets, a compact data format for orbital elements. Brahe supports both traditional and modern TLE formats:

- **Classic Format**: Traditional numeric NORAD catalog numbers (5 digits, up to 99999)
- **Alpha-5 Format**: Extended alphanumeric catalog numbers for satellites beyond ID 99999

The propagator automatically detects and handles both formats.

## Initialization

The `SGPPropagator` is initialized from TLE data. The TLE lines contain all orbital parameters needed for propagation.

### From 2-Line TLE

The most common initialization uses two lines of TLE data.

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()  # Required for accurate frame transformations

    # ISS TLE data (example)
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Create propagator with 60-second step size
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    print(f"NORAD ID: {prop.norad_id}")
    print(f"TLE epoch: {prop.epoch}")
    print(f"Initial position magnitude: {bh.norm(prop.initial_state[:3]) / 1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();  // Required for accurate frame transformations

    // ISS TLE data (example)
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator with 60-second step size
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    println!("NORAD ID: {}", prop.norad_id);
    println!("TLE epoch: {}", prop.epoch);
    println!("Initial position magnitude: {:.1} km",
             prop.initial_state.fixed_rows::<3>(0).norm() / 1e3);
    ```

### From 3-Line TLE

Three-line TLE format includes an optional satellite name on the first line.

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    # 3-line TLE with satellite name
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Create propagator with satellite name
    prop = bh.SGPPropagator.from_3le(name, line1, line2, 60.0)

    print(f"Satellite name: {prop.satellite_name}")
    print(f"NORAD ID: {prop.norad_id}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    // 3-line TLE with satellite name
    let name = "ISS (ZARYA)";
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator with satellite name
    let prop = SGPPropagator::from_3le(Some(name), line1, line2, 60.0).unwrap();

    println!("Satellite name: {:?}", prop.satellite_name);
    println!("NORAD ID: {}", prop.norad_id);
    ```

### Configuring Output Format

By default, SGP4 outputs states in ECI Cartesian coordinates. Use `with_output_format()` to configure the output frame and representation.

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Create with ECEF Cartesian output
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0).with_output_format(
        bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None
    )

    # Or with Keplerian output (ECI only)
    prop_kep = bh.SGPPropagator.from_tle(line1, line2, 60.0).with_output_format(
        bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES
    )

    print(f"ECEF propagator frame: {prop.frame}")
    print(f"Keplerian propagator representation: {prop_kep.representation}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create with ECEF Cartesian output
    let prop = SGPPropagator::from_tle(line1, line2, 60.0)
        .unwrap()
        .with_output_format(
            OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None
        );

    // Or with Keplerian output (ECI only)
    let prop_kep = SGPPropagator::from_tle(line1, line2, 60.0)
        .unwrap()
        .with_output_format(
            OrbitFrame::ECI, OrbitRepresentation::Keplerian, Some(AngleFormat::Degrees)
        );

    println!("ECEF propagator frame: {:?}", prop.frame);
    println!("Keplerian propagator representation: {:?}", prop_kep.representation);
    ```

## Stepping Through Time

The SGP propagator uses the same stepping interface as the Keplerian propagator through the `OrbitPropagator` trait.

### Single and Multiple Steps

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Single step (60 seconds)
    prop.step()
    print(f"After 1 step: {prop.current_epoch}")

    # Multiple steps
    prop.propagate_steps(10)
    print(f"After 11 total steps: {len(prop.trajectory)} states")

    # Step by custom duration
    prop.step_by(120.0)
    print(f"After custom step: {prop.current_epoch}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Single step (60 seconds)
    prop.step();
    println!("After 1 step: {}", prop.current_epoch());

    // Multiple steps
    prop.propagate_steps(10);
    println!("After 11 total steps: {} states", prop.trajectory.len());

    // Step by custom duration
    prop.step_by(120.0);
    println!("After custom step: {}", prop.current_epoch());
    ```

### Propagate to Target Epoch

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Propagate to specific epoch
    target = prop.epoch + 7200.0  # 2 hours later
    prop.propagate_to(target)

    print(f"Target epoch: {target}")
    print(f"Current epoch: {prop.current_epoch}")
    print(f"Trajectory contains {len(prop.trajectory)} states")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Propagate to specific epoch
    let target = prop.epoch + 7200.0;  // 2 hours later
    prop.propagate_to(target);

    println!("Target epoch: {}", target);
    println!("Current epoch: {}", prop.current_epoch());
    println!("Trajectory contains {} states", prop.trajectory.len());
    ```

## Direct State Queries

The SGP propagator implements the `StateProvider` trait, allowing direct state computation at arbitrary epochs without stepping.

### Single Epoch Queries

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Query state 1 orbit later (doesn't add to trajectory)
    query_epoch = prop.epoch + 5400.0  # ~90 minutes

    state_eci = prop.state_eci(query_epoch)         # ECI Cartesian
    state_ecef = prop.state_ecef(query_epoch)       # ECEF Cartesian
    state_kep = prop.state_as_osculating_elements(
        query_epoch, bh.AngleFormat.DEGREES
    )

    print(f"ECI position: [{state_eci[0]/1e3:.1f}, {state_eci[1]/1e3:.1f}, "
          f"{state_eci[2]/1e3:.1f}] km")
    print(f"Osculating semi-major axis: {state_kep[0]/1e3:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Query state 1 orbit later (doesn't add to trajectory)
    let query_epoch = prop.epoch + 5400.0;  // ~90 minutes

    let state_eci = prop.state_eci(query_epoch);          // ECI Cartesian
    let state_ecef = prop.state_ecef(query_epoch);        // ECEF Cartesian
    let state_kep = prop.state_as_osculating_elements(
        query_epoch, AngleFormat::Degrees
    );

    println!("ECI position: [{:.1}, {:.1}, {:.1}] km",
             state_eci[0]/1e3, state_eci[1]/1e3, state_eci[2]/1e3);
    println!("Osculating semi-major axis: {:.1} km", state_kep[0]/1e3);
    ```

### Batch Queries

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Generate states for multiple orbits
    orbital_period = 5400.0  # Approximate ISS period (seconds)
    query_epochs = [prop.epoch + i * orbital_period for i in range(5)]
    states_eci = prop.states_eci(query_epochs)

    print(f"Generated {len(states_eci)} states over {len(query_epochs)} orbits")
    for i, state in enumerate(states_eci):
        altitude = (bh.norm(state[:3]) - bh.R_EARTH) / 1e3
        print(f"  Orbit {i}: altitude = {altitude:.1f} km")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Generate states for multiple orbits
    let orbital_period = 5400.0;  // Approximate ISS period (seconds)
    let query_epochs: Vec<Epoch> = (0..5)
        .map(|i| prop.epoch + i as f64 * orbital_period)
        .collect();
    let states_eci = prop.states_eci(&query_epochs);

    println!("Generated {} states over {} orbits", states_eci.len(), query_epochs.len());
    for (i, state) in states_eci.iter().enumerate() {
        let altitude = (state.fixed_rows::<3>(0).norm() - R_EARTH) / 1e3;
        println!("  Orbit {}: altitude = {:.1} km", i, altitude);
    }
    ```

### Special: PEF Frame

SGP4 natively outputs states in the TEME (True Equator Mean Equinox) frame. For specialized applications, you can access states in the intermediate PEF (Pseudo-Earth-Fixed) frame:

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Get state in PEF frame (TEME rotated by GMST)
    state_pef = prop.state_pef(prop.epoch)
    print(f"PEF position: {state_pef[:3] / 1e3}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Get state in PEF frame (TEME rotated by GMST)
    let state_pef = prop.state_pef(prop.epoch);
    println!("PEF position: {:?}", state_pef.fixed_rows::<3>(0) / 1e3);
    ```

## Extracting Orbital Elements from TLE

The propagator can extract Keplerian orbital elements directly from the TLE data:

=== "Python"

    ```python
    import brahe as bh

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Extract Keplerian elements from TLE
    elements_deg = prop.get_elements(bh.AngleFormat.DEGREES)
    elements_rad = prop.get_elements(bh.AngleFormat.RADIANS)

    print(f"Semi-major axis: {elements_deg[0]/1e3:.1f} km")
    print(f"Eccentricity: {elements_deg[1]:.6f}")
    print(f"Inclination: {elements_deg[2]:.4f} degrees")
    print(f"RAAN: {elements_deg[3]:.4f} degrees")
    print(f"Argument of perigee: {elements_deg[4]:.4f} degrees")
    print(f"Mean anomaly: {elements_deg[5]:.4f} degrees")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Extract Keplerian elements from TLE
    let elements_deg = prop.get_elements(AngleFormat::Degrees).unwrap();
    let elements_rad = prop.get_elements(AngleFormat::Radians).unwrap();

    println!("Semi-major axis: {:.1} km", elements_deg[0]/1e3);
    println!("Eccentricity: {:.6}", elements_deg[1]);
    println!("Inclination: {:.4} degrees", elements_deg[2]);
    println!("RAAN: {:.4} degrees", elements_deg[3]);
    println!("Argument of perigee: {:.4} degrees", elements_deg[4]);
    println!("Mean anomaly: {:.4} degrees", elements_deg[5]);
    ```

## Trajectory Management

SGP propagators support the same trajectory management as Keplerian propagators, including frame conversions and memory management.

### Memory Management

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # Keep only 50 most recent states for memory efficiency
    prop.set_eviction_policy_max_size(50)

    # Propagate many steps
    prop.propagate_steps(200)
    print(f"Trajectory length: {len(prop.trajectory)}")  # Will be 50

    # Alternative: Keep states within 30 minutes of current
    prop.reset()
    prop.set_eviction_policy_max_age(1800.0)  # 1800 seconds = 30 minutes
    prop.propagate_steps(200)
    print(f"Trajectory length with age policy: {len(prop.trajectory)}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Keep only 50 most recent states for memory efficiency
    prop.set_eviction_policy_max_size(50).unwrap();

    // Propagate many steps
    prop.propagate_steps(200);
    println!("Trajectory length: {}", prop.trajectory.len());  // Will be 50

    // Alternative: Keep states within 30 minutes of current
    prop.reset();
    prop.set_eviction_policy_max_age(1800.0).unwrap();  // 1800 seconds = 30 minutes
    prop.propagate_steps(200);
    println!("Trajectory length with age policy: {}", prop.trajectory.len());
    ```

## Limitations and Considerations

### Immutable Initial Conditions

Unlike the Keplerian propagator, SGP4 initial conditions are derived from the TLE and **cannot be changed**. Attempting to call `set_initial_conditions()` will result in a panic:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # This will raise an error - SGP initial conditions come from TLE
    # prop.set_initial_conditions(...)  # Don't do this!

    # To use different orbital elements, create a KeplerianPropagator instead
    ```

=== "Rust"

    ```rust
    // This will panic - SGP initial conditions come from TLE
    // prop.set_initial_conditions(...);  // Don't do this!

    // To use different orbital elements, create a KeplerianPropagator instead
    ```

### Accuracy and Validity

- **Best accuracy**: Near-Earth satellites (altitude < 2000 km)
- **Valid duration**: Days to weeks (accuracy degrades beyond ~1-2 weeks)
- **Not suitable for**:
  - High-precision applications (GPS, GNSS)
  - Long-term propagation (months to years)
  - Highly eccentric orbits (e > 0.5)
  - Deep-space trajectories

For better accuracy or longer propagation periods, consider:
- Updating TLEs frequently (daily or weekly)
- Using numerical integrators with full force models
- Using specialized propagators (GPS ephemeris, lunar trajectories, etc.)

## Identity Tracking

Like Keplerian propagators, SGP propagators support identity tracking:

=== "Python"

    ```python
    import brahe as bh

    bh.initialize_eop()

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Create with identity (builder pattern)
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0) \
        .with_name("ISS") \
        .with_id(25544)

    print(f"Name: {prop.get_name()}")
    print(f"ID: {prop.get_id()}")
    print(f"NORAD ID from TLE: {prop.norad_id}")
    ```

=== "Rust"

    ```rust
    use brahe::*;

    initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create with identity (builder pattern)
    let prop = SGPPropagator::from_tle(line1, line2, 60.0)
        .unwrap()
        .with_name("ISS")
        .with_id(25544);

    println!("Name: {:?}", prop.get_name());
    println!("ID: {:?}", prop.get_id());
    println!("NORAD ID from TLE: {}", prop.norad_id);
    ```

## See Also

- [Orbit Propagation Overview](index.md) - Propagation concepts and trait hierarchy
- [Keplerian Propagation](keplerian_propagation.md) - Analytical two-body propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and operations
- [Two-Line Elements](../orbits/two_line_elements.md) - Working with TLE data
- [SGPPropagator API Reference](../../library_api/propagators/sgp_propagator.md)
