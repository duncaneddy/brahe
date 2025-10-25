<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - Practical Astrodynamics</em>
</p>
<p align="center">
<a href="https://github.com/duncaneddy/brahe/actions/workflows/unit_tests.yml" target="_blank">
    <img src="https://github.com/duncaneddy/brahe/actions/workflows/unit_tests.yml/badge.svg" alt="Tests">
</a>
<a href="https://codecov.io/gh/duncaneddy/brahe">  
  <img src="https://codecov.io/gh/duncaneddy/brahe/graph/badge.svg?token=1JDXP549Q4"></a>
<a href="https://crates.io/crates/brahe" target="_blank">
    <img src="https://img.shields.io/crates/v/brahe.svg" alt="Crate">
</a>
<a href="https://pypi.org/project/brahe" target="_blank">
    <img src="https://img.shields.io/pypi/v/brahe?color=blue" alt="PyPi">
</a>
<a href="https://duncaneddy.github.io/brahe" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs">
</a>
<a href="https://github.com/duncaneddy/brahe/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg", alt="License">
</a>
</p>

----

Documentation: https://duncaneddy.github.io/brahe

Rust Library Reference: https://docs.rs/crate/brahe/latest

Source Code: https://github.com/duncaneddy/brahe

----

# Brahe

Brahe is a modern satellite dynamics library for research and engineering applications. It is designed to be easy-to-learn, quick-to-deploy, and easy to build on. The north-star of the development is enabling users to solve meaningful problems quickly and correctly.

Brahe is permissively licensed under an [MIT License](https://github.com/duncaneddy/brahe/blob/main/LICENSE) to encourage enable people to use and build on the work without worrying about licensing restrictions. We want people to be able to stop reinventing the astrodynamics wheel because commercial licenses are expensive, or open-source options are hard to use or incomplete.

Finally, we also try to make the software library easy to understand and extend. Many astrodynamics libraries are written in a way that makes them hard to read, understand, or modify. Brahe is written in a modern style with an emphasis on code clarity and modularity to make it easier to understand how algorithms are implemented and to make it easier to extend the library to support new use-cases. This also has the added benefit of making it easier to verify and validate the correctness of the implementation.

If you do find this useful, please consider starring the repository on GitHub to help increase its visibility. If you're using Brahe for school, research, a commercial endeavour, or flying a mission. I'd love to know about it!

If you find a bug, have a feature request, want to contribute, please open an issue or a pull request on the GitHub repository. Contributions are welcome and encouraged! If you see something missing, but don't know how to start contributing, please open an issue and we can discuss it. We are building software to help everyone on this planet explore the universe. We encourage you to bring your unique perspective to help make us stronger. We appreciate contributions from everyone, no prior space experience is needed to participate.

We hope you find Brahe useful for your work!

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- **[Learn](https://duncaneddy.github.io/brahe/learn/)**: Provides short-form documentation of major concepts of the package.
- **[Examples](https://duncaneddy.github.io/brahe/examples/)**: Provides longer-form examples of how-to examples of accomplish common tasks.
- **[Python API Reference](https://duncaneddy.github.io/brahe/library_api/)**: Provides detailed reference documentation of the Python API.
- **[Rust API Reference](https://docs.rs/brahe)**: Provides detailed reference documentation of the Rust API.

## License

The project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

We want to make it easy for people to use and build on the work without worrying about licensing restrictions.

<!-- ## Citation / Acknowledgement -->
## Quick Start

### Python

To install the latest release of brahe for Python, simply run:

```bash
pip install brahe
```

You can then import the package in your Python code with:

```python
import brahe as bh
```

And do something fun like calculate the orbital-period of a satellite in low Earth orbit:

``` python
import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.R_EARTH + 400e3  # 400 km altitude

# Calculate the orbital period
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
# Outputs:
# Orbital Period: 92.56 minutes
```

Here are some common operations to get you started:

**Working with Time:**
``` python
import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.R_EARTH + 400e3  # 400 km altitude

# Calculate the orbital period
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
# Outputs:
# Orbital Period: 92.56 minutes
```

**Coordinate Transformations:**
``` python
import brahe as bh
import numpy as np

# Initialize Earth Orientation Parameter data
bh.initialize_eop()

# Define orbital elements
a = bh.constants.R_EARTH + 700e3    # Semi-major axis in meters (700 km altitude)
e = 0.001                           # Eccentricity
i = 98.7                            # Inclination in radians
raan = 15.0                         # Right Ascension of Ascending Node in radians
arg_periapsis = 30.0                # Argument of Periapsis in radians
mean_anomaly = 45.0                 # Mean Anomaly

# Create a state vector from orbital elements
state_kep = np.array([a, e, i, raan, arg_periapsis, mean_anomaly])

# Convert Keplerian state to ECI coordinates
state_eci = bh.state_osculating_to_cartesian(state_kep, bh.AngleFormat.DEGREES)
print(f"ECI Coordinates: {state_eci}")
# Outputs:
# ECI Coordinates: [ 2.02651406e+06 -5.27290081e+05  6.75606709e+06 -6.93198095e+03 -2.16097991e+03  1.91618569e+03]

# Define a time epoch
epoch = bh.Epoch(2024, 6, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)

# Convert ECI coordinates to ECEF coordinates at the given epoch
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)
print(f"ECEF Coordinates: {state_ecef}")
# Outputs:
# ECEF Coordinates: [ 1.86480173e+05 -2.07022599e+06  6.76081548e+06 -4.53886373e+03 5.77702345e+03  1.89973203e+03]

# Convert back from ECEF to ECI coordinates
state_eci_2 = bh.state_ecef_to_eci(epoch, state_ecef)
print(f"Recovered ECI Coordinates: {state_eci_2}")
# Outputs:
# Recovered ECI Coordinates: [ 2.02651406e+06 -5.27290081e+05  6.75606709e+06 -6.93198095e+03 -2.16097991e+03  1.91618569e+03]

# Convert back from ECI to Keplerian elements
state_kep_2 = bh.state_cartesian_to_osculating(state_eci_2, bh.AngleFormat.DEGREES)
print(f"Recovered Keplerian Elements: {state_kep_2}")
# Outputs:
# Recovered Keplerian Elements: [7.0781363e+06 1.0000000e-03 9.8700000e+01 1.5000000e+01 3.0000000e+01 4.5000000e+01]
```

**Propagating an Orbit:**
``` python
import numpy as np
import brahe as bh

# Define the initial Keplerian elements
a = bh.constants.R_EARTH + 700e3  # Semi-major axis: 700 km altitude
e = 0.001                         # Eccentricity
i = 98.7                          # Inclination in degrees
raan = 15.0                       # Right Ascension of Ascending Node in degrees
argp = 30.0                       # Argument of Perigee in degrees
mean_anomaly = 75.0               # Mean Anomaly at epoch in degrees

initial_state = np.array([a, e, i, raan, argp, mean_anomaly])

# Define the epoch time
epoch = bh.Epoch.now()

# Create the Keplerian Orbit Propagator
dt = 60.0  # Time step in seconds
propagator = bh.KeplerianPropagator.from_keplerian(epoch, initial_state, bh.AngleFormat.DEGREES, dt)

# Propagate the orbit for 3 time steps
propagator.propagate_steps(3)

# States are stored as a Trajectory object
assert len(propagator.trajectory) == 4  # Initial state + 3 propagated states

# Convert trajectory to ECI coordinates
eci_trajectory = propagator.trajectory.to_eci()

# Iterate over all stored states 
for epoch, state in eci_trajectory:
    print(f"Epoch: {epoch}, Position (ECI): {state[0]/1e3:.2f} km, {state[1]/1e3:.2f} km, {state[2]/1e3:.2f} km")

# Output:
# Epoch: 2025-10-24 22:14:56.707 UTC, Position (ECI): -1514.38 km, -1475.59 km, 6753.03 km
# Epoch: 2025-10-24 22:15:56.707 UTC, Position (ECI): -1935.70 km, -1568.01 km, 6623.80 km
# Epoch: 2025-10-24 22:16:56.707 UTC, Position (ECI): -2349.19 km, -1654.08 km, 6467.76 km
# Epoch: 2025-10-24 22:17:56.707 UTC, Position (ECI): -2753.17 km, -1733.46 km, 6285.55 km

# Propagate for 7 days
end_epoch = epoch + 86400 * 7  # 7 days later
propagator.propagate_to(end_epoch)

# Confirm the final epoch is as expected
assert abs(propagator.current_epoch - end_epoch) < 1e-6
print("Propagation complete. Final epoch:", propagator.current_epoch)
# Output:
# Propagation complete. Final epoch: 2025-10-31 22:18:40.413 UTC
```

**Computing ISS Access Windows:**
``` python
import brahe as bh

# Initialize EOP
bh.initialize_eop()

# Set the location
location = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")

# Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
propagator = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0)

# Configure Search Window
epoch_start = bh.Epoch.now()
epoch_end = epoch_start + 7 * 86400.0  # 7 days later

# Set access constraints -> Must be above 10 degrees elevation
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute access windows
windows = bh.location_accesses(
    location,
    propagator,
    epoch_start,
    epoch_end,
    constraint
)

assert len(windows) > 0, "Should find at least one access window"

# Print first 3 access windows
for window in windows[:3]:
    print(
        f"Access Window: {window.window_open} to {window.window_close}, Duration: {window.duration/60:.2f} minutes"
    )
# Outputs:
# Access Window: 2025-10-25 08:49:40.062 UTC to 2025-10-25 08:53:48.463 UTC, Duration: 4.14 minutes
# Access Window: 2025-10-25 10:25:40.245 UTC to 2025-10-25 10:31:48.463 UTC, Duration: 6.14 minutes
# Access Window: 2025-10-25 12:05:33.455 UTC to 2025-10-25 12:06:48.463 UTC, Duration: 1.25 minutes
```

----

### Rust

To use brahe in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
brahe = "0.5"
```

```rust
use brahe as bh;
```

And still calculate the orbital-period of a satellite in low Earth orbit:

``` rust
use brahe::{R_EARTH, orbital_period};

fn main() {
    // Define the semi-major axis of a low Earth orbit (in meters)
    let semi_major_axis = R_EARTH + 400e3; // 400 km altitude

    // Calculate the orbital period
    let period = orbital_period(semi_major_axis); 

    println!("Orbital Period: {:.2} minutes", period / 60.0);
    // Outputs:
    // Orbital Period: 92.56 minutes
}
```

You can do everything that you can do in Python in Rust as well:

**Working with Time:**
``` rust
use brahe::{Epoch, TimeSystem};

fn main() {
    // Create an epoch from a specific date and time
    let epc = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Print as ISO 8601 string
    println!("Epoch in UTC: {}", epc.isostring());
    // Output:
    // Epoch in UTC: 2024-01-01T12:00:00Z

    // Get the Modified Julian Date (MJD) in different time systems
    let mjd_tai = epc.mjd_as_time_system(TimeSystem::TAI);
    println!("MJD in TAI: {}", mjd_tai);
    // Output:
    // MJD in TAI: 60310.50042824074

    // Get the time as a Julian Date (JD) in GPS time system
    let jd_gps = epc.jd_as_time_system(TimeSystem::GPS);
    println!("JD in GPS: {}", jd_gps);
    // Output:
    // JD in GPS: 2460311.000208333

    // Take the difference between two epochs in different time systems
    let epc2 = Epoch::from_datetime(2024, 1, 2, 13, 30, 0.0, 0.0, TimeSystem::GPS);
    let delta_seconds = epc2 - epc;
    println!("Difference between epochs in seconds: {}", delta_seconds);
    // Output:
    // Difference between epochs in seconds: 91782.0

    // Get the epoch as a string in different time systems
    let epc_utc = epc2.to_string_as_time_system(TimeSystem::UTC);
    println!("Epoch in GPS: {}", epc2);
    println!("Epoch in UTC: {}", epc_utc);
    // Outputs:
    // Epoch in GPS: 2024-01-02 13:30:00.000 GPS
    // Epoch in UTC: 2024-01-02 13:29:42.000 UTC
}
```

**Coordinate Transformations:**
``` rust
use brahe as bh;
use brahe::{Epoch, TimeSystem, R_EARTH, state_osculating_to_cartesian,
            state_eci_to_ecef, state_ecef_to_eci, state_cartesian_to_osculating, AngleFormat};
use nalgebra::Vector6;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Define orbital elements
    let a = R_EARTH + 700e3;    // Semi-major axis in meters (700 km altitude)
    let e = 0.001;              // Eccentricity
    let i = 98.7;               // Inclination in degrees
    let raan = 15.0;            // Right Ascension of Ascending Node in degrees
    let arg_periapsis = 30.0;   // Argument of Periapsis in degrees
    let mean_anomaly = 45.0;    // Mean Anomaly in degrees

    // Create a state vector from orbital elements
    let state_kep = Vector6::new(a, e, i, raan, arg_periapsis, mean_anomaly);

    // Convert Keplerian state to ECI coordinates
    let state_eci = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);
    println!("ECI Coordinates: {:?}", state_eci);
    // Outputs:
    // ECI Coordinates: [2026514.0589990876, -527290.0808564089, 6756067.089961103, -6931.980949848838, -2160.9799111629056, 1916.1856855691967]

    // Define a time epoch
    let epoch = Epoch::from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Convert ECI coordinates to ECEF coordinates at the given epoch
    let state_ecef = state_eci_to_ecef(epoch, state_eci);
    println!("ECEF Coordinates: {:?}", state_ecef);
    // Outputs:
    // ECEF Coordinates: [186480.17260881448, -2070225.9929370368, 6760815.482882127, -4538.863726757974, 5777.023453395301, 1899.7320274086795]

    // Convert back from ECEF to ECI coordinates
    let state_eci_2 = state_ecef_to_eci(epoch, state_ecef);
    println!("Recovered ECI Coordinates: {:?}", state_eci_2);
    // Outputs:
    // Recovered ECI Coordinates: [2026514.0589990876, -527290.0808564089, 6756067.089961103, -6931.980949848838, -2160.9799111629056, 1916.1856855691967]

    // Convert back from ECI to Keplerian elements
    let state_kep_2 = state_cartesian_to_osculating(state_eci_2, AngleFormat::Degrees);
    println!("Recovered Keplerian Elements: {:?}", state_kep_2);
    // Outputs:
    // Recovered Keplerian Elements: [7078136.3, 0.001, 98.7, 15.0, 30.0, 45.0]
}
```

**Propagating an Orbit:**
``` rust
use brahe as bh;
use brahe::{Epoch, R_EARTH, KeplerianPropagator, AngleFormat};
use brahe::traits::{OrbitPropagator, OrbitalTrajectory, Trajectory};
use nalgebra::Vector6;

fn main() {
    // Define the initial Keplerian elements
    let a = R_EARTH + 700e3;  // Semi-major axis: 700 km altitude
    let e = 0.001;            // Eccentricity
    let i = 98.7;             // Inclination in degrees
    let raan = 15.0;          // Right Ascension of Ascending Node in degrees
    let argp = 30.0;          // Argument of Perigee in degrees
    let mean_anomaly = 75.0;  // Mean Anomaly at epoch in degrees

    let initial_state = Vector6::new(a, e, i, raan, argp, mean_anomaly);

    // Define the epoch time
    let epoch = Epoch::now();

    // Create the Keplerian Orbit Propagator
    let dt = 60.0;  // Time step in seconds
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch,
        initial_state,
        AngleFormat::Degrees,
        dt
    );

    // Propagate the orbit for 3 time steps
    propagator.propagate_steps(3);

    // States are stored as a Trajectory object
    assert_eq!(propagator.trajectory.len(), 4);  // Initial state + 3 propagated states

    // Convert trajectory to ECI coordinates
    let eci_trajectory = propagator.trajectory.to_eci();

    // Iterate over all stored states
    for i in 0..eci_trajectory.len() {
        let epoch = eci_trajectory.epochs[i];
        let state = eci_trajectory.states[i];
        println!(
            "Epoch: {}, Position (ECI): {:.2} km, {:.2} km, {:.2} km",
            epoch,
            state[0] / 1e3,
            state[1] / 1e3,
            state[2] / 1e3
        );
    }
    // Output (will vary based on current time):
    // Epoch: 2025-10-24 22:14:56.707 UTC, Position (ECI): -1514.38 km, -1475.59 km, 6753.03 km
    // Epoch: 2025-10-24 22:15:56.707 UTC, Position (ECI): -1935.70 km, -1568.01 km, 6623.80 km
    // Epoch: 2025-10-24 22:16:56.707 UTC, Position (ECI): -2349.19 km, -1654.08 km, 6467.76 km
    // Epoch: 2025-10-24 22:17:56.707 UTC, Position (ECI): -2753.17 km, -1733.46 km, 6285.55 km

    // Propagate for 7 days
    let end_epoch = epoch + 86400.0 * 7.0;  // 7 days later
    propagator.propagate_to(end_epoch);

    // Confirm the final epoch is close to expected time
    let time_diff = (propagator.current_epoch() - end_epoch).abs();
    assert!(time_diff < 1.0e-6, "Final epoch should be within 1 second of target");
    println!("Propagation complete. Final epoch: {}", propagator.current_epoch());
    // Output (will vary based on current time):
    // Propagation complete. Final epoch: 2025-10-31 22:18:40.413 UTC
}
```

**Computing ISS Access Windows:**
``` rust
use brahe as bh;
use brahe::{Epoch, PointLocation, ElevationConstraint, location_accesses};
use brahe::datasets::celestrak::get_tle_by_id_as_propagator;
use brahe::utils::Identifiable;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Set the location
    let location = PointLocation::new(-122.4194, 37.7749, 0.0)
        .with_name("San Francisco");

    // Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
    let propagator = get_tle_by_id_as_propagator(25544, None, 60.0).unwrap();

    // Configure Search Window
    let epoch_start = Epoch::now();
    let epoch_end = epoch_start + 7.0 * 86400.0;  // 7 days later

    // Set access constraints -> Must be above 10 degrees elevation
    let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();

    // Compute access windows
    let windows = location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        None,
        None
    );

    assert!(!windows.is_empty(), "Should find at least one access window");

    // Print first 3 access windows
    for window in windows.iter().take(3) {
        println!(
            "Access Window: {} to {}, Duration: {:.2} minutes",
            window.window_open,
            window.window_close,
            window.duration() / 60.0
        );
    }
    // Outputs (will vary based on current time and ISS orbit):
    // Access Window: 2025-10-25 08:49:40.062 UTC to 2025-10-25 08:53:48.463 UTC, Duration: 4.14 minutes
    // Access Window: 2025-10-25 10:25:40.245 UTC to 2025-10-25 10:31:48.463 UTC, Duration: 6.14 minutes
    // Access Window: 2025-10-25 12:05:33.455 UTC to 2025-10-25 12:06:48.463 UTC, Duration: 1.25 minutes
}
```