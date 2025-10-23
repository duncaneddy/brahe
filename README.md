<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - Practical Astrodynamics</em>
</p>
<p align="center">
<a href="https://github.com/duncaneddy/brahe/actions/workflows/commit.yml" target="_blank">
    <img src="https://github.com/duncaneddy/brahe/actions/workflows/commit.yml/badge.svg" alt="Test">
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

If you find a bug, have a feature request, or want to contribute, please open an issue or a pull request on the GitHub repository. Contributions are welcome and encouraged! If you see something missing, but don't know how to start contributing, please open an issue and we can discuss it.

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

```python
import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.EARTH_RADIUS + 400e3  # 400 km altitude

# Calculate the orbital period using Kepler's third law
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
```

Here are some common operations to get you started:

**Working with Time:**
```python
import brahe as bh

# Create an epoch from a specific date and time
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, bh.TimeSystem.UTC)

# Convert between time systems
mjd_utc = epc.mjd_as_time_system(bh.TimeSystem.UTC)
mjd_tai = epc.mjd_as_time_system(bh.TimeSystem.TAI)

# Time arithmetic
future_epc = epc + 3600  # Add 3600 seconds (1 hour)
time_diff = future_epc - epc  # Difference in seconds
```

**Coordinate Transformations:**
```python
import brahe as bh
import numpy as np

# Convert Orbital Elements to ECI State Vector
elements = [
    7000e3,    # Semi-major axis in meters
    0.001,     # Eccentricity
    98.7,      # Inclination in degrees
    0.0,       # RAAN in degrees
    0.0,       # Argument of perigee in degrees
    0.0        # True anomaly in degrees
]
eci_satellite = bh.state_osculating_to_cartesian(elements, bh.AngleFormat.DEGREES)

# Convert ECI State Vector to ECEF State Vector
epc = bh.Epoch(2024, 1, 1, 0, 0, 0.0, bh.TimeSystem.UTC)
ecef_satellite = bh.state_eci_to_ecef(epc, eci_state)

# Convert geodetic coordinates to ECEF
lat = 40.0  # degrees North
lon = -105.0  # degrees East
alt = 1000.0  # meters above ellipsoid
geod = np.array([lat, lon, alt])  # [latitude, longitude, altitude]
ecef_location = bh.position_geodetic_to_ecef(geod, bh.AngleFormat.DEGREES)

# Compute Topocentric-Horizon Coordinates of Satellite from Observer Location
enz = bh.relative_position_ecef_to_topocentric(ecef_location, ecef_satellite, angle_format=bh.AngleFormat.DEGREES)

print(f'Azimuth: {enz[0]:.2f} degrees')
print(f'Elevation: {enz[1]:.2f} degrees')
print(f'Range: {enz[2] / 1000:.2f} km')
```

**Propagating an Orbit:**
```python
import brahe as bh
import numpy as np

# Create a Two-Line Element (TLE) for a satellite
tle = bh.TLE(
    "ISS (ZARYA)",
    "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
)

# Create an SGP4 propagator
prop = bh.SGPPropagator.from_tle(tle)

# Propagate to a specific epoch
epc = bh.Epoch(2024, 6, 1, 0, 0, 0.0, bh.TimeSystem.UTC)
state = prop.propagate(epc)  # Returns [x, y, z, vx, vy, vz] in meters and m/s

print(f"Position: {state[:3] / 1000} km")
print(f"Velocity: {state[3:] / 1000} km/s")
```

**Computing ISS Access Windows:**
```python
```

### Rust

To use brahe in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
brahe = "0.5"
```

Here are some common operations to get you started:

**Working with Time:**
```rust
use brahe::time::Epoch;
use brahe::time::TimeSystem;

// Create an epoch from a specific date and time
let epc = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0, TimeSystem::UTC);

// Convert between time systems
let mjd_utc = epc.to_mjd(TimeSystem::UTC);
let mjd_tai = epc.to_mjd(TimeSystem::TAI);

// Time arithmetic
let future_epc = epc + 3600.0;  // Add 3600 seconds (1 hour)
let time_diff = future_epc - epc;  // Difference in seconds
```

**Coordinate Transformations:**
```rust
use brahe::coordinates::position_geodetic_to_ecef;
use brahe::frames::position_ecef_to_eci;
use brahe::time::Epoch;
use brahe::time::TimeSystem;
use brahe::constants::AngleFormat;
use nalgebra::Vector3;

// Convert geodetic coordinates (lat, lon, alt) to ECEF
let geod = Vector3::new(
    40.0_f64.to_radians(),   // latitude in radians
    -105.0_f64.to_radians(), // longitude in radians
    1000.0                   // altitude in meters
);
let ecef = position_geodetic_to_ecef(geod, AngleFormat::Radians);

// Convert ECEF to ECI at a specific epoch
let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0, TimeSystem::UTC);
let eci = position_ecef_to_eci(epc, ecef);
```

**Propagating an Orbit:**
```rust
use brahe::orbits::{TLE, SGPPropagator};
use brahe::time::Epoch;

// Create a Two-Line Element (TLE) for a satellite
let tle = TLE::from_lines(
    "ISS (ZARYA)",
    "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
).unwrap();

// Create an SGP4 propagator
let prop = SGPPropagator::from_tle(&tle);

// Propagate to a specific epoch
let epc = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0, TimeSystem::UTC);
let state = prop.propagate(&epc);  // Returns [x, y, z, vx, vy, vz] in meters and m/s

println!("Position: {:?} km", [state[0]/1000.0, state[1]/1000.0, state[2]/1000.0]);
println!("Velocity: {:?} km/s", [state[3]/1000.0, state[4]/1000.0, state[5]/1000.0]);
```

**Computing ISS Access Windows:**
```rust
```

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- **[Learn](https://duncaneddy.github.io/brahe/learn/)**: Provides short-form documentation of major concepts of the package.
- **[Examples](https://duncaneddy.github.io/brahe/examples/)**: Provides longer-form examples of how-to examples of accomplish common tasks.
- **[Python API Reference](https://duncaneddy.github.io/brahe/library_api/)**: Provides detailed reference documentation of the Python API.
- **[Rust API Reference](https://docs.rs/brahe)**: Provides detailed reference documentation of the Rust API.

