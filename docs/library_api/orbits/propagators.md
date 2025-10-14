# Orbit Propagators

Propagator classes for computing satellite trajectories.

## Keplerian Propagator

Analytical propagator using Keplerian orbital elements for simplified two-body dynamics.

### Class Documentation

The `KeplerianPropagator` class implements analytical Keplerian orbit propagation for two-body dynamics.

```python
from brahe import KeplerianPropagator, Epoch
import numpy as np

# Create propagator with initial epoch and state
epc = Epoch.now()
state = np.array([7000e3, 0, 0, 0, 7.5e3, 0])  # Position (m) and velocity (m/s)
prop = KeplerianPropagator(epc, state, "ECI", "CARTESIAN", "RADIANS")

# Propagate to a new epoch
new_epc = epc + 600  # 10 minutes later
new_state = prop.propagate(new_epc)
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/orbits/struct.KeplerianPropagator.html).

## SGP4 Propagator

Simplified General Perturbations 4 propagator for Earth-orbiting satellites using TLE data.

### Class Documentation

The `SGPPropagator` class implements the SGP4/SDP4 propagation model for Earth-orbiting satellites using Two-Line Element (TLE) data.

```python
from brahe import SGPPropagator, TLE, Epoch

# Create propagator from TLE
tle = TLE(
    "ISS (ZARYA)",
    "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
)
prop = SGPPropagator.from_tle(tle)

# Propagate to an epoch
epc = Epoch.now()
state = prop.propagate(epc)
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/orbits/struct.SGPPropagator.html).