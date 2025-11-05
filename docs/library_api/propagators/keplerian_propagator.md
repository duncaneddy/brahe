# Keplerian Propagator

Analytical two-body orbit propagator using Keplerian orbital elements.

::: brahe.KeplerianPropagator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

The Keplerian propagator provides fast, analytical orbit propagation for unperturbed two-body motion. It uses closed-form solutions to Kepler's equations for orbital element propagation.

**Key Features**:
- Fast analytical propagation (no numerical integration)
- Perfect for preliminary analysis and mission design
- No perturbations (atmospheric drag, J2, third-body, etc.)
- Suitable for high-altitude orbits where perturbations are minimal

**Module**: `brahe.orbits`

**When to Use**:
- Preliminary orbit analysis
- High-altitude orbits (GEO, cislunar)
- Short propagation times where perturbations are negligible
- Educational purposes

**When NOT to Use**:
- LEO orbits requiring accuracy beyond a few days
- When atmospheric drag is significant
- When J2 perturbations matter
- Precise orbit determination applications

## Example Usage

```python
import brahe as bh
import numpy as np

# Initial orbital elements [a, e, i, Ω, ω, M] in SI units (m, rad)
# Example: Geostationary orbit
a = 42164000.0        # Semi-major axis (m)
e = 0.0001            # Eccentricity
i = 0.0 * bh.DEG2RAD  # Inclination (rad)
raan = 0.0            # Right ascension of ascending node (rad)
argp = 0.0            # Argument of periapsis (rad)
M = 0.0               # Mean anomaly (rad)

elements = np.array([a, e, i, raan, argp, M])

# Create epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create propagator
prop = bh.KeplerianPropagator(
    epoch=epoch,
    elements=elements,
    element_type=bh.OrbitRepresentation.MEAN_ELEMENTS,
    frame=bh.OrbitFrame.ECI,
    gm=bh.GM_EARTH
)

# Propagate to a future time
future_epoch = epoch + 86400.0  # 1 day later
state = prop.propagate(future_epoch)  # Returns [x, y, z, vx, vy, vz]

# Propagate to multiple times
times = np.linspace(0, 7*86400, 100)  # 1 week in 100 steps
epochs = [epoch + dt for dt in times]
states = prop.propagate_multiple(epochs)

print(f"Propagated to {len(states)} epochs")
print(f"Final position: {states[-1][:3]} m")
```

## Orbital Elements

The propagator accepts orbital elements in the following order:
1. **a** - Semi-major axis (meters)
2. **e** - Eccentricity (dimensionless)
3. **i** - Inclination (radians)
4. **Ω** - Right ascension of ascending node (radians)
5. **ω** - Argument of periapsis (radians)
6. **M** or **ν** - Mean anomaly or true anomaly (radians)

Use `OrbitRepresentation` to specify element type:
- `MEAN_ELEMENTS` - Mean orbital elements with mean anomaly
- `OSCULATING_ELEMENTS` - Osculating elements with true anomaly

---

## See Also

- [SGPPropagator](sgp_propagator.md) - SGP4/SDP4 propagator for TLE data
- [Keplerian Elements](../orbits/keplerian.md) - Orbital element conversion functions
- [OrbitRepresentation](../orbits/enums.md#orbitrepresentation) - Element type specification
