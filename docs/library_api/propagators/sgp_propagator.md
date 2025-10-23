# SGP Propagator

The SGP4/SDP4 propagator for satellite orbit propagation using Two-Line Element (TLE) data.

::: brahe.SGPPropagator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

The SGP (Simplified General Perturbations) propagator implements the SGP4/SDP4 models for propagating satellites using TLE orbital data. This is the standard model used for tracking objects in Earth orbit and is maintained by NORAD/Space Force.

**Key Features**:
- Industry-standard orbit propagation
- Atmospheric drag modeling
- Automatic selection between SGP4 (near-Earth) and SDP4 (deep-space) models
- Compatible with standard TLE format

**Module**: `brahe.orbits`

## Example Usage

```python
import brahe as bh

# ISS TLE data (example)
line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9005"
line2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000300000"

# Create propagator from TLE
prop = bh.SGPPropagator.from_tle(line1, line2)

# Get current epoch
epoch = prop.epoch()

# Propagate to a specific time
future_epoch = epoch + 3600.0  # 1 hour later
state = prop.propagate(future_epoch)  # Returns [x, y, z, vx, vy, vz] in TEME frame

# Propagate to multiple times
import numpy as np
times = np.linspace(0, 86400, 100)  # 1 day in 100 steps
epochs = [epoch + dt for dt in times]
states = prop.propagate_multiple(epochs)  # Returns array of states
```

## See Also

- [KeplerianPropagator](keplerian_propagator.md) - Analytical two-body propagator
- [TLE](../orbits/tle.md) - Two-Line Element format details
- [Keplerian Elements](../orbits/keplerian.md) - Orbital element functions
