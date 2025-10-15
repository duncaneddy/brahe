# Two-Line Elements (TLE) and Simplified General Perturbations (SGP) Models

Two-Line Element (TLE) sets are a standardized data format for distributing satellite orbital elements. Originally developed by NORAD (North American Aerospace Defense Command) in the 1960s, TLEs remain the most widely used format for sharing satellite ephemeris data.

??? warning
    TLEs are designed specifically for use with the Simplified General Perturbation 4 (SGP4) orbit propagation models. They are not suitable for other propagation methods without conversion. For high-precision applications, consider using osculating elements or numerical propagators.

??? abstract
    There is an upcoming problem of running out of NORAD IDs. The current format supports 5-digit IDs (up to 99999). Space Force has officially
    augmented the format my moving to something known as "alpha-5" TLEs,
    which use a letter as the leading character to extend the range. This is not yet widely adopted.

    Given the limited representational capacity of TLEs, there is a push
    to move to more modern formats such as General Perturbations (GP) elements.

## Overview

A TLE encodes the orbital state of a satellite at a specific epoch, along with drag and perturbation coefficients needed for propagation using the SGP4/SDP4 models.

## Format Variations

### Two-Line Element (2LE)

The standard format consists of two 69-character lines:

```
1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990
2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104
```

**Line 1** contains:
- Satellite catalog number (NORAD ID)
- International designator
- Epoch (year and day of year with fractional day)
- First derivative of mean motion (ballistic coefficient)
- Second derivative of mean motion
- BSTAR drag term
- Element set number
- Checksum

**Line 2** contains:
- Satellite catalog number (repeated)
- Inclination (degrees)
- Right ascension of ascending node (degrees)
- Eccentricity (decimal point assumed)
- Argument of perigee (degrees)
- Mean anomaly (degrees)
- Mean motion (revolutions per day)
- Revolution number at epoch
- Checksum

### Three-Line Element (3LE)

The extended format adds a satellite name line:

```
ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990
2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104
```

The name line (line 0) can be up to 24 characters and helps identify satellites when working with large datasets.

### Checksums

Each TLE line ends with a modulo-10 checksum:
- Computed from all characters in the line (excluding the checksum itself)
- Digits contribute their value (`0-9`)
- Minus signs (`-`) contribute 1
- All other characters contribute 0
- Sum taken modulo 10

Brahe automatically validates checksums when parsing TLEs and computes them when creating new TLE lines.

## TLE Limitations

### Accuracy Degradation

TLEs are mean orbital elements, not osculating (instantaneous) elements. They're designed to work specifically with SGP4/SDP4 propagators and may give incorrect results with other propagation methods. They were designed to provide reasonable accuracy over short periods of time to support general object tracking. Because they fit a mean-element model to observations, they inherently do not model the true orbit perfectly and contain some level of error, even at epoch.

**Typical accuracy:**
- **Within hours of epoch**: ~1 km position error
- **Within days of epoch**: ~1-10 km position error
- **Beyond 1-2 weeks**: Accuracy degrades significantly

For high-precision applications, use osculating elements or numerical propagators with force models.

### Atmospheric Drag

The BSTAR drag coefficient in line 1 is a fitted parameter, not a physical measurement. It compensates for:
- Atmospheric density variations
- Satellite ballistic properties
- Solar activity effects

TLEs for the same satellite issued at different times may have different BSTAR values as the atmosphere and solar conditions change.

### Data Currency

TLEs become stale quickly due to:
- Atmospheric drag (LEO satellites)
- Solar radiation pressure
- Gravitational perturbations
- Orbital maneuvers

**Recommended refresh intervals:**
- **LEO satellites**: Daily to weekly
- **MEO satellites**: Weekly
- **GEO satellites**: Monthly

## TLE Data Sources

There are a few primary sources for obtaining TLE data:

### CelesTrak

Maintained by T.S. Kelso, CelesTrak provides free TLE data grouped by satellite type and mission. Access via Brahe's datasets module:

```python
import brahe as bh

# Get GPS constellation TLEs
gps_ephemeris = bh.datasets.celestrak.get_ephemeris("gnss")

# Get recently launched satellites
recent = bh.datasets.celestrak.get_ephemeris("last-30-days")
```

<!-- ### Space-Track.org

Operated by the U.S. Space Force, Space-Track provides:
- Historical TLE data
- Higher update frequency
- More satellite categories
- Conjunction warnings
- Debris tracking

Requires free registration. (Future Brahe support planned) -->

## Working with TLEs in Brahe

### Parsing and Validation

```python
import brahe as bh

# Validate TLE format
line1 = "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990"
line2 = "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"

is_valid = bh.validate_tle_lines(line1, line2)

# Extract epoch
epoch = bh.epoch_from_tle(tle)

# Convert to Keplerian elements
keplerian = bh.keplerian_elements_from_tle(tle)
# Returns: [a, e, i, raan, argp, M] in SI units (meters, radians)
```

### Propagation

Because SGP4 is a semi-analytical model, the [SGPPropagator](orbit_propagation/sgp_propagation.md) class is used to propagate TLEs is considered an "Analytic Propagator" in Brahe, which provides direct computation of the orbital state at any epoch without numerical integration.

The following methods are available:
- `SGPPropagtor.state(epoch)` - Get state vector at specified epoch in the native TLE frame (TEME)
- `SGPPropagator.state_eci(epoch)` - Get state vector in ECI frame (GCRF)
- `SGPPropagator.state_ecef(epoch)` - Get state vector in ECEF frame (ITRF)
- `SGPPropagator.state_pef(epoch)` - Get position in the intermediate Psudo Earth-Fixed (PEF) frame

```python

import brahe as bh
import numpy as np

# Create a Two-Line Element (TLE) for a satellite
tle = bh.SGPPropagator.from_3le(
    "ISS (ZARYA)",
    "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104",
    60.0 # Propagator step size. Only used for auto-stepping computations
)

# Propagate to a specific epoch
epc = bh.Epoch(2024, 6, 1, 0, 0, 0.0, bh.TimeSystem.UTC)

state = tle.state_eci(epc)  # Returns [x, y, z, vx, vy, vz] in meters and m/s

# You can also get the state in other frames and formats
state_ecef = tle.state_ecef(epc)  # ECEF frame
```

### Batch Processing

```python
import brahe as bh

# Get ephemeris and initialize propagators directly
propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
    "gnss",
    step_size=60.0
)

# Propagate all satellites to same epoch
epoch = bh.Epoch.from_datetime(2021, 1, 2, 12, 0, 0, tsys="UTC")
states = [prop.state(epoch) for prop in propagators]
```

<!-- ### Element Extraction

```python

```

### TLE Creation

```python
import brahe as bh
``` -->

## See Also

- [SGP Propagation](orbit_propagation/sgp_propagation.md) - Using TLEs for orbit propagation
- [Datasets](datasets/index.md) - Downloading satellite ephemeris data
- [TLE Functions API Reference](../library_api/orbits/tle.md) - Functions for manipulating and working with TLEs
