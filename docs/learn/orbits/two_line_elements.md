# Two-Line Elements (TLEs)

Two-Line Elements are a standardized format for distributing satellite orbital parameters.

## Overview

A Two-Line Element Set (TLE) is a data format used to convey orbital parameters for Earth-orbiting objects. TLEs are widely used for:

- Satellite tracking
- Orbit prediction
- Space situational awareness
- Amateur radio satellite tracking

## TLE Format

A TLE consists of three lines:

```
ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005
2 25544  51.6400 247.4627 0001220  89.6300 270.4997 15.54057614123456
```

**Line 0**: Satellite name (optional)
**Line 1**: General orbital information
**Line 2**: Orbital elements

### Line 1 Fields

- Satellite catalog number
- Classification (U=unclassified)
- International designator
- Epoch (year and day of year)
- First derivative of mean motion (ballistic coefficient)
- Second derivative of mean motion
- Drag term (B*)
- Element set number
- Checksum

### Line 2 Fields

- Satellite catalog number
- Inclination (degrees)
- Right ascension of ascending node (degrees)
- Eccentricity (decimal point assumed)
- Argument of perigee (degrees)
- Mean anomaly (degrees)
- Mean motion (revolutions per day)
- Revolution number
- Checksum

## Using TLEs in Brahe

### Parsing TLEs

```python
import brahe as bh

# Parse TLE from strings
line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
line2 = "2 25544  51.6400 247.4627 0001220  89.6300 270.4997 15.54057614123456"

tle = bh.TLE(line1, line2, name="ISS (ZARYA)")
```

### Extracting Orbital Elements

```python
import brahe as bh

# Get orbital elements from TLE
inclination = tle.inclination()  # radians
raan = tle.raan()  # radians
eccentricity = tle.eccentricity()
argp = tle.argp()  # radians
mean_anomaly = tle.mean_anomaly()  # radians
mean_motion = tle.mean_motion()  # revolutions/day
```

### Propagation with SGP4

```python
import brahe as bh

# Create SGP4 propagator from TLE
propagator = bh.SGPPropagator.from_tle(tle)

# Propagate to specific epoch
epoch = bh.Epoch.from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, bh.UTC)
state = propagator.propagate(epoch)
```

## TLE Sources

### CelesTrak

CelesTrak provides TLEs for thousands of satellites:

```python
import brahe as bh

# Download TLEs for all active satellites
tles = bh.celestrak_fetch_latest_tles("active")

# Download TLEs for specific satellite group
iss_tles = bh.celestrak_fetch_latest_tles("stations")
```

### Space-Track

Space-Track.org (requires free account) provides:

- Historical TLEs
- High-precision TLEs
- TLE predictions

## TLE Accuracy and Limitations

### Accuracy

- **Short-term** (< 1 day): Position accuracy ~1 km
- **Medium-term** (1-7 days): Accuracy degrades to ~10 km
- **Long-term** (> 7 days): Not recommended, errors can exceed 100 km

### Limitations

- **Simplified model**: SGP4/SDP4 uses simplified atmospheric drag
- **No maneuvers**: TLEs don't account for spacecraft maneuvers
- **Aging**: Accuracy degrades rapidly over time
- **Low orbits**: Drag modeling less accurate at low altitudes

## Best Practices

1. **Use recent TLEs**: Download fresh TLEs daily for operational systems
2. **Don't extrapolate far**: Limit propagation to a few days from TLE epoch
3. **Validate results**: Cross-check with other data sources when possible
4. **Update after maneuvers**: Get new TLEs after spacecraft maneuvers

## See Also

- [TLE API Reference](../library_api/orbits/tle.md)
- [SGP Propagation](orbit_propagation/sgp_propagation.md)
- [CelesTrak Dataset](../library_api/datasets/celestrak.md)
- [CelesTrak Website](https://celestrak.org/)
