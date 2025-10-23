# Access Computation

Access computation determines when and under what conditions satellites can observe or communicate with ground locations. This is fundamental for mission planning, ground station contact scheduling, imaging opportunity analysis, and communications link budgets.

Brahe provides a comprehensive access computation system that handles the geometric complexity of satellite-ground visibility while offering flexibility through constraints, extensible properties, and performance optimization.

## Overview

Access computation involves:

- **Geometric visibility**: Is the satellite above the horizon from the ground location's perspective?
- **Constraint satisfaction**: Do viewing angles, lighting conditions, or other requirements meet mission needs?
- **Time window identification**: When do access periods start and end?
- **Property computation**: What are the specific characteristics (elevation, azimuth, range, etc.) during access?

## Core Concepts

### Locations

Locations represent ground positions that satellites can access:

- **PointLocation**: Discrete points (ground stations, imaging targets)
- **PolygonLocation**: Areas of interest (countries, regions, imaging swaths)

All locations support:
- Geodetic coordinates (latitude, longitude, altitude)
- ECEF coordinates (Earth-fixed Cartesian)
- GeoJSON interoperability
- Extensible properties via metadata dictionary
- Identifiable trait (name, ID, UUID)

See [Locations](locations.md) for details.

### Constraints

Constraints define access criteria that must be satisfied:

**Built-in Constraints**:
- `ElevationConstraint`: Minimum/maximum elevation angles
- `ElevationMaskConstraint`: Azimuth-dependent elevation masks
- `OffNadirConstraint`: Satellite pointing angles
- `LookDirectionConstraint`: Left/right/either looking direction
- `LocalTimeConstraint`: Time-of-day access windows
- `OrbitTypeConstraint`: Ascending/descending pass filtering

**Logical Composition**:
- `ConstraintAll`: AND logic (all constraints must be satisfied)
- `ConstraintAny`: OR logic (any constraint must be satisfied)
- `ConstraintNot`: NOT logic (constraint must NOT be satisfied)

**Custom Constraints**:
- Python-defined constraints via `AccessPropertyComputer`
- Arbitrary logic based on epoch, satellite state, and location

See [Constraints](constraints.md) for details.

### Access Windows

Access windows represent periods when constraints are satisfied:

```python
import brahe as bh

# Each window provides:
window.window_open      # Start epoch
window.window_close     # End epoch
window.duration         # Duration in seconds
window.location_id      # Location identifier
window.propagator_id    # Satellite identifier
window.properties       # Computed properties during access
```

### Access Properties

Properties characterize access windows with geometric and custom metrics:

**Built-in Properties**:
- `elevation`: Satellite elevation angle (degrees)
- `azimuth`: Satellite azimuth angle (degrees)
- `range`: Satellite-to-location distance (meters)
- `range_rate`: Range rate of change (m/s)
- `off_nadir`: Satellite off-nadir angle (degrees)
- `look_direction`: LEFT/RIGHT/BOTH viewing geometry

**Custom Properties**:
- Compute arbitrary metrics using Python functions
- Access to epoch, satellite state (ECI/ECEF), and location
- Properties stored in window for post-processing

See [Computation](computation.md) for details.

## Basic Workflow

1. **Define locations**: Create ground sites or areas of interest
2. **Setup propagators**: Initialize satellite orbit propagators
3. **Configure constraints**: Specify access criteria
4. **Compute accesses**: Find time windows when constraints are satisfied
5. **Analyze results**: Extract properties and schedule activities

**Example**:

```python
import brahe as bh
import numpy as np

# Initialize EOP provider
eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
bh.set_global_eop_provider(eop)

# Define ground station
svalbard = bh.PointLocation(15.4, 78.2, 0.0).with_name("Svalbard")

# Create satellite propagator
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 600e3, 0.001, 97.8, 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
propagator = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("Satellite-1")

# Define constraint (10° minimum elevation)
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute accesses over 24 hours
windows = bh.location_accesses(
    [svalbard],
    [propagator],
    epoch,
    epoch + 86400.0,
    constraint
)

print(f"Found {len(windows)} access windows")
for window in windows:
    duration_min = window.duration / 60.0
    print(f"  {window.window_open} - {window.window_close} ({duration_min:.1f} min)")
```

## Performance Optimization

Access computation can be computationally intensive for large-scale analyses. Brahe provides several optimization strategies:

### Parallelization

By default, access computation runs in parallel using 90% of available CPU cores:

```python
# Default: parallel with 90% of cores
windows = bh.location_accesses(
    locations,
    propagators,
    start_epoch,
    end_epoch,
    constraint
)

# Explicit configuration
config = bh.AccessSearchConfig(
    initial_time_step=60.0,
    adaptive_step=False,
    parallel=True,        # Enable parallelization
    num_threads=None      # Use global default (90% of cores)
)

windows = bh.location_accesses(
    locations,
    propagators,
    start_epoch,
    end_epoch,
    constraint,
    config=config
)
```

### Thread Pool Configuration

Control parallelization globally or per-computation:

```python
# Set specific number of threads (must be called before any parallel operations)
bh.set_num_threads(4)

# Or use all available CPU cores
bh.set_max_threads()

# Or go LUDICROUS SPEED (alias for set_max_threads)
bh.set_ludicrous_speed()

# Configure per-computation
config = bh.AccessSearchConfig(
    parallel=True,
    num_threads=8  # Use 8 threads for this computation
)

# Or disable parallelization entirely
config = bh.AccessSearchConfig(parallel=False)
```

### Adaptive Time Stepping

Adaptive stepping increases efficiency by using larger time steps when constraints are far from satisfied:

```python
config = bh.AccessSearchConfig(
    initial_time_step=60.0,
    adaptive_step=True,           # Enable adaptive stepping
    adaptive_fraction=0.75,       # Aggressiveness factor
    parallel=True
)
```

**When adaptive stepping helps**:
- Long search periods with sparse accesses
- Tight elevation constraints (e.g., > 60°)
- Complex composed constraints

**When to disable**:
- Very frequent accesses (LEO constellation to global coverage)
- Short search periods
- When deterministic timing is required

### Performance Tips

1. **Batch processing**: Compute multiple location-satellite pairs in one call
2. **Pre-filter candidates**: Use geometric screening before detailed constraint checking
3. **Adjust time step**: Balance between accuracy and performance
4. **Use parallelization**: Leverage multiple cores for large problems
5. **Cache propagators**: Reuse state providers across multiple access computations

## Use Cases

**Ground Station Contact Scheduling**:
- Find all passes above minimum elevation
- Filter by local time constraints (operational hours)
- Compute pass characteristics (max elevation, duration)

**Imaging Opportunity Analysis**:
- Identify when targets are within sensor field of view
- Filter by off-nadir angle and look direction
- Compute sun angles and lighting conditions

**Communications Link Analysis**:
- Determine line-of-sight windows
- Compute range and range-rate for Doppler estimation
- Assess elevation angles for link margin

**Constellation Coverage Analysis**:
- Analyze global or regional coverage
- Identify coverage gaps
- Optimize constellation design

## Common Patterns

### Multiple Locations and Satellites

```python
# Define multiple ground stations
locations = [
    bh.PointLocation(15.4, 78.2, 0.0).with_name("Svalbard"),
    bh.PointLocation(-64.5, -31.5, 0.0).with_name("Malargue"),
    bh.PointLocation(-117.2, 34.1, 0.0).with_name("Goldstone"),
]

# Define constellation
propagators = []
for i in range(5):
    oe = np.array([bh.R_EARTH + 550e3, 0.001, 97.8, i*30.0, 0.0, i*45.0])
    state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name(f"Sat-{i}")
    propagators.append(prop)

# Compute all location-satellite combinations
windows = bh.location_accesses(locations, propagators, start, end, constraint)

# Windows are sorted by opening time
# Each window has location_id and propagator_id for filtering
```

### Complex Constraints

```python
# Combine multiple constraints with AND logic
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=10.0),
    bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0),  # Daylight operations
    bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT),
])

# OR logic: Either high elevation OR ascending pass
constraint = bh.ConstraintAny([
    bh.ElevationConstraint(min_elevation_deg=60.0),
    bh.OrbitTypeConstraint(orbit_type=bh.AscDsc.ASC),
])

# NOT logic: Exclude nighttime passes
daytime = bh.ConstraintNot(
    bh.LocalTimeConstraint(min_hour=18.0, max_hour=8.0)  # NOT nighttime
)
```

### Custom Property Computation

```python
import math

# Define custom property computer
def compute_slant_range_km(epoch, sat_state_eci, sat_state_ecef, location_ecef):
    """Compute slant range in kilometers"""
    range_m = np.linalg.norm(sat_state_ecef[:3] - location_ecef)
    return {"slant_range_km": range_m / 1000.0}

# Create property computer
computer = bh.AccessPropertyComputer(compute_slant_range_km)

# Compute accesses with custom properties
config = bh.AccessSearchConfig()
windows = bh.location_accesses(
    locations,
    propagators,
    start,
    end,
    constraint,
    config=config,
    property_computers=[computer]
)

# Access custom properties
for window in windows:
    props = window.properties
    print(f"Slant range: {props['slant_range_km']:.1f} km")
```

## See Also

- [Locations](locations.md) - Ground location types and GeoJSON support
- [Constraints](constraints.md) - Built-in and custom constraint types
- [Computation](computation.md) - Access algorithms and property computation
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md) - Complete ground station example
- [Example: Computing Imaging Opportunities](../../examples/imaging_opportunities.md) - Imaging scenario
- [API Reference: Access Module](../../library_api/access/index.md) - Complete API documentation
