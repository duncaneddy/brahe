# Constraints

Constraints define the criteria that must be satisfied for satellite access to ground locations. Brahe provides a comprehensive constraint system with built-in geometric constraints, logical composition operators, and support for custom user-defined constraints.

## Overview

Access constraints answer questions like:

- "Is the satellite above 10° elevation?"
- "Is the satellite looking in the right direction?"
- "Is it daytime at the ground location?"
- "Are all of these conditions satisfied simultaneously?"

Constraints are evaluated at each time step during access computation to determine when access windows open and close.

## Core Concept

All constraints implement the `AccessConstraint` trait, which provides:

```python
def evaluate(epoch, sat_state_ecef, location_ecef) -> bool:
    """
    Returns True if constraint is satisfied, False otherwise.

    Args:
        epoch: Time of evaluation
        sat_state_ecef: Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
        location_ecef: Location in ECEF [x, y, z] (meters)
    """
    pass
```

This simple interface enables powerful composition and extension.

## Built-in Constraints

### Elevation Constraints

#### ElevationConstraint

Constrains access based on satellite elevation angle above the local horizon:

```python
import brahe as bh

# Minimum elevation only (typical ground station)
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Both bounds (avoid low and zenith angles)
constraint = bh.ElevationConstraint(
    min_elevation_deg=10.0,
    max_elevation_deg=85.0
)

# Maximum only (avoid zenith)
constraint = bh.ElevationConstraint(max_elevation_deg=80.0)
```

**When to use**:
- Ground station visibility (min elevation > 5-10°)
- Link budget requirements (min elevation affects signal strength)
- Avoiding multipath (exclude low elevations)
- Zenith avoidance for dish antennas (max elevation)

**Design note**: At least one bound (min or max) must be specified. An unbounded constraint would match everything and serves no purpose.

#### ElevationMaskConstraint

Azimuth-dependent elevation masks for terrain obstructions or antenna limitations:

```python
# Define mask as (azimuth_deg, min_elevation_deg) pairs
mask = [
    (0.0, 10.0),     # North: 10° minimum
    (90.0, 5.0),     # East: 5° minimum
    (180.0, 15.0),   # South: 15° minimum (mountain)
    (270.0, 5.0),    # West: 5° minimum
    (360.0, 10.0),   # Wrap to north
]

constraint = bh.ElevationMaskConstraint(mask)
```

**Key features**:
- Linear interpolation between azimuth points
- Automatic wrapping at 0°/360°
- Must be sorted by azimuth in ascending order

**When to use**:
- Ground stations with terrain obstructions
- Antenna mechanical limitations
- Building/structure interference
- Regulatory restrictions by direction

### Satellite Pointing Constraints

#### OffNadirConstraint

Constrains satellite off-nadir angle (angle between satellite-to-location vector and nadir):

```python
# Imaging sensor with 30° max off-nadir
constraint = bh.OffNadirConstraint(
    min_off_nadir_deg=0.0,
    max_off_nadir_deg=30.0
)

# Minimum off-nadir (avoid direct nadir)
constraint = bh.OffNadirConstraint(min_off_nadir_deg=5.0)
```

**When to use**:
- Imaging missions with sensor field-of-view limits
- Avoiding geometry that causes distortion
- Nadir-pointing vs. off-nadir pointing modes
- Synthetic aperture radar (SAR) geometry requirements

**Geometry note**: 0° off-nadir = directly below satellite (nadir pointing). Larger angles = more oblique viewing.

#### LookDirectionConstraint

Constrains satellite look direction (left/right relative to velocity vector):

```python
# Right-looking only
constraint = bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT)

# Left-looking only
constraint = bh.LookDirectionConstraint(look_direction=bh.LookDirection.LEFT)

# Either direction (permissive)
constraint = bh.LookDirectionConstraint(look_direction=bh.LookDirection.BOTH)
```

**When to use**:
- Imaging satellites with fixed-side sensors
- SAR missions with specific look-direction requirements
- Avoiding sun glint (prefer specific look direction)
- Stereo imaging pairs (require consistent look direction)

**Geometry note**: Look direction is computed relative to satellite velocity vector using cross product. BOTH is equivalent to no constraint.

### Temporal Constraints

#### LocalTimeConstraint

Constrains access based on local solar time at the ground location:

```python
# Daylight operations only (8 AM - 6 PM local)
constraint = bh.LocalTimeConstraint(
    min_hour=8.0,
    max_hour=18.0
)

# Nighttime operations (10 PM - 4 AM local)
constraint = bh.LocalTimeConstraint(
    min_hour=22.0,
    max_hour=4.0
)

# Early morning (wrap around midnight)
constraint = bh.LocalTimeConstraint(
    min_hour=4.0,
    max_hour=8.0
)
```

**Key features**:
- Based on sun position (local solar time)
- Automatically handles day/night wrap-around
- Hours are floating-point (e.g., 13.5 = 1:30 PM)

**When to use**:
- Operational hour restrictions
- Imaging with sun angle requirements
- Avoiding local midnight (thermal constraints)
- Sun-synchronous orbit planning

**Implementation note**: Uses sun position calculations, not time zones. This is more accurate for satellite applications.

### Orbit Geometry Constraints

#### OrbitTypeConstraint (AscDscConstraint)

Filters by ascending vs. descending passes:

```python
# Ascending passes only (southbound to northbound)
constraint = bh.AscDscConstraint(orbit_type=bh.AscDsc.ASC)

# Descending passes only (northbound to southbound)
constraint = bh.AscDscConstraint(orbit_type=bh.AscDsc.DSC)

# Either (no filtering)
constraint = bh.AscDscConstraint(orbit_type=bh.AscDsc.BOTH)
```

**When to use**:
- Sun-synchronous orbits (different local times for asc/dsc)
- Imaging with specific sun-angle requirements
- Ground station scheduling (separate asc/dsc antennas)
- Radar interferometry (consistent geometry)

**Geometry note**: Determined by sign of latitude rate (dφ/dt).

## Logical Composition

Combine constraints with Boolean logic:

### ConstraintAll (AND)

All child constraints must be satisfied:

```python
# Ground station with elevation and time constraints
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=10.0),
    bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0),
])

# Complex imaging requirements
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=30.0),
    bh.OffNadirConstraint(max_off_nadir_deg=25.0),
    bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT),
    bh.AscDscConstraint(orbit_type=bh.AscDsc.ASC),
])
```

**Behavior**: Returns `True` only if ALL child constraints return `True`.

**Short-circuit evaluation**: Stops checking as soon as any constraint returns `False` (performance optimization).

### ConstraintAny (OR)

At least one child constraint must be satisfied:

```python
# High elevation OR ascending pass
constraint = bh.ConstraintAny([
    bh.ElevationConstraint(min_elevation_deg=60.0),  # Very high passes
    bh.AscDscConstraint(orbit_type=bh.AscDsc.ASC),  # OR ascending
])

# Multiple time windows
constraint = bh.ConstraintAny([
    bh.LocalTimeConstraint(min_hour=8.0, max_hour=12.0),   # Morning
    bh.LocalTimeConstraint(min_hour=14.0, max_hour=18.0),  # Afternoon
])
```

**Behavior**: Returns `True` if ANY child constraint returns `True`.

**Short-circuit evaluation**: Stops checking as soon as any constraint returns `True`.

### ConstraintNot (NOT)

Inverts constraint result:

```python
# Exclude nighttime (i.e., daytime only)
constraint = bh.ConstraintNot(
    bh.LocalTimeConstraint(min_hour=20.0, max_hour=6.0)  # NOT nighttime
)

# Avoid nadir pointing
constraint = bh.ConstraintNot(
    bh.OffNadirConstraint(max_off_nadir_deg=5.0)  # NOT near-nadir
)
```

**Behavior**: Returns opposite of child constraint (`not child.evaluate(...)`).

**Design tip**: Often clearer to use opposite bounds rather than NOT, but NOT is essential for complex compositions.

### Nested Composition

Combine logical operators arbitrarily:

```python
# (High elevation AND daytime) OR (Medium elevation AND right-looking)
constraint = bh.ConstraintAny([
    bh.ConstraintAll([
        bh.ElevationConstraint(min_elevation_deg=60.0),
        bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0),
    ]),
    bh.ConstraintAll([
        bh.ElevationConstraint(min_elevation_deg=30.0),
        bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT),
    ]),
])

# Complex exclusion logic
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=10.0),  # Basic visibility
    bh.ConstraintNot(  # NOT (nighttime AND low elevation)
        bh.ConstraintAll([
            bh.LocalTimeConstraint(min_hour=20.0, max_hour=6.0),
            bh.ElevationConstraint(max_elevation_deg=30.0),
        ])
    ),
])
```

**Performance note**: Evaluation is lazy and short-circuits. Order child constraints with most likely to fail first.

## Custom Constraints

Define application-specific constraints by subclassing `AccessConstraintComputer`:

### Basic Custom Constraint

```python
import brahe as bh
import numpy as np

class SlantRangeConstraint(bh.AccessConstraintComputer):
    """
    Custom constraint that limits access based on slant range.

    Only allows access when satellite is within 2000 km of location.
    """

    def __init__(self, max_range_km=2000.0):
        self.max_range_m = max_range_km * 1000.0

    def evaluate(self, epoch, sat_state_ecef, location_ecef):
        """
        Check if satellite is within maximum slant range.

        Args:
            epoch: Current evaluation time
            sat_state_ecef: Satellite state [x,y,z,vx,vy,vz] in ECEF (m, m/s)
            location_ecef: Location position [x,y,z] in ECEF (m)

        Returns:
            bool: True if within range, False otherwise
        """
        sat_pos = sat_state_ecef[:3]
        range_m = np.linalg.norm(sat_pos - location_ecef)
        return range_m < self.max_range_m

    def name(self):
        """Return constraint name"""
        return f"SlantRange(max={self.max_range_m/1000:.0f}km)"

# Use with access computation
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=10.0),  # Built-in constraint
    SlantRangeConstraint(max_range_km=2000.0),       # Custom constraint
])

windows = bh.location_accesses(
    locations, propagators, start, end, constraint
)
```

### Advanced Custom Constraint

```python
class NorthernHemisphereConstraint(bh.AccessConstraintComputer):
    """
    Only allows access when satellite is in northern hemisphere.

    Useful for sun-synchronous orbits or regional coverage requirements.
    """

    def evaluate(self, epoch, sat_state_ecef, location_ecef):
        """Check if satellite Z-coordinate is positive (northern hemisphere)"""
        z_coord = sat_state_ecef[2]  # Z in ECEF
        return z_coord >= 0.0

    def name(self):
        return "NorthernHemisphere"

class SunAngleConstraint(bh.AccessConstraintComputer):
    """
    Constraint based on sun elevation angle at location.

    Requires daylight conditions for optical imaging.
    """

    def __init__(self, min_sun_elevation_deg=10.0, max_sun_elevation_deg=70.0):
        self.min_sun_elev = np.radians(min_sun_elevation_deg)
        self.max_sun_elev = np.radians(max_sun_elevation_deg)

    def evaluate(self, epoch, sat_state_ecef, location_ecef):
        """Check if sun elevation is within acceptable range"""
        # Compute sun position (placeholder - use actual ephemeris)
        # Real implementation would use:
        # sun_pos = bh.sun_position(epoch)
        # sun_elev = compute_sun_elevation(sun_pos, location_ecef)

        # For demonstration:
        sun_elevation = 0.5  # Radians (placeholder)

        return self.min_sun_elev <= sun_elevation <= self.max_sun_elev

    def name(self):
        return f"SunAngle({np.degrees(self.min_sun_elev):.0f}°-{np.degrees(self.max_sun_elev):.0f}°)"

# Combine with built-in constraints
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=30.0),
    bh.OffNadirConstraint(max_off_nadir_deg=25.0),
    SunAngleConstraint(min_sun_elevation_deg=10.0, max_sun_elevation_deg=70.0),
])
```

### Stateful Custom Constraints

Custom constraints can maintain internal state:

```python
class EvaluationCounterConstraint(bh.AccessConstraintComputer):
    """
    Example constraint that counts how many times it's been evaluated.

    Useful for performance profiling or debugging.
    """

    def __init__(self):
        self.evaluation_count = 0

    def evaluate(self, epoch, sat_state_ecef, location_ecef):
        """Always returns True, but counts evaluations"""
        self.evaluation_count += 1
        return True

    def name(self):
        return f"EvaluationCounter(count={self.evaluation_count})"

# Use in access computation
counter = EvaluationCounterConstraint()
windows = bh.location_accesses(locations, propagators, start, end, counter)
print(f"Constraint evaluated {counter.evaluation_count} times")
```

**When to use custom constraints**:
- Domain-specific requirements not covered by built-in constraints
- Complex logic requiring external data (weather, sun angles, etc.)
- Research/experimental constraint types

**Performance note**: Custom Python constraints are slower than built-in Rust constraints (~100-1000× slower). For performance-critical applications, combine custom constraints with restrictive built-in constraints using `ConstraintAll` to minimize custom constraint evaluations.

## Design Patterns

### Layered Constraints

Build constraints from permissive to restrictive:

```python
# Layer 1: Basic visibility
basic = bh.ElevationConstraint(min_elevation_deg=5.0)

# Layer 2: Add operational constraints
operational = bh.ConstraintAll([
    basic,
    bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0),
])

# Layer 3: Add mission-specific requirements
mission = bh.ConstraintAll([
    operational,
    bh.OffNadirConstraint(max_off_nadir_deg=30.0),
    bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT),
])
```

**Benefit**: Easy to test intermediate constraint sets and identify which layer is most restrictive.

### Constraint Reuse

Define common constraint components once:

```python
# Common constraint sets
GROUND_STATION_BASIC = bh.ElevationConstraint(min_elevation_deg=10.0)

DAYLIGHT_OPS = bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0)

IMAGING_GEOMETRY = bh.ConstraintAll([
    bh.OffNadirConstraint(max_off_nadir_deg=30.0),
    bh.LookDirectionConstraint(look_direction=bh.LookDirection.RIGHT),
])

# Compose for specific missions
ground_contact = bh.ConstraintAll([GROUND_STATION_BASIC, DAYLIGHT_OPS])
imaging_mission = bh.ConstraintAll([GROUND_STATION_BASIC, IMAGING_GEOMETRY])
```

### Constraint Validation

Test constraints in isolation before composition:

```python
# Create test scenario
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
sat_state = ...  # Test satellite state
loc_ecef = ...   # Test location

# Test individual constraints
elev_constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
print(f"Elevation constraint: {elev_constraint.evaluate(epoch, sat_state, loc_ecef)}")

time_constraint = bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0)
print(f"Time constraint: {time_constraint.evaluate(epoch, sat_state, loc_ecef)}")

# Test composition
combined = bh.ConstraintAll([elev_constraint, time_constraint])
print(f"Combined: {combined.evaluate(epoch, sat_state, loc_ecef)}")
```

## Performance Considerations

### Constraint Ordering

In `ConstraintAll`, place fast-to-fail constraints first:

```python
# Good: Cheap elevation check first
constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=10.0),  # Fast geometric check
    compute_intensive_custom_constraint(),             # Expensive computation
])

# Suboptimal: Expensive check runs even when elevation fails
constraint = bh.ConstraintAll([
    compute_intensive_custom_constraint(),             # Runs first
    bh.ElevationConstraint(min_elevation_deg=10.0),  # Would have failed anyway
])
```

**Rule of thumb**: Order by computational cost (cheapest first).

### Constraint Complexity

- **Simple constraints** (elevation, off-nadir): ~1 microsecond per evaluation
- **Time-based constraints**: ~10 microseconds (sun position calculation)
- **Custom Python constraints**: 100-1000 microseconds (Python call overhead)

For million-evaluation problems, minimize custom constraint complexity.

## See Also

- [Locations](locations.md) - Ground location types
- [Computation](computation.md) - How constraints are evaluated during access search
- [API Reference: Constraints](../../library_api/access/constraints.md)
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md)
- [Example: Computing Imaging Opportunities](../../examples/imaging_opportunities.md)
