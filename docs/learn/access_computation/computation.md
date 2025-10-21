# Access Computation

This document explains how Brahe computes access windows—the algorithms, design decisions, and performance considerations that make access computation both accurate and efficient.

## Overview

Access computation involves finding time periods when a satellite can observe or communicate with a ground location while satisfying specified constraints. This is fundamentally a root-finding problem: we need to identify when constraint satisfaction transitions from `False` to `True` (window opens) and from `True` to `False` (window closes).

## Algorithm Overview

Brahe uses a two-phase approach:

1. **Coarse grid search**: Quickly identify candidate access periods using large time steps
2. **Bisection refinement**: Precisely locate window boundaries using binary search

This hybrid approach balances speed (coarse search) with precision (refinement).

## Phase 1: Coarse Grid Search

### Basic Algorithm

The coarse search evaluates constraints at regular time intervals to identify candidate access periods.

### Adaptive Time Stepping

For improved efficiency, Brahe optionally adapts the time step based on orbital period:

```python
# Configuration
config = bh.AccessSearchConfig(
    initial_time_step=60.0,
    adaptive_step=True,
    adaptive_fraction=0.75
)
```

**How it works**:
1. Compute orbital period from semi-major axis
2. Set step size = `orbital_period * adaptive_fraction`
3. Use this step for first step immediately after finding a candidate window because it's unlikely for LEO to have another access until the next orbit.

**When to use**:
- When computing accesses for LEO satellites
- Long search periods (days to weeks)
- Computational efficiency is critical

**When to avoid**:
- Non-LEO orbits (GEO, HEO) where orbital period is very long

## Phase 2: Bisection Refinement

Once candidates are identified, bisection search refines boundaries to high precision.

### Bisection Algorithm

### Window Opening vs. Closing

Bisection refines both boundaries:

1. **Window open**: Refine between last `False` and first `True` from coarse search
2. **Window close**: Refine between last `True` and first `False` from coarse search

Each window requires two bisection searches (open and close boundaries).

## Phase 3: Property Computation

After window boundaries are refined, properties are computed to characterize the access.

### Core Properties

Computed automatically for every window:

```python
# At window open
azimuth_open = compute_azimuth(sat_pos, location)

# At window close
azimuth_close = compute_azimuth(sat_pos, location)

# Throughout window (via sampling)
elevation_max = max(compute_elevation(sat_pos, location) for t in window)
elevation_min = min(compute_elevation(sat_pos, location) for t in window)

# At window midtime
off_nadir_min = compute_off_nadir(sat_pos, sat_vel, location)
local_time = compute_local_solar_time(location, epoch)
look_direction = compute_look_direction(sat_vel, location_vector)
asc_dsc = compute_ascending_or_descending(sat_state)
```
### Custom Properties

User-defined property computers can compute additional window properties by subclassing `AccessPropertyComputer`:

```python
import brahe as bh
import numpy as np

class SlantRangeComputer(bh.AccessPropertyComputer):
    """
    Computes slant range and related metrics for each access window.
    """

    def compute(self, window, satellite_state_ecef, location_ecef):
        """
        Compute slant range properties.

        Args:
            window: AccessWindow with timing information
            satellite_state_ecef: Satellite state [x,y,z,vx,vy,vz] at window midtime (m, m/s)
            location_ecef: Location position [x,y,z] (m)

        Returns:
            dict: Property name -> value mapping
        """
        sat_pos = satellite_state_ecef[:3]
        slant_range_m = np.linalg.norm(sat_pos - location_ecef)

        return {
            "slant_range_km": slant_range_m / 1000.0,
            "within_2000km": slant_range_m < 2000e3,
            "slant_range_category": self._categorize_range(slant_range_m)
        }

    def property_names(self):
        """Return list of property names this computer produces"""
        return ["slant_range_km", "within_2000km", "slant_range_category"]

    def _categorize_range(self, range_m):
        """Helper to categorize range"""
        if range_m < 1000e3:
            return "close"
        elif range_m < 2000e3:
            return "medium"
        else:
            return "far"

# Use with access computation
computer = SlantRangeComputer()
windows = bh.location_accesses(
    locations, propagators, start, end,
    constraint,
    property_computers=[computer]
)

# Access custom properties
for window in windows:
    slant_range = window.properties.additional["slant_range_km"]
    category = window.properties.additional["slant_range_category"]
    print(f"Range: {slant_range:.1f} km ({category})")
```

### Advanced Property Computer

```python
class DopplerComputer(bh.AccessPropertyComputer):
    """
    Computes Doppler shift at window midtime.

    Useful for communications link budget analysis.
    """

    def __init__(self, frequency_hz=2.4e9):
        """
        Args:
            frequency_hz: Carrier frequency in Hz (default: 2.4 GHz)
        """
        self.frequency = frequency_hz
        self.c = 299792458.0  # Speed of light (m/s)

    def compute(self, window, satellite_state_ecef, location_ecef):
        """Compute Doppler shift"""
        # Extract satellite velocity
        sat_vel = satellite_state_ecef[3:6]

        # Line-of-sight vector
        sat_pos = satellite_state_ecef[:3]
        los = location_ecef - sat_pos
        los_unit = los / np.linalg.norm(los)

        # Radial velocity (positive = approaching)
        radial_vel = np.dot(sat_vel, los_unit)

        # Doppler shift
        doppler_hz = -(radial_vel / self.c) * self.frequency

        return {
            "doppler_shift_hz": doppler_hz,
            "doppler_shift_khz": doppler_hz / 1000.0,
            "radial_velocity_mps": radial_vel,
            "is_approaching": radial_vel > 0.0
        }

    def property_names(self):
        return [
            "doppler_shift_hz",
            "doppler_shift_khz",
            "radial_velocity_mps",
            "is_approaching"
        ]
```

### Time Series Properties

Property computers can return time series data:

```python
class ElevationProfileComputer(bh.AccessPropertyComputer):
    """
    Computes elevation angle profile throughout the window.
    """

    def __init__(self, sample_rate_sec=10.0):
        self.sample_rate = sample_rate_sec

    def compute(self, window, satellite_state_ecef, location_ecef):
        """
        Sample elevation throughout window.

        Note: This is called at window midtime, so for time series
        you would need to resample the trajectory or use the StateProvider.
        This is a simplified example.
        """
        # In real implementation, would resample throughout window
        # For now, just return midtime elevation
        from brahe.access import compute_elevation

        midtime_elevation = compute_elevation(
            satellite_state_ecef[:3],
            location_ecef
        )

        return {
            "elevation_profile": {
                "times": [0.0],  # Seconds from window start
                "values": [midtime_elevation]
            },
            "peak_elevation": midtime_elevation
        }

    def property_names(self):
        return ["elevation_profile", "peak_elevation"]
```

**When to use property computers**:
- Computing derived metrics for link budgets or mission analysis
- Recording time-series data for later analysis
- Annotating windows with mission-specific information
- Filtering windows based on computed properties

**Performance note**: Property computers are called once per window at the midtime. For expensive computations, this is more efficient than computing properties for every constraint evaluation.

## Complete Pipeline

The full access computation pipeline:

```
1. For each (location, propagator) pair:

   2. Coarse grid search
      ↓
      [Candidate windows with ~60s boundary uncertainty]

   3. For each candidate:

      4. Bisection refinement (window open)
         ↓
      5. Bisection refinement (window close)
         ↓
         [Precise window boundaries within 0.01s]

      6. Compute core properties
         ↓
      7. Compute custom properties (if any)
         ↓
         [Complete AccessWindow]

8. Sort all windows by opening time
   ↓
   [Final sorted list of AccessWindow objects]
```

## Parallelization

For large-scale problems, Brahe parallelizes access computation across location-propagator pairs.

Access computation is parallel by default, utilizing up to 90% of available CPU cores on the machine. This can
be disabled or configured as needed.

**Parallelization occurs at the pair level**:
- Each (location, propagator) pair is independent
- No shared state during window finding
- Results aggregated and sorted after all pairs complete

**Performance scaling**:
- Near-linear speedup for many pairs (e.g., 10 locations × 10 satellites = 100 pairs)
- Limited benefit for single pair (no parallelism)

### Configuration

```python
# Default: parallel with 90% of cores
windows = bh.location_accesses(locations, propagators, start, end, constraint)

# Explicit control
config = bh.AccessSearchConfig(
    parallel=True,        # Enable parallelization
    num_threads=4         # Use 4 threads (overrides global default)
)

windows = bh.location_accesses(
    locations, propagators, start, end, constraint, config=config
)

# Sequential (debugging or reproducibility)
config = bh.AccessSearchConfig(parallel=False)
```

**Thread pool management**:
```python
# Set global default (must be called before any parallel operations)
bh.set_max_threads(8)

# Query current setting
num_threads = bh.get_max_threads()
```

### Optimization Guidelines

**Choose appropriate time step**:
- Smaller step: more accurate, slower
- Larger step: faster, may miss short windows
- Rule of thumb: `dt ≤ min_expected_window_duration / 3`

**Use adaptive stepping when**:
- Search period >> orbital period
- LEO satellites

**Use parallelization when**:
- Multiple location-propagator pairs (N × M > 10)
- Multiple CPU cores available

**Avoid parallelization when**:
- Single location-propagator pair
- Custom constraints use external resources (databases, files)

## Accuracy Considerations

### Numerical Precision

**Bisection tolerance**:
- Default 0.01s provides sub-second precision
- Can be tightened to 0.001s or more for high-precision applications
- Diminishing returns below ~0.001s due to floating-point limits and state propagation errors

## Implementation Notes

### State Provider Architecture

Access computation works with any `StateProvider`:

```rust
pub trait StateProvider {
    fn state(&self, epoch: &Epoch) -> Vector6<f64>;
    fn state_eci(&self, epoch: &Epoch) -> Vector6<f64>;
    fn state_ecef(&self, epoch: &Epoch) -> Vector6<f64>;
}
```

This abstraction allows:
- Analytical propagators (Keplerian, SGP4)
- Pre-computed trajectories (OrbitTrajectory)
- Hybrid approaches (mix propagator types)
- Future propagators (numerical integrators, etc.)

All use the same access computation code—no special-casing required.

## See Also

- [Locations](locations.md) - Ground location types and properties
- [Constraints](constraints.md) - Constraint system and composition
- [Access Computation Index](index.md) - Overview and usage examples
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md) - Complete workflow
- [API Reference: Access Module](../../library_api/access/index.md) - Complete API documentation
