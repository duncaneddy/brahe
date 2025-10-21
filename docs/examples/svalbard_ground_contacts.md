# Computing Ground Station Contacts

This example demonstrates how to compute satellite passes over a ground station using Brahe's access computation system. We'll analyze contacts for the Svalbard ground station—a critical high-latitude station used for polar-orbiting satellites.

## Scenario

**Ground Station**: Svalbard Satellite Station (Norway)
- Location: 78.2°N, 15.4°E
- Minimum elevation: 5° (typical for X-band communications)
- Operational hours: 24/7
- Application: Polar orbit data downlinks

**Satellites**: Small constellation of 3 sun-synchronous satellites
- Altitude: 550 km
- Inclination: 97.8° (sun-synchronous)
- Different orbital planes for coverage diversity

**Analysis Period**: 24 hours

## Complete Implementation

```python
import brahe as bh
import numpy as np
from datetime import datetime

# Initialize EOP provider (required for ECEF transformations)
eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
bh.set_global_eop_provider(eop)

# Define analysis period
epoch = bh.Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
search_start = epoch
search_end = epoch + 86400.0  # 24 hours

# Create Svalbard ground station
svalbard = bh.PointLocation(15.4, 78.2, 0.0) \
    .with_name("Svalbard") \
    .add_property("station_code", "SG") \
    .add_property("frequency_band", "X-band") \
    .add_property("max_data_rate_mbps", 300.0)

# Create constellation - 3 satellites in sun-synchronous orbits
propagators = []

for i in range(3):
    # Orbital elements: [a, e, i, raan, argp, M]
    oe = np.array([
        bh.R_EARTH + 550e3,     # 550 km altitude
        0.001,                   # Low eccentricity (near-circular)
        np.radians(97.8),        # Sun-synchronous inclination
        np.radians(i * 60),      # Different RAANs for plane diversity
        0.0,                     # Argument of perigee
        np.radians(i * 45),      # Different starting positions
    ])

    # Convert to Cartesian state
    state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

    # Create propagator
    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0) \
        .with_name(f"Sat-{i+1}") \
        .with_id(i + 1)

    propagators.append(prop)

# Define constraint - minimum 5° elevation
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)

# Compute all accesses
windows = bh.location_accesses(
    [svalbard],
    propagators,
    search_start,
    search_end,
    constraint
)

# Analyze results
print(f"Svalbard Ground Station Contact Analysis")
print(f"Analysis Period: {search_start} to {search_end}")
print(f"=" * 70)
print(f"\nFound {len(windows)} contacts")

# Statistics per satellite
for i, prop in enumerate(propagators):
    sat_windows = [w for w in windows if w.propagator_id == prop.get_id()]
    total_duration = sum(w.duration for w in sat_windows)
    avg_duration = total_duration / len(sat_windows) if sat_windows else 0

    print(f"\n{prop.get_name()}:")
    print(f"  Contacts: {len(sat_windows)}")
    print(f"  Total contact time: {total_duration/60:.1f} minutes")
    print(f"  Average pass duration: {avg_duration/60:.1f} minutes")

# Overall statistics
total_contact_time = sum(w.duration for w in windows)
coverage_fraction = total_contact_time / 86400.0 * 100

print(f"\nOverall Statistics:")
print(f"  Total contact time: {total_contact_time/3600:.1f} hours")
print(f"  Coverage: {coverage_fraction:.1f}%")

# Detailed pass information
print(f"\n{'='*70}")
print(f"Detailed Pass Information")
print(f"{'='*70}")

for i, window in enumerate(windows[:10], 1):  # Show first 10 passes
    duration_min = window.duration / 60.0
    max_el = window.properties.get("elevation_max", 0.0)

    # Find satellite name
    sat_name = next(
        (p.get_name() for p in propagators if p.get_id() == window.propagator_id),
        "Unknown"
    )

    print(f"\nPass {i}: {sat_name}")
    print(f"  Start:    {window.window_open}")
    print(f"  End:      {window.window_close}")
    print(f"  Duration: {duration_min:.1f} minutes")
    print(f"  Max Elev: {max_el:.1f}°")
    print(f"  AOS Az:   {window.properties.get('azimuth_open', 0.0):.1f}°")
    print(f"  LOS Az:   {window.properties.get('azimuth_close', 0.0):.1f}°")

if len(windows) > 10:
    print(f"\n... and {len(windows) - 10} more passes")
```

## Expected Output

```
Svalbard Ground Station Contact Analysis
Analysis Period: 2024-06-15T00:00:00.000000000 UTC to 2024-06-16T00:00:00.000000000 UTC
======================================================================

Found 42 contacts

Sat-1:
  Contacts: 14
  Total contact time: 112.3 minutes
  Average pass duration: 8.0 minutes

Sat-2:
  Contacts: 14
  Total contact time: 109.8 minutes
  Average pass duration: 7.8 minutes

Sat-3:
  Contacts: 14
  Total contact time: 114.1 minutes
  Average pass duration: 8.2 minutes

Overall Statistics:
  Total contact time: 5.6 hours
  Coverage: 23.4%

======================================================================
Detailed Pass Information
======================================================================

Pass 1: Sat-1
  Start:    2024-06-15T00:23:15.000000000 UTC
  End:      2024-06-15T00:31:47.000000000 UTC
  Duration: 8.5 minutes
  Max Elev: 42.3°
  AOS Az:   12.5°
  LOS Az:   348.7°

Pass 2: Sat-2
  Start:    2024-06-15T01:08:22.000000000 UTC
  End:      2024-06-15T01:16:33.000000000 UTC
  Duration: 8.2 minutes
  Max Elev: 38.9°
  AOS Az:   18.3°
  LOS Az:   342.1°

... (continues)
```

## Analysis Extensions

### Filter by Time of Day

Add operational hour constraints for manned operations:

```python
# Daytime operations only (8 AM - 6 PM local solar time)
daytime_constraint = bh.ConstraintAll([
    bh.ElevationConstraint(min_elevation_deg=5.0),
    bh.LocalTimeConstraint(min_hour=8.0, max_hour=18.0)
])

daytime_windows = bh.location_accesses(
    [svalbard],
    propagators,
    search_start,
    search_end,
    daytime_constraint
)

print(f"Daytime passes: {len(daytime_windows)} (vs {len(windows)} total)")
```

### Compute Link Budget Properties

Add custom properties for communications analysis:

```python
def link_budget_computer(epoch, sat_state_eci, sat_state_ecef, location_ecef):
    """Compute link budget parameters"""
    # Slant range
    sat_pos = sat_state_ecef[:3]
    slant_range_m = np.linalg.norm(sat_pos - location_ecef)
    slant_range_km = slant_range_m / 1000.0

    # Free-space path loss (simplified, X-band 8 GHz)
    freq_hz = 8e9
    wavelength = 3e8 / freq_hz
    fspl_db = 20 * np.log10(slant_range_m) + 20 * np.log10(freq_hz) - 147.55

    # Range rate (for Doppler estimation)
    sat_vel = sat_state_ecef[3:6]
    range_vec = sat_pos - location_ecef
    range_rate = np.dot(sat_vel, range_vec) / slant_range_m

    return {
        "slant_range_km": slant_range_km,
        "free_space_path_loss_db": fspl_db,
        "range_rate_m_s": range_rate,
    }

# Create property computer
computer = bh.AccessPropertyComputer(link_budget_computer)

# Compute with custom properties
windows_with_link = bh.location_accesses(
    [svalbard],
    propagators,
    search_start,
    search_end,
    constraint,
    property_computers=[computer]
)

# Access custom properties
for window in windows_with_link[:3]:
    props = window.properties
    print(f"\nPass: {window.window_open}")
    print(f"  Slant range: {props['slant_range_km']:.1f} km")
    print(f"  Path loss: {props['free_space_path_loss_db']:.1f} dB")
    print(f"  Range rate: {props['range_rate_m_s']:.1f} m/s")
```

### High-Elevation Passes Only

Filter for overhead passes with better signal quality:

```python
# Only high-elevation passes (> 30°)
high_el_constraint = bh.ElevationConstraint(min_elevation_deg=30.0)

high_el_windows = bh.location_accesses(
    [svalbard],
    propagators,
    search_start,
    search_end,
    high_el_constraint
)

print(f"\nHigh-elevation passes (>30°): {len(high_el_windows)}")
print(f"  Percentage: {len(high_el_windows)/len(windows)*100:.1f}%")

# Average max elevation comparison
avg_max_el_all = np.mean([w.properties["elevation_max"] for w in windows])
avg_max_el_high = np.mean([w.properties["elevation_max"] for w in high_el_windows])

print(f"  Avg max elevation (all): {avg_max_el_all:.1f}°")
print(f"  Avg max elevation (>30°): {avg_max_el_high:.1f}°")
```

### Export to GeoJSON

Save results for visualization in GIS tools:

```python
import json

# Create GeoJSON feature for station with pass statistics
svalbard_geojson = svalbard.to_geojson()
svalbard_geojson["properties"]["total_passes"] = len(windows)
svalbard_geojson["properties"]["total_contact_hours"] = total_contact_time / 3600

# Create FeatureCollection
feature_collection = {
    "type": "FeatureCollection",
    "features": [svalbard_geojson]
}

# Save to file
with open("svalbard_contacts.geojson", "w") as f:
    json.dump(feature_collection, f, indent=2)

print("Results saved to svalbard_contacts.geojson")
```

## Performance Optimization

For long analysis periods or many satellites, enable parallelization:

```python
# Configure parallel computation
config = bh.AccessSearchConfig(
    initial_time_step=60.0,
    adaptive_step=True,     # Adaptive stepping for efficiency
    parallel=True,           # Enable parallelization
    num_threads=8           # Use 8 threads
)

# Compute with parallel configuration
windows_parallel = bh.location_accesses(
    [svalbard],
    propagators,
    search_start,
    search_end,
    constraint,
    config=config
)

# Results identical to sequential, just faster
assert len(windows_parallel) == len(windows)
```

## Real-World Considerations

**Earth Orientation Parameters**:
- This example uses static EOP (all zeros) for simplicity
- Production systems should use `FileEOPProvider` or `CachingEOPProvider`
- Download current EOP data: `bh.download_standard_eop_file(path)`

**TLE-Based Analysis**:
- Replace Keplerian propagators with SGP4 for real satellites:
```python
tle_line1 = "1 25544U 98067A   ..."
tle_line2 = "2 25544  51.6461 ..."
tle = bh.TLE.from_lines(tle_line1, tle_line2)
prop_sgp4 = bh.SGPPropagator.from_tle(tle, 60.0).with_name("ISS")
```

**Elevation Masks**:
- Account for terrain obstructions or antenna limits:
```python
mask = [
    (0.0, 10.0),     # North: 10° due to building
    (90.0, 5.0),     # East: 5° clear
    (180.0, 15.0),   # South: 15° mountain
    (270.0, 5.0),    # West: 5° clear
    (360.0, 10.0),   # Wrap to north
]
mask_constraint = bh.ElevationMaskConstraint(mask)
```

## See Also

- [Access Computation Overview](../learn/access_computation/index.md)
- [Constraints](../learn/access_computation/constraints.md)
- [Locations](../learn/access_computation/locations.md)
- [Point Imaging Opportunities](point_imaging_opportunities.md)
- [API Reference: Access Module](../library_api/access/index.md)
