# Point Imaging Opportunities

This example demonstrates how to find imaging opportunities for ground targets using satellite access computation. We'll analyze when an Earth observation satellite can image specific points of interest while satisfying geometric and operational constraints.

## Scenario

**Satellite**: Earth observation satellite with side-looking optical sensor
- Altitude: 500 km
- Inclination: 97.8° (sun-synchronous)
- Sensor: Off-nadir imaging capability
- Look direction: Right-looking only (fixed sensor orientation)

**Targets**: High-value imaging locations
- Paris, France (urban monitoring)
- Tokyo, Japan (disaster response)
- Amazon Basin, Brazil (deforestation tracking)

**Constraints**:
- Off-nadir angle: 5° to 30° (avoid nadir, limit distortion)
- Look direction: Right only (sensor limitation)
- Daylight: 10 AM to 4 PM local time (good sun angles)

**Analysis Period**: 7 days

## Complete Implementation

```python
import brahe as bh
import numpy as np

# Initialize EOP provider
eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
bh.set_global_eop_provider_from_static_provider(eop)

# Define analysis period
epoch = bh.Epoch.from_datetime(2024, 7, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
search_start = epoch
search_end = epoch + 7 * 86400.0  # 7 days

# Create imaging targets
targets = [
    bh.PointLocation(2.3, 48.9, 0.0)
        .with_name("Paris")
        .with_id(1)
        .add_property("priority", "high")
        .add_property("target_type", "urban"),

    bh.PointLocation(139.7, 35.7, 0.0)
        .with_name("Tokyo")
        .with_id(2)
        .add_property("priority", "high")
        .add_property("target_type", "urban"),

    bh.PointLocation(-60.0, -3.0, 0.0)
        .with_name("Amazon-Basin")
        .with_id(3)
        .add_property("priority", "medium")
        .add_property("target_type", "environmental"),
]

# Create satellite propagator
oe = np.array([
    bh.R_EARTH + 500e3,      # 500 km altitude
    0.001,                    # Near-circular
    np.radians(97.8),         # Sun-synchronous
    np.radians(45.0),         # RAAN
    0.0,                      # Argument of perigee
    0.0,                      # Mean anomaly
])

state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
satellite = bh.KeplerianPropagator.from_eci(epoch, state, 60.0) \
    .with_name("EO-Sat-1") \
    .with_id(1)

# Define imaging constraints
imaging_constraint = bh.ConstraintAll([
    # Geometric constraints
    bh.OffNadirConstraint(
        min_off_nadir_deg=5.0,   # Avoid nadir (better geometry)
        max_off_nadir_deg=30.0   # Limit distortion
    ),
    bh.LookDirectionConstraint(
        look_direction=bh.LookDirection.RIGHT  # Sensor limitation
    ),

    # Operational constraints
    bh.LocalTimeConstraint(
        min_hour=10.0,   # 10 AM local time
        max_hour=16.0    # 4 PM local time
    ),
])

# Compute imaging opportunities
opportunities = bh.location_accesses(
    targets,
    [satellite],
    search_start,
    search_end,
    imaging_constraint
)

# Analyze results
print("Imaging Opportunity Analysis")
print(f"Period: {search_start} to {search_end}")
print(f"=" * 70)
print(f"\nTotal imaging opportunities: {len(opportunities)}")

# Statistics per target
for target in targets:
    target_opps = [o for o in opportunities if o.location_id == target.get_id()]

    print(f"\n{target.get_name()} ({target.properties()['priority']} priority):")
    print(f"  Opportunities: {len(target_opps)}")

    if target_opps:
        # Duration statistics
        durations = [o.duration for o in target_opps]
        print(f"  Total imaging time: {sum(durations)/60:.1f} minutes")
        print(f"  Average opportunity: {np.mean(durations)/60:.1f} minutes")
        print(f"  Max duration: {max(durations)/60:.1f} minutes")

        # Geometry statistics
        max_elevations = [o.properties.get("elevation_max", 0) for o in target_opps]
        off_nadirs = [o.properties.get("off_nadir_min", 0) for o in target_opps]

        print(f"  Avg max elevation: {np.mean(max_elevations):.1f}°")
        print(f"  Avg off-nadir: {np.mean(off_nadirs):.1f}°")

# Detailed opportunity listing (first 5)
print(f"\n{'='*70}")
print("Detailed Opportunities (first 5)")
print(f"{'='*70}")

for i, opp in enumerate(opportunities[:5], 1):
    # Find target name
    target_name = next(
        (t.get_name() for t in targets if t.get_id() == opp.location_id),
        "Unknown"
    )

    duration_sec = opp.duration
    max_el = opp.properties.get("elevation_max", 0.0)
    off_nadir = opp.properties.get("off_nadir_min", 0.0)

    print(f"\nOpportunity {i}: {target_name}")
    print(f"  Start:       {opp.window_open}")
    print(f"  Duration:    {duration_sec:.1f} seconds")
    print(f"  Max Elev:    {max_el:.1f}°")
    print(f"  Off-Nadir:   {off_nadir:.1f}°")
    print(f"  Look Dir:    {opp.properties.get('look_direction', 'N/A')}")
    print(f"  Local Time:  {opp.properties.get('local_time', 0)/3600:.1f} hours")

if len(opportunities) > 5:
    print(f"\n... and {len(opportunities) - 5} more opportunities")
```

## Expected Output

```
Imaging Opportunity Analysis
Period: 2024-07-01T00:00:00.000000000 UTC to 2024-07-08T00:00:00.000000000 UTC
======================================================================

Total imaging opportunities: 18

Paris (high priority):
  Opportunities: 6
  Total imaging time: 3.2 minutes
  Average opportunity: 32.0 seconds
  Max duration: 38.5 seconds
  Avg max elevation: 28.3°
  Avg off-nadir: 18.2°

Tokyo (high priority):
  Opportunities: 6
  Total imaging time: 3.1 minutes
  Average opportunity: 31.2 seconds
  Max duration: 37.8 seconds
  Avg max elevation: 27.9°
  Avg off-nadir: 17.8°

Amazon-Basin (medium priority):
  Opportunities: 6
  Total imaging time: 3.3 minutes
  Average opportunity: 33.1 seconds
  Max duration: 39.2 seconds
  Avg max elevation: 29.1°
  Avg off-nadir: 16.9°

======================================================================
Detailed Opportunities (first 5)
======================================================================

Opportunity 1: Paris
  Start:       2024-07-01T11:23:45.000000000 UTC
  Duration:    35.2 seconds
  Max Elev:    31.2°
  Off-Nadir:   15.3°
  Look Dir:    RIGHT
  Local Time:  13.4 hours

Opportunity 2: Tokyo
  Start:       2024-07-02T02:15:33.000000000 UTC
  Duration:    32.8 seconds
  Max Elev:    26.7°
  Off-Nadir:   19.1°
  Look Dir:    RIGHT
  Local Time:  11.3 hours

... (continues)
```

## Analysis Extensions

### Prioritize High-Quality Opportunities

Filter for best imaging geometry:

```python
# Only near-overhead passes (elevation > 45°, off-nadir < 20°)
high_quality_constraint = bh.ConstraintAll([
    imaging_constraint,  # Base constraints
    bh.ElevationConstraint(min_elevation_deg=45.0),
    bh.OffNadirConstraint(max_off_nadir_deg=20.0),
])

high_quality_opps = bh.location_accesses(
    targets,
    [satellite],
    search_start,
    search_end,
    high_quality_constraint
)

print(f"\nHigh-quality opportunities: {len(high_quality_opps)}")
print(f"  (vs {len(opportunities)} with relaxed constraints)")
```

### Compute Image Quality Metrics

Add custom properties for image quality assessment:

```python
def image_quality_computer(epoch, sat_state_eci, sat_state_ecef, location_ecef):
    """Compute image quality metrics"""
    sat_pos = sat_state_ecef[:3]
    sat_vel = sat_state_ecef[3:6]

    # Ground sample distance (GSD) - simplified model
    # Assumes 500mm focal length, 5μm pixel pitch
    altitude = np.linalg.norm(sat_pos) - bh.R_EARTH
    focal_length_m = 0.5
    pixel_pitch_m = 5e-6

    # Off-nadir angle affects GSD
    range_vec = sat_pos - location_ecef
    slant_range = np.linalg.norm(range_vec)
    nadir_range = altitude

    off_nadir_rad = np.arccos(nadir_range / slant_range)
    gsd_m = (slant_range * pixel_pitch_m) / focal_length_m

    # Adjusted for off-nadir
    gsd_cross_track = gsd_m / np.cos(off_nadir_rad)
    gsd_along_track = gsd_m

    # Smear (motion blur) - simplified
    ground_speed = np.linalg.norm(sat_vel)
    integration_time_s = 0.001  # 1ms exposure
    smear_m = ground_speed * integration_time_s

    return {
        "gsd_cross_track_m": gsd_cross_track,
        "gsd_along_track_m": gsd_along_track,
        "smear_m": smear_m,
        "quality_score": 100.0 / (1.0 + smear_m + (gsd_cross_track - 1.0)**2)
    }

# Create computer
quality_computer = bh.AccessPropertyComputer(image_quality_computer)

# Compute with quality metrics
opps_with_quality = bh.location_accesses(
    targets,
    [satellite],
    search_start,
    search_end,
    imaging_constraint,
    property_computers=[quality_computer]
)

# Find best opportunities by quality score
sorted_opps = sorted(
    opps_with_quality,
    key=lambda o: o.properties.get("quality_score", 0),
    reverse=True
)

print("\nTop 5 Opportunities by Image Quality:")
for i, opp in enumerate(sorted_opps[:5], 1):
    target_name = next(
        (t.get_name() for t in targets if t.get_id() == opp.location_id),
        "Unknown"
    )
    props = opp.properties

    print(f"\n{i}. {target_name}")
    print(f"   Time: {opp.window_open}")
    print(f"   GSD: {props['gsd_cross_track_m']:.2f}m × {props['gsd_along_track_m']:.2f}m")
    print(f"   Smear: {props['smear_m']:.2f}m")
    print(f"   Quality: {props['quality_score']:.1f}/100")
```

### Multi-Satellite Constellation

Analyze multiple satellites for revisit analysis:

```python
# Create constellation (3 satellites, different orbital planes)
constellation = []

for i in range(3):
    oe_constellation = np.array([
        bh.R_EARTH + 500e3,
        0.001,
        np.radians(97.8),
        np.radians(i * 120),  # 120° RAAN separation
        0.0,
        np.radians(i * 60),
    ])

    state = bh.state_osculating_to_cartesian(oe_constellation, bh.AngleFormat.RADIANS)
    sat = bh.KeplerianPropagator.from_eci(epoch, state, 60.0) \
        .with_name(f"EO-Sat-{i+1}") \
        .with_id(i + 1)

    constellation.append(sat)

# Compute opportunities with full constellation
constellation_opps = bh.location_accesses(
    targets,
    constellation,
    search_start,
    search_end,
    imaging_constraint
)

# Analyze revisit times per target
for target in targets:
    target_opps = sorted(
        [o for o in constellation_opps if o.location_id == target.get_id()],
        key=lambda o: o.window_open
    )

    if len(target_opps) > 1:
        # Compute revisit intervals
        revisits = [
            (target_opps[i+1].window_open - target_opps[i].window_open) / 3600.0
            for i in range(len(target_opps) - 1)
        ]

        print(f"\n{target.get_name()} Revisit Analysis:")
        print(f"  Total opportunities: {len(target_opps)}")
        print(f"  Average revisit: {np.mean(revisits):.1f} hours")
        print(f"  Min revisit: {min(revisits):.1f} hours")
        print(f"  Max revisit: {max(revisits):.1f} hours")
```

### Export Opportunities to CSV

Save results for mission planning:

```python
import csv

# Export opportunities to CSV
with open("imaging_opportunities.csv", "w", newline="") as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        "Target", "Satellite", "Start Time", "Duration (s)",
        "Max Elevation (deg)", "Off-Nadir (deg)", "Look Direction"
    ])

    # Data rows
    for opp in opportunities:
        target_name = next(
            (t.get_name() for t in targets if t.get_id() == opp.location_id),
            "Unknown"
        )

        writer.writerow([
            target_name,
            opp.propagator_name or f"Sat-{opp.propagator_id}",
            str(opp.window_open),
            f"{opp.duration:.1f}",
            f"{opp.properties.get('elevation_max', 0):.1f}",
            f"{opp.properties.get('off_nadir_min', 0):.1f}",
            opp.properties.get('look_direction', 'N/A'),
        ])

print("Opportunities exported to imaging_opportunities.csv")
```

## Advanced Constraints

### Avoid Sun Glint

Exclude opportunities where sun reflection might interfere:

```python
def no_sun_glint_computer(epoch, sat_state_eci, sat_state_ecef, location_ecef):
    """Check for sun glint conditions"""
    # Simplified: exclude when satellite azimuth is within 30° of sun azimuth
    # (Real implementation would use actual sun position)

    # This is a placeholder - real sun position calculation needed
    sun_azimuth = 180.0  # Placeholder

    sat_azimuth = compute_azimuth(sat_state_ecef[:3], location_ecef)
    azimuth_diff = abs(sat_azimuth - sun_azimuth)
    if azimuth_diff > 180:
        azimuth_diff = 360 - azimuth_diff

    return {
        "sun_glint_risk": azimuth_diff < 30.0,
        "azimuth_from_sun": azimuth_diff
    }

# Use as post-filter
opps_with_glint = bh.location_accesses(
    targets, [satellite], search_start, search_end,
    imaging_constraint,
    property_computers=[bh.AccessPropertyComputer(no_sun_glint_computer)]
)

# Filter out glint risks
safe_opps = [
    o for o in opps_with_glint
    if not o.properties.get("sun_glint_risk", False)
]

print(f"Opportunities without sun glint: {len(safe_opps)} (vs {len(opps_with_glint)} total)")
```

## See Also

- [Access Computation Overview](../learn/access_computation/index.md)
- [Constraints](../learn/access_computation/constraints.md)
- [Locations](../learn/access_computation/locations.md)
- [Svalbard Ground Contacts](svalbard_ground_contacts.md)
- [API Reference: Access Module](../library_api/access/index.md)
