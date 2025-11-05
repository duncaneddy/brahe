# Ground Tracks

Ground track plotting visualizes the path a satellite traces over Earth's surface. This is essential for mission planning, coverage analysis, and understanding when and where a satellite can communicate with ground stations. Brahe's `plot_groundtrack` function renders satellite trajectories on a world map with optional ground station markers and communication coverage cones.

See also: [plot_groundtrack API Reference](../../library_api/plots/ground_tracks.md)

## Interactive Ground Track (Plotly)

The plotly backend creates interactive maps that you can pan, zoom, and explore. Hover over the satellite track to see precise coordinates.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/groundtrack_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/groundtrack_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="groundtrack_plotly.py"
    --8<-- "./plots/learn/plots/groundtrack_plotly.py"
    ```

This example shows:

- **ISS ground track** over one orbital period (red line)
- **Cape Canaveral ground station** (blue marker)
- **Communication cone** showing the region where the ISS is visible above 10° elevation

The interactive plot allows you to:

- Zoom into specific regions
- Pan across the map
- Hover to see exact coordinates
- Toggle layers on/off

## Static Ground Track (Matplotlib)

The matplotlib backend produces publication-ready static figures ideal for reports and papers.

<figure markdown="span">
    ![Ground Track Plot](../../figures/plot_groundtrack_matplotlib_light.svg#only-light)
    ![Ground Track Plot](../../figures/plot_groundtrack_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="groundtrack_matplotlib.py"
    --8<-- "./plots/learn/plots/groundtrack_matplotlib.py"
    ```

The static plot shows the same information in a clean, professional format suitable for:

- Academic publications
- Technical reports
- Batch figure generation
- Custom post-processing with matplotlib

## Key Features

### Multiple Trajectories

Plot multiple satellites simultaneously with different colors:

```python
import brahe as bh

# Create multiple propagators
prop1 = bh.KeplerianPropagator.from_eci(epoch1, state1, 60.0)
prop2 = bh.KeplerianPropagator.from_eci(epoch2, state2, 60.0)

# Propagate and get trajectories
prop1.propagate_to(epoch1 + duration)
prop2.propagate_to(epoch2 + duration)

# Plot both
fig = bh.plot_groundtrack(
    trajectories=[
        {"trajectory": prop1.trajectory, "color": "red", "line_width": 2},
        {"trajectory": prop2.trajectory, "color": "blue", "line_width": 2}
    ],
    backend="plotly"
)
```

### Ground Station Networks

Visualize multiple ground stations with different groups and colors:

```python
import numpy as np

# AWS ground stations
aws_stations = [
    bh.PointLocation(np.radians(40.7128), np.radians(-74.0060), 0.0),  # NYC
    bh.PointLocation(np.radians(37.7749), np.radians(-122.4194), 0.0),  # SF
]

# KSAT ground stations
ksat_stations = [
    bh.PointLocation(np.radians(78.2232), np.radians(15.6267), 0.0),  # Svalbard
]

fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj}],
    ground_stations=[
        {"stations": aws_stations, "color": "orange", "alpha": 0.3},
        {"stations": ksat_stations, "color": "blue", "alpha": 0.3}
    ],
    backend="matplotlib"
)
```

### Coverage Zones

Add polygon zones for restricted areas, target regions, or sensor footprints:

```python
# Define a restricted zone
vertices = [
    (np.radians(30.0), np.radians(-100.0)),  # lat, lon
    (np.radians(35.0), np.radians(-100.0)),
    (np.radians(35.0), np.radians(-95.0)),
    (np.radians(30.0), np.radians(-95.0))
]
zone = bh.PolygonLocation(vertices)

fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj}],
    zones=[{
        "zone": zone,
        "fill": True,
        "fill_color": "red",
        "fill_alpha": 0.2,
        "edge": True,
        "edge_color": "red"
    }]
)
```

### Track Filtering

Control how much of the trajectory is displayed:

```python
# Show only the last 2 orbits
fig = bh.plot_groundtrack(
    trajectories=[{
        "trajectory": traj,
        "track_length": 2,
        "track_units": "orbits"  # or "seconds"
    }]
)
```

### Map Customization

Control map appearance and region:

```python
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj}],
    basemap="natural_earth",  # "stock" or None
    show_borders=True,
    show_coastlines=True,
    show_grid=True,
    extent=[-180, -60, 20, 50],  # [lon_min, lon_max, lat_min, lat_max]
    backend="matplotlib"
)
```

## Common Use Cases

**Mission Planning**: Identify ground contact opportunities and coverage gaps

```python
# Visualize 24 hours of coverage for a LEO constellation
for sat in constellation:
    sat.propagate_to(epoch + 24*3600)

fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": s.trajectory} for s in constellation],
    ground_stations=[{"stations": network_stations}],
    gs_min_elevation=15.0
)
```

**Coverage Analysis**: Determine when targets are accessible

```python
# Show which ground stations can see the satellite
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj}],
    ground_stations=[{"stations": all_stations, "alpha": 0.2}],
    gs_cone_altitude=altitude,  # Satellite altitude for cone calculation
    gs_min_elevation=10.0  # Minimum elevation angle
)
```

**Orbit Visualization**: Understand satellite behavior

```python
# Compare different orbit types
fig = bh.plot_groundtrack(
    trajectories=[
        {"trajectory": leo_traj, "color": "red", "label": "LEO"},
        {"trajectory": meo_traj, "color": "blue", "label": "MEO"},
        {"trajectory": geo_traj, "color": "green", "label": "GEO"}
    ]
)
```

## Tips

- Use `backend="plotly"` for interactive exploration and presentations
- Use `backend="matplotlib"` with `scienceplots` for publication-quality figures
- Set `gs_cone_altitude` to your satellite's altitude for accurate coverage visualization
- Adjust `gs_min_elevation` based on antenna pointing constraints (typically 5-15°)
- Use `extent` parameter to zoom into specific regions of interest

## See Also

- [plot_groundtrack API Reference](../../library_api/plots/ground_tracks.md)
- [Access Geometry](access_geometry.md) - Detailed visibility analysis
- [PointLocation](../../library_api/access/locations.md) - Ground station definitions
- [PolygonLocation](../../library_api/access/locations.md) - Zone definitions
