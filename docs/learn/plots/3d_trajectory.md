# 3D Trajectory Visualization

Three-dimensional trajectory plots display orbital paths in Earth-Centered Inertial (ECI) coordinates, providing intuitive spatial understanding of satellite motion. The `plot_trajectory_3d` function renders trajectories with optional Earth sphere visualization, camera controls, and support for multiple orbits with different colors and labels.

See also: [plot_trajectory_3d API Reference](../../library_api/plots/trajectory_3d.md)

## Interactive 3D Trajectory (Plotly)

The plotly backend creates fully interactive 3D plots. Click and drag to rotate, scroll to zoom, and double-click to reset the view.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/plot_trajectory_3d_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/plot_trajectory_3d_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="trajectory_3d_plotly.py"
    --8<-- "./plots/learn/plots/trajectory_3d_plotly.py"
    ```

The interactive visualization shows the ISS orbit in 3D space with Earth at the origin. You can:

- **Rotate**: Click and drag to change viewing angle
- **Zoom**: Scroll or pinch to zoom in/out
- **Pan**: Right-click and drag to pan
- **Reset**: Double-click to return to default view

## Static 3D Trajectory (Matplotlib)

The matplotlib backend produces publication-ready 3D figures with customizable viewing angles.

![3D Trajectory Plot](../../figures/plot_trajectory_3d_matplotlib.png)

??? "Plot Source"

    ``` python title="trajectory_3d_matplotlib.py"
    --8<-- "./plots/learn/plots/trajectory_3d_matplotlib.py"
    ```

## Key Features

### Multiple Trajectories

Compare different orbits simultaneously:

```python
import brahe as bh
import numpy as np

bh.initialize_eop()
epoch = bh.Epoch.from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# LEO orbit
oe_leo = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(50.0), 0.0, 0.0, 0.0])
state_leo = bh.state_osculating_to_cartesian(oe_leo, bh.AngleFormat.RADIANS)
prop_leo = bh.KeplerianPropagator.from_eci(epoch, state_leo, 60.0)
prop_leo.propagate_to(epoch + bh.orbital_period(oe_leo[0]))

# GEO orbit
oe_geo = np.array([bh.R_EARTH + 35786e3, 0.001, np.radians(0.1), 0.0, 0.0, 0.0])
state_geo = bh.state_osculating_to_cartesian(oe_geo, bh.AngleFormat.RADIANS)
prop_geo = bh.KeplerianPropagator.from_eci(epoch, state_geo, 60.0)
prop_geo.propagate_to(epoch + bh.orbital_period(oe_geo[0]))

# Plot both
fig = bh.plot_trajectory_3d(
    [
        {"trajectory": prop_leo.trajectory, "color": "red", "label": "LEO"},
        {"trajectory": prop_geo.trajectory, "color": "blue", "label": "GEO"}
    ],
    show_earth=True,
    backend="plotly"
)
```

### Camera Control

Set initial viewing angle for matplotlib plots:

```python
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    view_azimuth=45.0,     # Horizontal rotation (degrees)
    view_elevation=30.0,    # Vertical angle (degrees)
    view_distance=None,     # Auto-calculate or set manually
    backend="matplotlib"
)
```

### Unit Conversion

Display in different units:

```python
# Kilometers (default)
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    units="km"
)

# Meters
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    units="m"
)

# Earth radii (normalized)
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    normalize=True  # Divide all distances by Earth radius
)
```

### Earth Visualization

Control Earth sphere appearance:

```python
# Simple blue sphere
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    show_earth=True,
    earth_texture="simple",
    backend="matplotlib"
)

# Blue Marble texture (plotly only)
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    show_earth=True,
    earth_texture="blue_marble",
    backend="plotly"
)

# No Earth sphere
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj}],
    show_earth=False
)
```

## Common Use Cases

### Orbit Comparison

Visualize how different orbital parameters affect trajectory shape:

```python
# Compare circular vs elliptical orbits
oe_circular = np.array([bh.R_EARTH + 600e3, 0.001, np.radians(50.0), 0.0, 0.0, 0.0])
oe_elliptical = np.array([bh.R_EARTH + 600e3, 0.3, np.radians(50.0), 0.0, 0.0, 0.0])

# ... create propagators and trajectories ...

fig = bh.plot_trajectory_3d(
    [
        {"trajectory": traj_circular, "color": "green", "label": "e=0.001"},
        {"trajectory": traj_elliptical, "color": "red", "label": "e=0.3"}
    ]
)
```

### Maneuver Visualization

Show before and after orbital maneuvers:

```python
# Pre-maneuver orbit
prop1.propagate_to(maneuver_epoch)
traj_before = prop1.trajectory

# Apply delta-V and propagate
new_state = apply_maneuver(prop1.current_state(), delta_v)
prop2 = bh.KeplerianPropagator.from_eci(maneuver_epoch, new_state, 60.0)
prop2.propagate_to(maneuver_epoch + duration)
traj_after = prop2.trajectory

fig = bh.plot_trajectory_3d(
    [
        {"trajectory": traj_before, "color": "blue", "label": "Before"},
        {"trajectory": traj_after, "color": "red", "label": "After"}
    ]
)
```

### Constellation Geometry

Visualize satellite constellation distribution:

```python
constellation_trajs = []
for sat in constellation:
    sat.propagate_to(epoch + period)
    constellation_trajs.append({"trajectory": sat.trajectory})

fig = bh.plot_trajectory_3d(
    constellation_trajs,
    show_earth=True,
    backend="plotly"
)
```

## Tips

- Use `backend="plotly"` for presentations and interactive exploration
- Use `backend="matplotlib"` with specific view angles for publication figures
- Set `units="km"` for typical LEO/MEO orbits, `normalize=True` for highly elliptical orbits
- Adjust `view_azimuth` and `view_elevation` to highlight specific orbital features
- For crowded plots, reduce `line_width` or use transparency

## See Also

- [plot_trajectory_3d API Reference](../../library_api/plots/trajectory_3d.md)
- [Ground Tracks](ground_tracks.md) - 2D projection on Earth's surface
- [Orbital Elements](orbital_trajectories.md) - Element evolution over time
- [Coordinate Systems](../coordinates/index.md) - Understanding ECI frames
