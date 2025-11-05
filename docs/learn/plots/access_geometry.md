# Access Geometry

Access geometry plots visualize satellite visibility from ground stations, showing where satellites appear in the sky and how their elevation changes over time. Brahe provides three complementary views: polar plots showing azimuth and elevation, elevation vs azimuth plots showing the observed horizon, and time-series plots tracking elevation angle during passes.

All plot types support optional **elevation masks** to visualize terrain obstructions, antenna constraints, or other azimuth-dependent visibility limits.

## Polar Access Plot (Azimuth/Elevation)

Polar plots display the satellite's path across the sky in azimuth-elevation coordinates, providing an intuitive "looking up" view from the ground station.

### Interactive Polar Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/access_polar_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/access_polar_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="access_polar_plotly.py"
    --8<-- "./plots/learn/plots/access_polar_plotly.py"
    ```

### Static Polar Plot (Matplotlib)

<figure markdown="span">
    ![Access Polar Plot](../../figures/plot_access_polar_matplotlib_light.svg#only-light)
    ![Access Polar Plot](../../figures/plot_access_polar_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="access_polar_matplotlib.py"
    --8<-- "./plots/learn/plots/access_polar_matplotlib.py"
    ```

The polar plot shows:

- **Radial axis**: Elevation angle (0° at edge, 90° at center)
- **Angular axis**: Azimuth (0° = North, 90° = East, 180° = South, 270° = West)
- **Satellite path**: Track showing where the satellite appears in the sky

## Elevation vs Azimuth Plot (Observed Horizon)

Elevation vs azimuth plots show satellite paths across the observed horizon, with azimuth on the X-axis and elevation on the Y-axis. This view is particularly useful for visualizing terrain obstructions and azimuth-dependent visibility constraints using **elevation masks**.

### Interactive Elevation vs Azimuth Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/access_elevation_azimuth_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/access_elevation_azimuth_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="access_elevation_azimuth_plotly.py"
    --8<-- "./plots/learn/plots/access_elevation_azimuth_plotly.py"
    ```

### Static Elevation vs Azimuth Plot (Matplotlib)

<figure markdown="span">
    ![Access Elevation vs Azimuth Plot](../../figures/plot_access_elevation_azimuth_matplotlib_light.svg#only-light)
    ![Access Elevation vs Azimuth Plot](../../figures/plot_access_elevation_azimuth_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="access_elevation_azimuth_matplotlib.py"
    --8<-- "./plots/learn/plots/access_elevation_azimuth_matplotlib.py"
    ```

The elevation vs azimuth plot shows:

- **X-axis**: Azimuth angle (0° to 360°, North = 0°/360°)
- **Y-axis**: Elevation angle (0° to 90°)
- **Satellite trajectory**: Path across the sky from observer's perspective
- **Elevation mask** (shaded region): Visibility constraints varying with azimuth
- **Discontinuity handling**: Trajectories crossing 0°/360° azimuth are split to avoid artifacts

### Elevation Masks

Elevation masks define azimuth-dependent minimum elevation constraints. They can represent:

- **Terrain obstructions**: Mountains, buildings, trees
- **Antenna constraints**: Dish beamwidth, gimbal limits
- **Operational requirements**: RF interference avoidance zones

The example above uses a sinusoidal mask: **15° + 10° sin(2×azimuth)**, varying between 5° and 25° around the horizon.

#### Using Elevation Masks

Elevation masks can be specified in three ways:

```python
# Constant elevation (simple threshold)
fig = bh.plot_access_elevation_azimuth(
    windows, prop,
    elevation_mask=10.0,  # 10° everywhere
    backend="matplotlib"
)

# Function of azimuth (variable constraint)
mask_fn = lambda az: 15.0 + 10.0 * np.sin(np.radians(2 * az))
fig = bh.plot_access_elevation_azimuth(
    windows, prop,
    elevation_mask=mask_fn,
    backend="matplotlib"
)

# Array of values (measured terrain profile)
azimuths = np.linspace(0, 360, 361)
elevations = [measured_elevation(az) for az in azimuths]
fig = bh.plot_access_elevation_azimuth(
    windows, prop,
    elevation_mask=elevations,  # Must match azimuth sampling
    backend="matplotlib"
)
```

Elevation masks are also supported in polar plots (`plot_access_polar`) where they appear as shaded regions around the plot edge.

## Elevation vs Time Plot

Time-series plots show how elevation angle changes throughout a satellite pass, useful for link budget analysis and antenna pointing.

### Interactive Elevation Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/access_elevation_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/access_elevation_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="access_elevation_plotly.py"
    --8<-- "./plots/learn/plots/access_elevation_plotly.py"
    ```

### Static Elevation Plot (Matplotlib)

<figure markdown="span">
    ![Access Elevation Plot](../../figures/plot_access_elevation_matplotlib_light.svg#only-light)
    ![Access Elevation Plot](../../figures/plot_access_elevation_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"


    ``` python title="access_elevation_matplotlib.py"
    --8<-- "./plots/learn/plots/access_elevation_matplotlib.py"
    ```

---

## See Also

- [plot_access_polar API Reference](../../library_api/plots/access_geometry.md)
- [plot_access_elevation_azimuth API Reference](../../library_api/plots/access_geometry.md)
- [plot_access_elevation API Reference](../../library_api/plots/access_geometry.md)
- [location_accesses](../../library_api/access/index.md) - Computing access windows
- [Ground Tracks](ground_tracks.md) - Visualizing coverage on maps
- [Access Constraints](../../library_api/access/constraints.md) - Defining visibility rules
