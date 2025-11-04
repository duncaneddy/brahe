# Access Geometry

Access geometry plots visualize satellite visibility from ground stations, showing where satellites appear in the sky and how their elevation changes over time. Brahe provides two complementary views: polar plots showing azimuth and elevation, and time-series plots tracking elevation angle during passes.

See also: [plot_access_polar](../../library_api/plots/access.md), [plot_access_elevation](../../library_api/plots/access.md)

## Polar Access Plot (Azimuth/Elevation)

Polar plots display the satellite's path across the sky in azimuth-elevation coordinates, providing an intuitive "looking up" view from the ground station.

### Interactive Polar Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/plot_access_polar_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/plot_access_polar_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="access_polar_plotly.py"
    --8<-- "./plots/learn/plots/access_polar_plotly.py"
    ```

### Static Polar Plot (Matplotlib)

![Access Polar Plot](../../figures/plot_access_polar_matplotlib_light.svg#only-light)
![Access Polar Plot](../../figures/plot_access_polar_matplotlib_dark.svg#only-dark)

??? "Plot Source"

    ``` python title="access_polar_matplotlib.py"
    --8<-- "./plots/learn/plots/access_polar_matplotlib.py"
    ```

The polar plot shows:

- **Radial axis**: Elevation angle (0° at edge, 90° at center)
- **Angular axis**: Azimuth (0° = North, 90° = East, 180° = South, 270° = West)
- **Satellite path**: Track showing where the satellite appears in the sky

## Elevation vs Time Plot

Time-series plots show how elevation angle changes throughout a satellite pass, useful for link budget analysis and antenna pointing.

### Interactive Elevation Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/plot_access_elevation_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/plot_access_elevation_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="access_elevation_plotly.py"
    --8<-- "./plots/learn/plots/access_elevation_plotly.py"
    ```

### Static Elevation Plot (Matplotlib)

![Access Elevation Plot](../../figures/plot_access_elevation_matplotlib_light.svg#only-light)
![Access Elevation Plot](../../figures/plot_access_elevation_matplotlib_dark.svg#only-dark)

??? "Plot Source"

    ``` python title="access_elevation_matplotlib.py"
    --8<-- "./plots/learn/plots/access_elevation_matplotlib.py"
    ```

## Generating Access Windows

Both plot types require access windows computed using `location_accesses`:

```python
import brahe as bh
import numpy as np

bh.initialize_eop()

# Create satellite propagator
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)

# Define ground station
station = bh.PointLocation(
    np.radians(40.7128),   # Latitude (rad)
    np.radians(-74.0060),  # Longitude (rad)
    0.0                    # Altitude (m)
).with_name("New York")

# Compute access windows over 24 hours
epoch = prop.epoch
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
accesses = bh.location_accesses(
    [station], [prop],
    epoch, epoch + 24*3600,
    constraint
)

# Plot first access window
if len(accesses) > 0:
    fig_polar = bh.plot_access_polar(accesses[0])
    fig_elev = bh.plot_access_elevation(accesses[0])
```

## Common Use Cases

### Link Budget Analysis

Elevation plots help determine signal strength variation:

```python
# Higher elevation = stronger signal due to shorter path
fig = bh.plot_access_elevation(access_window, backend="plotly")

# Identify maximum elevation point (closest approach)
# Determine contact duration above threshold elevation
```

### Antenna Pointing

Polar plots visualize required antenna motion:

```python
# See complete azimuth/elevation profile for antenna tracking
fig = bh.plot_access_polar(access_window)

# Identify if satellite passes overhead (high elevation)
# Plan antenna slew rates based on azimuth changes
```

### Pass Classification

Categorize satellite passes by geometry:

```python
for access in accesses:
    fig = bh.plot_access_polar(access)

    # Overhead pass: High maximum elevation
    # Horizon pass: Low maximum elevation
    # North/South/East/West pass: Azimuth range
```

### Multiple Passes Comparison

Compare different passes from same station:

```python
# Plot elevation profiles for all passes in 24 hours
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for i, access in enumerate(accesses):
    # Extract elevation data from access window
    # ... plot on same axes ...
    pass

plt.xlabel("Time")
plt.ylabel("Elevation (degrees)")
plt.title("24-Hour ISS Passes")
```

## Understanding Access Geometry

### Polar Plot Interpretation

- **Entry point**: Where satellite rises above horizon
- **Exit point**: Where satellite sets below horizon
- **Peak elevation**: Closest point (center of plot)
- **Azimuth changes**: How fast antenna must slew

### Elevation Plot Interpretation

- **Duration**: Time between acquisition and loss of signal (AOS/LOS)
- **Maximum elevation**: Best signal strength point
- **Rise/fall rates**: Antenna elevation motor requirements
- **Flat portions**: Rare, indicates satellite hovering (geosynchronous only)

## Tips

- Use `backend="plotly"` to hover and see exact azimuth/elevation/time values
- Polar plots are best for understanding antenna pointing requirements
- Elevation plots are best for link budget and duration analysis
- Higher elevation passes generally provide longer contact times and better signal
- Typical minimum elevation thresholds: 5° (good horizon), 10° (standard), 15° (conservative)

## See Also

- [plot_access_polar API Reference](../../library_api/plots/access.md)
- [plot_access_elevation API Reference](../../library_api/plots/access.md)
- [location_accesses](../../library_api/access/computation.md) - Computing access windows
- [Ground Tracks](ground_tracks.md) - Visualizing coverage on maps
- [Access Constraints](../../library_api/access/constraints.md) - Defining visibility rules
