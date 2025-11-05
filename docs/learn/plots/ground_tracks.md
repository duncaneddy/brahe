# Ground Tracks

Ground track plotting visualizes the path a satellite traces over Earth's surface. This is essential for mission planning, coverage analysis, and understanding when and where a satellite can communicate with ground stations. Brahe's `plot_groundtrack` function renders satellite trajectories on a world map with optional ground station markers and communication coverage cones.

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

## Configuration and Customization

### Multiple Spacecraft

Plot multiple satellites simultaneously to compare orbits or analyze constellations:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/groundtrack_multiple_spacecraft_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/groundtrack_multiple_spacecraft_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="groundtrack_multiple_spacecraft.py"
    --8<-- "./plots/learn/plots/groundtrack_multiple_spacecraft.py"
    ```

This example shows three different LEO orbits:
- **Red**: Sun-synchronous orbit (98° inclination, 700km altitude)
- **Blue**: Medium inclination (55°, 600km altitude)
- **Green**: Equatorial orbit (5° inclination, 800km altitude)

### Ground Station Networks

Visualize satellite visibility over ground station networks with geodetic coverage zones:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/groundtrack_nasa_nen_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/groundtrack_nasa_nen_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="groundtrack_nasa_nen.py"
    --8<-- "./plots/learn/plots/groundtrack_nasa_nen.py"
    ```

This example demonstrates:
- Loading the NASA Near Earth Network (NEN) ground stations from built-in datasets
- Computing geodetic coverage zones for each station at 10° minimum elevation
- Displaying coverage as semi-transparent filled polygons on the map
- Visualizing actual ground footprints for a 550km altitude LEO satellite

The coverage zones are computed as properly transformed geodetic shapes, showing the actual region on Earth's surface where the satellite is visible above the minimum elevation angle.

Available ground station networks include: `"atlas"`, `"aws"`, `"ksat"`, `"leaf"`, `"nasa dsn"`, `"nasa nen"`, `"ssc"`, and `"viasat"`.

### Map Styles

Choose from different basemap styles to suit your presentation needs:

#### Natural Earth (High-Quality Vector)

<figure markdown="span">
    ![Natural Earth Basemap](../../figures/groundtrack_basemaps_natural_earth_light.svg#only-light)
    ![Natural Earth Basemap](../../figures/groundtrack_basemaps_natural_earth_dark.svg#only-dark)
</figure>

#### Stock (Cartopy Built-in, Minimal)

<figure markdown="span">
    ![Stock Basemap](../../figures/groundtrack_basemaps_stock_light.svg#only-light)
    ![Stock Basemap](../../figures/groundtrack_basemaps_stock_dark.svg#only-dark)
</figure>

#### Blue Marble (Satellite Imagery)

<figure markdown="span">
    ![Blue Marble Basemap](../../figures/groundtrack_basemaps_blue_marble_light.svg#only-light)
    ![Blue Marble Basemap](../../figures/groundtrack_basemaps_blue_marble_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="groundtrack_basemaps.py"
    --8<-- "./plots/learn/plots/groundtrack_basemaps.py"
    ```

The basemap styles offer different visualization approaches:
- **Natural Earth**: High-quality vector map with political boundaries and natural features (default)
- **Stock**: Minimal Cartopy background without geographic features, ideal for clean presentations
- **Blue Marble**: NASA satellite imagery texture provides photorealistic Earth background

Set the `basemap` parameter to `"natural_earth"` (default), `"stock"`, or `None` to control the map style. For Blue Marble, use `basemap=None` and manually overlay the texture image.

!!! note "Backend Capabilities"
    **Matplotlib backend** supports all basemap styles including Natural Earth shapefiles and Blue Marble textures.

    **Plotly backend** uses Scattergeo which only supports outline-based maps with solid colors. Custom textures (Natural Earth shapefiles, Blue Marble imagery) are not available in the plotly backend. Use `basemap="natural_earth"` for a light gray landmass color or `basemap="stock"` for tan.

## Advanced Examples

### Maximum Coverage Gap Analysis

This advanced example identifies the longest period without ground station contact and visualizes only that critical gap segment:

This demonstrates how to:
- Compute access windows between a satellite and ground network
- Find the longest gap between consecutive contacts
- Extract and plot only the gap segment (without the full 24-hour ground track)
- Handle antimeridian wraparound with custom plotting

<figure markdown="span">
    ![Maximum Coverage Gap](../../figures/groundtrack_max_gap_light.svg#only-light)
    ![Maximum Coverage Gap](../../figures/groundtrack_max_gap_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="groundtrack_max_gap.py"
    --8<-- "./plots/learn/plots/groundtrack_max_gap.py"
    ```

This example uses the `split_ground_track_at_antimeridian()` helper function to properly handle longitude wraparound when plotting custom ground track segments. The helper function detects jumps across the ±180° boundary and splits the track into separate segments for correct rendering.

## Additional Features

### Coverage Zones

Add polygon zones for restricted areas, target regions, or sensor footprints:

```python
import numpy as np

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

### Map Extent

Zoom into specific regions using the `extent` parameter:

```python
# Focus on North America
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj}],
    extent=[-130, -60, 20, 50],  # [lon_min, lon_max, lat_min, lat_max]
    backend="matplotlib"
)
```

## Tips

- Use `backend="plotly"` for interactive exploration and presentations
- Use `backend="matplotlib"` for publication-quality static figures
- Set `gs_cone_altitude` to your satellite's altitude for accurate coverage visualization
- Adjust `gs_min_elevation` based on antenna pointing constraints (typically 5-15°)
- Use `extent` parameter to zoom into specific regions of interest
- Control displayed track length with `track_length` and `track_units` parameters
- Use `split_ground_track_at_antimeridian()` when creating custom ground track overlays to handle longitude wraparound
- Choose basemap style based on your audience: `"natural_earth"` for presentations, `"stock"` for quick analysis, `None` for minimal distraction

## See Also

- [plot_groundtrack API Reference](../../library_api/plots/ground_tracks.md) - Complete function documentation
- [split_ground_track_at_antimeridian API Reference](../../library_api/plots/ground_tracks.md) - Wraparound handling
- [Access Geometry](access_geometry.md) - Detailed visibility analysis
- [PointLocation](../../library_api/access/locations.md) - Ground station definitions
- [PolygonLocation](../../library_api/access/locations.md) - Zone definitions
