# 3D Trajectory Visualization

Three-dimensional trajectory plots display orbital paths in a central body's centered-inertial frame, providing intuitive spatial understanding of satellite motion. The `plot_trajectory_3d` function renders trajectories with an optional central body sphere, camera controls, and support for multiple orbits with different colors and labels. Earth is the default central body; other bodies (Moon, Mars, and any other body in the visual registry) are supported via the `central_body` parameter.

## Interactive 3D Trajectory (Plotly)

The plotly backend creates fully interactive 3D plots. Click and drag to rotate, scroll to zoom, and double-click to reset the view.

### Simple Texture (Interactive)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/trajectory_3d_plotly_simple_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/trajectory_3d_plotly_simple_dark.html"  loading="lazy"></iframe>
</div>

### Blue Marble Texture

!!! note
    Textures are provided as image-only in documentation to reduce page load times. Interactive versions can be generated using the provided code.

<figure markdown="span">
    ![3D Trajectory with Blue Marble](../../figures/trajectory_3d_plotly_light.svg#only-light)
    ![3D Trajectory with Blue Marble](../../figures/trajectory_3d_plotly_dark.svg#only-dark)
</figure>

### Natural Earth Texture

<figure markdown="span">
    ![3D Trajectory with Natural Earth](../../figures/trajectory_3d_plotly_natural_earth_light.svg#only-light)
    ![3D Trajectory with Natural Earth](../../figures/trajectory_3d_plotly_natural_earth_dark.svg#only-dark)
</figure>

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

!!! warning "Matplotlib 3D Visualization Limitations"
    The matplotlib 3D backend does not have a true 3D perspective camera model. Instead is uses a 2D layering system where entire objects (e.g., the entire orbit line, the entire sphere surface) are drawn one on top of the other based on a single, fixed `zorder` value. 

    This can lead to visual artifacts where parts of objects that should be behind other objects are incorrectly drawn in front. For example, the far side of an orbit may appear in front of the Earth sphere.

The matplotlib backend produces publication-ready 3D figures with customizable viewing angles.

<figure markdown="span">
    ![3D Trajectory Plot](../../figures/plot_trajectory_3d_matplotlib_light.svg#only-light)
    ![3D Trajectory Plot](../../figures/plot_trajectory_3d_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="trajectory_3d_matplotlib.py"
    --8<-- "./plots/learn/plots/trajectory_3d_matplotlib.py"
    ```

---

## Plotting Around Other Central Bodies

`plot_trajectory_3d` is not limited to Earth. The `central_body` parameter accepts either a registry key from `brahe.plots.bodies.BODY_VISUALS` (`'earth'`, `'moon'`, `'mars'`, `'sun'`, and the other planets) or a custom dict `{name, radius, texture}` (radius in meters) for bodies outside the registry. Trajectories plotted around a non-Earth central body must already be in `OrbitFrame.BodyCenteredInertial(naif_id)` for that body's NAIF ID; Earth trajectories in any frame are converted via `to_eci()` as before.

- `show_body` (bool): show the central body sphere at the origin. Default: `True`.
- `texture`: texture for the central body sphere (plotly only). Accepts `'simple'`, `'blue_marble'` or `'natural_earth_50m'`/`'natural_earth_10m'` (Earth only), any `brahe.plots.texture_utils.PLANET_TEXTURES` key, or a path to an image file. Defaults to the central body's registry texture (or `'simple'` for custom bodies without one).
- `additional_bodies`: a list of extra textured spheres to draw alongside the central body, each a dict with `position` (meters, in the same frame as the plotted trajectories), `radius` (meters), `texture`, and `name`.

The following example plots a lunar orbit around the Moon, with an Earth sphere included for scale via `additional_bodies`:

```python
import brahe as bh
import numpy as np

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
radius, speed = bh.R_MOON + 100e3, 1600.0
states = np.column_stack([
    radius * np.cos(angles), radius * np.sin(angles), np.zeros(20),
    -speed * np.sin(angles), speed * np.cos(angles), np.zeros(20),
])
lunar_traj = bh.OrbitTrajectory.from_orbital_data(
    [epoch + i * 60 for i in range(20)], states,
    bh.OrbitFrame.BodyCenteredInertial(301),
    bh.OrbitRepresentation.CARTESIAN, None, None,
)

fig = bh.plot_trajectory_3d(
    [{"trajectory": lunar_traj, "label": "LLO"}],
    central_body="moon",
    additional_bodies=[
        {"position": [-384.4e6, 0.0, 0.0], "radius": bh.R_EARTH,
         "texture": "blue_marble", "name": "Earth"}
    ],
    backend="plotly",
)
```

The Moon's registry texture and most planet textures are downloaded on first use from Solar System Scope and require attribution:

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

The packaged `blue_marble` and downloaded `natural_earth_50m`/`natural_earth_10m` Earth textures are not part of that set and require no additional attribution.

## See Also

- [plot_trajectory_3d API Reference](../../library_api/plots/3d_trajectory.md)
- [Synodic Plots](synodic_plots.md) - Trajectories in rotating two-body frames
- [Ground Tracks](ground_tracks.md) - 2D projection on Earth's surface
- [Orbital Elements](orbital_trajectories.md) - Element evolution over time
- [Coordinate Systems](../coordinates/index.md) - Understanding ECI frames
