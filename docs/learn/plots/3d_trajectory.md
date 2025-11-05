# 3D Trajectory Visualization

Three-dimensional trajectory plots display orbital paths in Earth-Centered Inertial (ECI) coordinates, providing intuitive spatial understanding of satellite motion. The `plot_trajectory_3d` function renders trajectories with optional Earth sphere visualization, camera controls, and support for multiple orbits with different colors and labels.

## Interactive 3D Trajectory (Plotly)

The plotly backend creates fully interactive 3D plots. Click and drag to rotate, scroll to zoom, and double-click to reset the view.

### Simple Texture (Interactive)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/trajectory_3d_plotly_simple_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/trajectory_3d_plotly_simple_dark.html"  loading="lazy"></iframe>
</div>

### Blue Marble Texture

!!! note
    Textues are provided as image-only in documentation to reduce page load times. Interactive versions can be generated using the provided code.

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

## See Also

- [plot_trajectory_3d API Reference](../../library_api/plots/3d_trajectory.md)
- [Ground Tracks](ground_tracks.md) - 2D projection on Earth's surface
- [Orbital Elements](orbital_trajectories.md) - Element evolution over time
- [Coordinate Systems](../coordinates/index.md) - Understanding ECI frames
