"""
3D trajectory visualization in ECI frame.

Provides 3D plots of orbital trajectories with optional Earth sphere.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.texture_utils import load_earth_texture
from brahe._brahe import OrbitTrajectory, OrbitFrame, OrbitRepresentation


def plot_trajectory_3d(
    trajectories,
    time_range=None,
    units="km",
    normalize=False,
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=None,
    show_earth=True,
    earth_texture=None,
    sphere_resolution_lon=1080,
    sphere_resolution_lat=540,
    backend="matplotlib",
    width=None,
    height=None,
) -> object:
    """Plot 3D trajectory in ECI frame.

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array [N×3] or [N×6] (positions in ECI)
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        units (str, optional): 'm' or 'km'. Default: 'km'
        normalize (bool, optional): Normalize to Earth radii. Default: False
        view_azimuth (float, optional): Camera azimuth angle (degrees). Default: 45.0
        view_elevation (float, optional): Camera elevation angle (degrees). Default: 30.0
        view_distance (float, optional): Camera distance multiplier. Default: 2.5 (larger = further out)
        show_earth (bool, optional): Show Earth sphere at origin. Default: True
        earth_texture (str, optional): Texture to use for Earth sphere (plotly only). Options:
            - 'simple': Solid lightblue sphere (fast rendering)
            - 'blue_marble': NASA Blue Marble texture (packaged with brahe, default for plotly)
            - 'natural_earth_50m': Natural Earth 50m shaded relief (auto-downloads ~20MB)
            - 'natural_earth_10m': Natural Earth 10m shaded relief (auto-downloads ~180MB)
            Note: matplotlib always uses a simple solid sphere regardless of this setting.
            Default: 'blue_marble' for plotly
        sphere_resolution_lon (int, optional): Longitude resolution for textured sphere (plotly only).
            Higher values = better quality but slower rendering. Default: 1080
        sphere_resolution_lat (int, optional): Latitude resolution for textured sphere (plotly only).
            Higher values = better quality but slower rendering. Default: 540
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'
        width (int, optional): Figure width in pixels (plotly only). Default: None (responsive)
        height (int, optional): Figure height in pixels (plotly only). Default: None (responsive)

    Returns:
        Generated figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        traj = prop.propagate(epoch, epoch + bh.orbital_period(oe[0]), 60.0)

        # Plot 3D trajectory with matplotlib (simple sphere)
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            show_earth=True,
            backend='matplotlib'
        )

        # Plot 3D trajectory with plotly (blue marble texture by default)
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            show_earth=True,
            backend='plotly'
        )

        # Plot with explicit texture choice
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            show_earth=True,
            earth_texture='natural_earth_50m',
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting 3D trajectory with backend={backend}")
    logger.debug(f"Units: {units}, normalize={normalize}, show_earth={show_earth}")

    validate_backend(backend)

    # Set backend-specific default textures
    if earth_texture is None:
        earth_texture = "simple" if backend == "matplotlib" else "blue_marble"

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _trajectory_3d_matplotlib(
            traj_groups,
            time_range,
            units,
            normalize,
            view_azimuth,
            view_elevation,
            view_distance,
            show_earth,
        )
    else:  # plotly
        result = _trajectory_3d_plotly(
            traj_groups,
            time_range,
            units,
            normalize,
            view_azimuth,
            view_elevation,
            view_distance,
            show_earth,
            earth_texture,
            sphere_resolution_lon,
            sphere_resolution_lat,
            width,
            height,
        )

    elapsed = time.time() - start_time
    logger.info(f"3D trajectory plot completed in {elapsed:.2f}s")
    return result


def _normalize_trajectory_groups(trajectories):
    """Normalize trajectory input to list of dicts with defaults."""
    defaults = {
        "color": None,
        "line_width": 2.0,
        "label": None,
    }

    if trajectories is None:
        return []

    if not isinstance(trajectories, list):
        return [{**defaults, "trajectory": trajectories}]

    if len(trajectories) == 0:
        return []

    if not isinstance(trajectories[0], dict):
        # List of trajectories without config
        return [{**defaults, "trajectory": t} for t in trajectories]

    # List of dicts - apply defaults
    return [{**defaults, **group} for group in trajectories]


def _trajectory_3d_matplotlib(
    traj_groups,
    time_range,
    units,
    normalize,
    view_azimuth,
    view_elevation,
    view_distance,
    show_earth,
):
    """Matplotlib implementation of 3D trajectory plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Unit conversion
    scale = 1.0
    if normalize:
        scale = 1.0 / bh.R_EARTH
        unit_label = "Earth Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot Earth sphere if requested
    if show_earth:
        _plot_earth_sphere_matplotlib(ax, scale)

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        # Validate that trajectory is an OrbitTrajectory
        if not isinstance(trajectory, OrbitTrajectory):
            raise TypeError(
                f"Trajectory must be an OrbitTrajectory object, got {type(trajectory)}"
            )

        # Convert to ECI Cartesian if needed
        if (
            trajectory.frame != OrbitFrame.ECI
            or trajectory.representation != OrbitRepresentation.CARTESIAN
        ):
            logger.debug(
                f"Converting trajectory from {trajectory.frame}/{trajectory.representation} to ECI/CARTESIAN"
            )
            trajectory = trajectory.to_eci()

        # Extract positions using to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        pos_x = states[:, 0]
        pos_y = states[:, 1]
        pos_z = states[:, 2]

        # Scale positions
        pos_x = pos_x * scale
        pos_y = pos_y * scale
        pos_z = pos_z * scale

        # Plot 3D line
        ax.plot(
            pos_x, pos_y, pos_z, color=color, linewidth=line_width_group, label=label
        )

    # Set viewing angle
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # Set labels
    ax.set_xlabel(f"X ({unit_label})")
    ax.set_ylabel(f"Y ({unit_label})")
    ax.set_zlabel(f"Z ({unit_label})")
    ax.set_title("3D Trajectory (ECI Frame)")

    # Equal aspect ratio
    _set_axes_equal(ax)

    ax.legend()

    return fig


def _trajectory_3d_plotly(
    traj_groups,
    time_range,
    units,
    normalize,
    view_azimuth,
    view_elevation,
    view_distance,
    show_earth,
    earth_texture,
    sphere_resolution_lon,
    sphere_resolution_lat,
    width,
    height,
):
    """Plotly implementation of 3D trajectory plot."""
    fig = go.Figure()

    # Unit conversion
    scale = 1.0
    if normalize:
        scale = 1.0 / bh.R_EARTH
        unit_label = "Earth Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot Earth sphere if requested
    if show_earth:
        _plot_earth_sphere_plotly(
            fig, scale, earth_texture, sphere_resolution_lon, sphere_resolution_lat
        )

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        # Validate that trajectory is an OrbitTrajectory
        if not isinstance(trajectory, OrbitTrajectory):
            raise TypeError(
                f"Trajectory must be an OrbitTrajectory object, got {type(trajectory)}"
            )

        # Convert to ECI Cartesian if needed
        if (
            trajectory.frame != OrbitFrame.ECI
            or trajectory.representation != OrbitRepresentation.CARTESIAN
        ):
            logger.debug(
                f"Converting trajectory from {trajectory.frame}/{trajectory.representation} to ECI/CARTESIAN"
            )
            trajectory = trajectory.to_eci()

        # Extract positions using to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        pos_x = states[:, 0]
        pos_y = states[:, 1]
        pos_z = states[:, 2]

        # Scale positions
        pos_x = pos_x * scale
        pos_y = pos_y * scale
        pos_z = pos_z * scale

        # Add 3D scatter trace
        fig.add_trace(
            go.Scatter3d(
                x=pos_x,
                y=pos_y,
                z=pos_z,
                mode="lines",
                line=dict(color=color, width=line_width_group),
                name=label,
            )
        )

    # Set default view distance if not specified (larger = further out)
    if view_distance is None:
        view_distance = 2.5  # Default: zoom out to 2.5x distance

    # Configure layout
    layout_config = {
        "title": "3D Trajectory (ECI Frame)",
        "scene": dict(
            xaxis_title=f"X ({unit_label})",
            yaxis_title=f"Y ({unit_label})",
            zaxis_title=f"Z ({unit_label})",
            aspectmode="data",
            xaxis=dict(showgrid=True, showbackground=False),
            yaxis=dict(showgrid=True, showbackground=False),
            zaxis=dict(showgrid=True, showbackground=False),
            camera=dict(
                eye=dict(
                    x=view_distance
                    * np.cos(np.radians(view_azimuth))
                    * np.cos(np.radians(view_elevation)),
                    y=view_distance
                    * np.sin(np.radians(view_azimuth))
                    * np.cos(np.radians(view_elevation)),
                    z=view_distance * np.sin(np.radians(view_elevation)),
                )
            ),
        ),
    }

    # Only set width/height if explicitly provided
    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    return fig


def _plot_earth_sphere_matplotlib(ax, scale):
    """Plot simple Earth sphere on matplotlib 3D axis."""
    # Simple solid sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = bh.R_EARTH * scale * np.outer(np.cos(u), np.sin(v))
    y = bh.R_EARTH * scale * np.outer(np.sin(u), np.sin(v))
    z = bh.R_EARTH * scale * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.6, edgecolor="none")


def _plot_earth_sphere_plotly(fig, scale, texture, n_lon=1080, n_lat=540):
    """Plot Earth sphere on plotly 3D figure using Mesh3d with optional texture mapping."""
    # Load texture if requested
    texture_img = load_earth_texture(texture)

    if texture_img is None:
        # Simple solid sphere using Surface (faster for untextured)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = bh.R_EARTH * scale * np.outer(np.cos(u), np.sin(v))
        y = bh.R_EARTH * scale * np.outer(np.sin(u), np.sin(v))
        z = bh.R_EARTH * scale * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Blues",
                showscale=False,
                opacity=0.7,
                name="Earth",
            )
        )
    else:
        # Textured sphere using Mesh3d with facecolor
        # Use configurable resolution for texture quality/performance trade-off

        # Generate sphere vertices
        lons = np.linspace(0, 2 * np.pi, n_lon)
        lats = np.linspace(0, np.pi, n_lat)

        vertices = []
        for lat in lats:
            for lon in lons:
                x = bh.R_EARTH * scale * np.sin(lat) * np.cos(lon)
                y = bh.R_EARTH * scale * np.sin(lat) * np.sin(lon)
                z = bh.R_EARTH * scale * np.cos(lat)
                vertices.append([x, y, z])

        vertices = np.array(vertices)
        x_verts, y_verts, z_verts = vertices[:, 0], vertices[:, 1], vertices[:, 2]

        # Generate triangular faces
        faces = []
        for i in range(n_lat - 1):
            for j in range(n_lon - 1):
                # Vertex indices
                v0 = i * n_lon + j
                v1 = i * n_lon + (j + 1)
                v2 = (i + 1) * n_lon + j
                v3 = (i + 1) * n_lon + (j + 1)

                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        faces = np.array(faces)
        i_faces, j_faces, k_faces = faces[:, 0], faces[:, 1], faces[:, 2]

        # Map texture to face colors
        # Get texture dimensions
        img_array = np.array(texture_img)
        img_height, img_width = img_array.shape[:2]

        # For each face, compute the average texture coordinate and sample color
        face_colors = []
        for face in faces:
            # Get vertices of this face
            v_indices = face

            # Compute average latitude/longitude for this face
            avg_x = np.mean(vertices[v_indices, 0])
            avg_y = np.mean(vertices[v_indices, 1])
            avg_z = np.mean(vertices[v_indices, 2])

            # Convert back to lat/lon
            r = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
            if r > 0:
                lat = np.arccos(avg_z / r)
                lon = np.arctan2(avg_y, avg_x)
            else:
                lat, lon = 0, 0

            # Map to texture coordinates (0 to 1)
            # lon: 0 to 2π maps to 0 to 1
            # lat: 0 to π maps to 0 to 1
            u_coord = (lon % (2 * np.pi)) / (2 * np.pi)
            v_coord = lat / np.pi

            # Sample texture at this location
            tex_x = int(u_coord * (img_width - 1))
            tex_y = int(v_coord * (img_height - 1))

            # Get RGB color from texture
            rgb = img_array[tex_y, tex_x, :3]
            face_colors.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")

        # Create Mesh3d with face colors
        fig.add_trace(
            go.Mesh3d(
                x=x_verts,
                y=y_verts,
                z=z_verts,
                i=i_faces,
                j=j_faces,
                k=k_faces,
                facecolor=face_colors,
                showscale=False,
                name="Earth",
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.8,
                ),
                lightposition=dict(x=10000, y=10000, z=10000),
            )
        )


def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Args:
        ax: matplotlib 3D axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
