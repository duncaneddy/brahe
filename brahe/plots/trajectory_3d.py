"""
3D trajectory visualization for arbitrary central bodies.

Provides 3D plots of orbital trajectories with an optional textured sphere
for the central body and any number of additional labeled bodies.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.texture_utils import load_body_texture
from brahe.plots.bodies import resolve_body
from brahe._brahe import OrbitTrajectory, OrbitFrame, OrbitRepresentation


def plot_trajectory_3d(
    trajectories,
    time_range=None,
    units="km",
    normalize=False,
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=None,
    central_body="earth",
    show_body=True,
    texture=None,
    additional_bodies=None,
    sphere_resolution_lon=360,
    sphere_resolution_lat=180,
    backend="matplotlib",
    width=None,
    height=None,
) -> object:
    """Plot 3D trajectories about a central body.

    Trajectories are plotted in the central body's centered-inertial frame.
    For Earth (the default), trajectories in any frame are converted via
    ``to_eci()``. For other central bodies, trajectories must already be in
    ``OrbitFrame.BodyCenteredInertial(naif_id)`` for that body's NAIF ID;
    custom bodies without a NAIF ID are plotted using the trajectory's raw
    Cartesian position, unconverted.

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        units (str, optional): 'm' or 'km'. Default: 'km'
        normalize (bool, optional): Normalize distances by the central body's
            radius. Default: False
        view_azimuth (float, optional): Camera azimuth angle (degrees). Default: 45.0
        view_elevation (float, optional): Camera elevation angle (degrees). Default: 30.0
        view_distance (float, optional): Camera distance multiplier. Default: 2.5 (larger = further out)
        central_body (str or dict, optional): Central body to plot trajectories
            around. Either a ``brahe.plots.bodies.BODY_VISUALS`` registry key
            (e.g. ``'earth'``, ``'moon'``, ``'mars'``) or a custom dict
            ``{name, radius, texture}`` (radius in meters). Default: 'earth'
        show_body (bool, optional): Show the central body sphere at the
            origin. Default: True
        texture (str, optional): Texture to use for the central body sphere
            (plotly only). Options:
            - 'simple': Solid lightblue sphere (fast rendering)
            - 'blue_marble': NASA Blue Marble texture (packaged with brahe, Earth only)
            - 'natural_earth_50m': Natural Earth 50m shaded relief (auto-downloads ~20MB, Earth only)
            - 'natural_earth_10m': Natural Earth 10m shaded relief (auto-downloads ~180MB, Earth only)
            - Any other ``brahe.plots.texture_utils.PLANET_TEXTURES`` key (auto-downloads,
              CC BY 4.0, Solar System Scope: https://www.solarsystemscope.com/textures/)
            - A path to an image file
            Note: matplotlib always uses a simple solid sphere regardless of this setting.
            Default: 'simple' for matplotlib; the central body's registry texture (or
            'simple' for custom bodies without one) for plotly
        additional_bodies (list of dict, optional): Extra textured spheres to
            draw alongside the central body, each with:
            - position (array-like, length 3): Position in meters, in the
              same frame as the plotted trajectories
            - radius (float): Radius in meters
            - texture (str, Path, or None, optional): Texture, as for
              ``texture`` above
            - name (str): Label used in the legend/hover text
        sphere_resolution_lon (int, optional): Longitude resolution for textured sphere (plotly only).
            Higher values = better quality but slower rendering, with a larger output file
            (the textured sphere is encoded as a per-face-colored Mesh3d). Default: 360
        sphere_resolution_lat (int, optional): Latitude resolution for textured sphere (plotly only).
            Higher values = better quality but slower rendering, with a larger output file
            (the textured sphere is encoded as a per-face-colored Mesh3d). Default: 180
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'
        width (int, optional): Figure width in pixels (plotly only). Default: None (responsive)
        height (int, optional): Figure height in pixels (plotly only). Default: None (responsive)

    Returns:
        object: Generated figure (matplotlib.figure.Figure or plotly.graph_objects.Figure)

    Raises:
        ValueError: If ``central_body`` is not recognized, or a trajectory's
            frame does not match the expected body-centered-inertial frame.
        TypeError: If a trajectory is not an OrbitTrajectory object.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        traj = prop.propagate(epoch, epoch + bh.orbital_period(oe[0]), 60.0)

        # Plot 3D trajectory around Earth with matplotlib (simple sphere)
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            backend='matplotlib'
        )

        # Plot 3D trajectory with plotly (Blue Marble texture)
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            texture='blue_marble',
            backend='plotly'
        )

        # Create a small lunar orbit
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        radius, speed = bh.R_MOON + 100e3, 1600.0
        states = np.column_stack([
            radius*np.cos(angles), radius*np.sin(angles), np.zeros(20),
            -speed*np.sin(angles), speed*np.cos(angles), np.zeros(20)
        ])
        lunar_traj = bh.OrbitTrajectory.from_orbital_data(
            [epoch + i*60 for i in range(20)], states,
            bh.OrbitFrame.BodyCenteredInertial(301),
            bh.OrbitRepresentation.CARTESIAN, None, None
        )

        # Plot a Moon-centered trajectory with an Earth sphere shown for scale
        fig = bh.plot_trajectory_3d(
            [{"trajectory": lunar_traj, "label": "LLO"}],
            central_body='moon',
            additional_bodies=[
                {"position": [-384.4e6, 0.0, 0.0], "radius": bh.R_EARTH,
                 "texture": "blue_marble", "name": "Earth"}
            ],
            backend='plotly'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting 3D trajectory with backend={backend}")
    logger.debug(f"Units: {units}, normalize={normalize}, show_body={show_body}")

    validate_backend(backend)

    body = resolve_body(central_body)

    # Set backend-specific default textures
    if texture is None:
        texture = "simple" if backend == "matplotlib" else (body["texture"] or "simple")

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)
    bodies = additional_bodies or []

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
            body,
            show_body,
            bodies,
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
            body,
            show_body,
            texture,
            bodies,
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


def _coerce_trajectory_frame(trajectory, body):
    """Validate/convert a trajectory into the central body's centered-inertial frame."""
    if not isinstance(trajectory, OrbitTrajectory):
        raise TypeError(
            f"Trajectory must be an OrbitTrajectory object, got {type(trajectory)}"
        )

    if body["naif_id"] == 399:
        if (
            trajectory.frame != OrbitFrame.ECI
            or trajectory.representation != OrbitRepresentation.CARTESIAN
        ):
            logger.debug(
                f"Converting trajectory from {trajectory.frame}/{trajectory.representation} to ECI/CARTESIAN"
            )
            trajectory = trajectory.to_eci()
        return trajectory

    if body["naif_id"] is not None:
        expected_frame = OrbitFrame.BodyCenteredInertial(body["naif_id"])
        if trajectory.frame != expected_frame:
            raise ValueError(
                f"Trajectory frame {trajectory.frame} does not match the expected "
                f"frame {expected_frame} for central body '{body['name']}'"
            )

    return trajectory


def _trajectory_3d_matplotlib(
    traj_groups,
    time_range,
    units,
    normalize,
    view_azimuth,
    view_elevation,
    view_distance,
    body,
    show_body,
    additional_bodies,
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
        scale = 1.0 / body["radius"]
        unit_label = f"{body['name']} Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot central body sphere if requested
    if show_body:
        _plot_body_sphere_matplotlib(ax, body["radius"], scale, body["name"])

    for extra in additional_bodies:
        extra_body = resolve_body(extra)
        center = np.asarray(extra["position"], dtype=float) * scale
        _plot_body_sphere_matplotlib(
            ax, extra_body["radius"], scale, extra_body["name"], center=tuple(center)
        )

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        trajectory = _coerce_trajectory_frame(trajectory, body)

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
    ax.set_title(f"3D Trajectory ({body['name']}-Centered Inertial)")

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
    body,
    show_body,
    texture,
    additional_bodies,
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
        scale = 1.0 / body["radius"]
        unit_label = f"{body['name']} Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot central body sphere if requested
    if show_body:
        _plot_body_sphere_plotly(
            fig,
            body["radius"],
            scale,
            texture,
            sphere_resolution_lon,
            sphere_resolution_lat,
            body["name"],
        )

    for extra in additional_bodies:
        extra_body = resolve_body(extra)
        center = np.asarray(extra["position"], dtype=float) * scale
        _plot_body_sphere_plotly(
            fig,
            extra_body["radius"],
            scale,
            extra_body["texture"],
            sphere_resolution_lon,
            sphere_resolution_lat,
            extra_body["name"],
            center=tuple(center),
        )

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        trajectory = _coerce_trajectory_frame(trajectory, body)

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
        "title": f"3D Trajectory ({body['name']}-Centered Inertial)",
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


def _plot_body_sphere_matplotlib(ax, radius, scale, name, center=(0.0, 0.0, 0.0)):
    """Plot a simple solid-color sphere for a body on a matplotlib 3D axis."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * scale * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * scale * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * scale * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.6, edgecolor="none", label=name)


def _plot_body_sphere_plotly(
    fig,
    radius,
    scale,
    texture,
    n_lon=360,
    n_lat=180,
    name="Body",
    center=(0.0, 0.0, 0.0),
):
    """Plot a body sphere on a plotly 3D figure using Mesh3d with optional texture mapping."""
    # Load texture if requested
    texture_img = load_body_texture(texture)

    if texture_img is None:
        # Simple solid sphere using Surface (faster for untextured)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radius * scale * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * scale * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * scale * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Blues",
                showscale=False,
                opacity=0.7,
                name=name,
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
                x = radius * scale * np.sin(lat) * np.cos(lon)
                y = radius * scale * np.sin(lat) * np.sin(lon)
                z = radius * scale * np.cos(lat)
                vertices.append([x, y, z])

        vertices = np.array(vertices)
        vertices += np.asarray(center)
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

            # Compute average position relative to the sphere center
            avg_x = np.mean(vertices[v_indices, 0]) - center[0]
            avg_y = np.mean(vertices[v_indices, 1]) - center[1]
            avg_z = np.mean(vertices[v_indices, 2]) - center[2]

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
                name=name,
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
