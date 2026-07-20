# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Star-Field Sensor Simulation

Animates a star tracker's boresight sweeping along the velocity vector of a
sun-synchronous low Earth orbit, showing which Hipparcos stars fall inside a
30 deg full-angle field of view over one orbital period.
"""

import os
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go

import brahe as bh

SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:all]
# --8<-- [start:scenario]
# Sensor and star-catalog configuration
HALF_ANGLE_DEG = 15.0  # Sensor half-angle (30 deg full field of view)
STAR_MAG_LIMIT = 5.2  # Naked-eye-bright Hipparcos stars
STAR_SHELL_RADIUS = 3.0 * bh.R_EARTH  # Display radius for the star sphere
CONE_LENGTH = 3000e3  # FOV cone visualization length, meters
CONE_SEGMENTS = 24  # Cone base polygon resolution
PROPAGATION_STEP = (
    20.0  # Propagation/animation frame step, seconds (smaller = smoother)
)

epoch = bh.Epoch.from_datetime(2026, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.4, 0.0, 0.0, 0.0])
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
period = bh.orbital_period(oe[0])

# Propagate one orbital period with the Keplerian propagator, one animation
# frame per PROPAGATION_STEP (a fine step keeps the boresight/star motion smooth)
prop = bh.KeplerianPropagator.from_eci(epoch, state0, PROPAGATION_STEP)
prop.propagate_to(epoch + period)
traj = prop.trajectory

states = traj.to_matrix()  # [n_frames, 6]: ECI position (m) and velocity (m/s)
positions = states[:, 0:3]
velocities = states[:, 3:6]
n_frames = positions.shape[0]

# The sensor boresight points along the velocity vector (along-track)
boresights = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
# --8<-- [end:scenario]

# --8<-- [start:catalog]
# Load the Hipparcos catalog and keep naked-eye-bright stars
hipparcos = bh.datasets.star_catalogs.get_hipparcos()
bright_stars = hipparcos.filter_by_magnitude(STAR_MAG_LIMIT)
star_records = bright_stars.records()

star_names = [record.name() or record.id() for record in star_records]
star_vmags = np.array([record.vmag for record in star_records])
star_unit_vectors = np.array([record.unit_vector() for record in star_records])
star_positions = star_unit_vectors * STAR_SHELL_RADIUS
# --8<-- [end:catalog]

# --8<-- [start:visibility]
# A star is inside the field of view when the angle between the boresight
# and the star direction is smaller than the sensor half-angle
cos_half_angle = np.cos(np.radians(HALF_ANGLE_DEG))
star_boresight_dot = star_unit_vectors @ boresights.T  # [n_stars, n_frames]
visible_mask = star_boresight_dot > cos_half_angle

visible_counts = visible_mask.sum(axis=0)
print(f"Loaded {len(star_records)} stars brighter than Vmag {STAR_MAG_LIMIT}")
print(f"Frames: {n_frames}")
print(
    "Visible stars per frame: "
    f"min={int(visible_counts.min())}, "
    f"median={int(np.median(visible_counts))}, "
    f"max={int(visible_counts.max())}"
)
# --8<-- [end:visibility]

# Display quantities (km) for scene axes; visibility/geometry above stays in
# the library's native SI units (meters)
positions_km = positions * 1e-3
star_positions_km = star_positions * 1e-3
cone_length_km = CONE_LENGTH * 1e-3
star_shell_radius_km = STAR_SHELL_RADIUS * 1e-3


# --8<-- [start:cone]
def boresight_basis(boresight):
    """Build an orthonormal (u, v) basis spanning the plane perpendicular
    to the boresight direction.
    """
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, boresight)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(boresight, reference)
    u /= np.linalg.norm(u)
    v = np.cross(boresight, u)
    return u, v


def fov_cone_mesh(apex, boresight, length, half_angle_deg, n_segments):
    """Build FOV cone mesh vertices and triangle faces for `go.Mesh3d`.

    The cone apex sits at the satellite position and its axis is aligned
    with the boresight direction; vertex 0 is the apex and vertices
    1..n_segments form the base circle.
    """
    u, v = boresight_basis(boresight)

    radius = length * np.tan(np.radians(half_angle_deg))
    theta = np.linspace(0.0, 2 * np.pi, n_segments, endpoint=False)
    base_center = apex + boresight * length
    base = (
        base_center
        + radius * np.outer(np.cos(theta), u)
        + radius * np.outer(np.sin(theta), v)
    )

    vertices = np.vstack([apex, base])
    idx = np.arange(n_segments)
    i_faces = np.zeros(n_segments, dtype=int)
    j_faces = 1 + idx
    k_faces = 1 + (idx + 1) % n_segments
    return vertices, i_faces, j_faces, k_faces


def marker_sizes(vmags):
    """Map visual magnitude to marker size so brighter stars render larger."""
    return np.clip(9.0 - 1.2 * vmags, 3.0, 9.0)


# --8<-- [end:cone]


# --8<-- [start:sensor_frame]
def sensor_frame_angles(star_unit_vectors, boresight):
    """Project star unit vectors into boresight-relative angular offsets.

    Decomposes each star's angular separation from the boresight into a
    cross-boresight (x) and an "elevation-like" (y) component, both in
    degrees, using the same (u, v) basis as the FOV cone. A constant
    angular separation from the boresight (the FOV boundary) is therefore
    an exact circle of that radius in (x, y).
    """
    u, v = boresight_basis(boresight)
    cos_theta = np.clip(star_unit_vectors @ boresight, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    phi = np.arctan2(star_unit_vectors @ v, star_unit_vectors @ u)
    return theta_deg * np.cos(phi), theta_deg * np.sin(phi)


def animation_controls(n_frames):
    """Shared Play/Pause buttons + scrub slider for the animated figures."""
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.05,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=40, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                    ],
                ),
            ],
        )
    ]
    sliders = [
        dict(
            active=0,
            x=0.15,
            len=0.85,
            currentvalue=dict(visible=False),
            ticklen=0,
            steps=[
                dict(
                    label="",
                    method="animate",
                    args=[
                        [str(k)],
                        dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                    ],
                )
                for k in range(n_frames)
            ],
        )
    ]
    return updatemenus, sliders


# --8<-- [end:sensor_frame]


# --8<-- [start:figure]
def create_figure(theme):
    colors = get_theme_colors(theme)
    earth_color = "#a9c6e8" if theme == "light" else "#2f4f6f"
    # Stars sit on the solid (axis-free) background: the pale gold reads well on
    # the dark theme, but the white light-theme background needs a darker gold.
    star_color = colors["quaternary"] if theme == "dark" else "#8a6d1f"

    r_earth_km = bh.R_EARTH * 1e-3
    lon = np.linspace(0, 2 * np.pi, 60)
    lat = np.linspace(0, np.pi, 30)
    earth_x = r_earth_km * np.outer(np.cos(lon), np.sin(lat))
    earth_y = r_earth_km * np.outer(np.sin(lon), np.sin(lat))
    earth_z = r_earth_km * np.outer(np.ones_like(lon), np.cos(lat))

    fig = go.Figure()

    # Trace 0: Earth sphere (static)
    fig.add_trace(
        go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            colorscale=[[0, earth_color], [1, earth_color]],
            showscale=False,
            showlegend=True,
            opacity=0.55,
            name="Earth",
            hoverinfo="skip",
        )
    )

    # Trace 1: full orbit path (static)
    fig.add_trace(
        go.Scatter3d(
            x=positions_km[:, 0],
            y=positions_km[:, 1],
            z=positions_km[:, 2],
            mode="lines",
            line=dict(color=colors["primary"], width=3),
            name="Orbit",
            hoverinfo="skip",
        )
    )

    # Trace 2: satellite marker (animated)
    fig.add_trace(
        go.Scatter3d(
            x=[positions_km[0, 0]],
            y=[positions_km[0, 1]],
            z=[positions_km[0, 2]],
            mode="markers",
            marker=dict(size=5, color=colors["secondary"]),
            name="Satellite",
            hoverinfo="skip",
        )
    )

    # Trace 3: FOV cone (animated)
    cone_vertices, cone_i, cone_j, cone_k = fov_cone_mesh(
        positions_km[0], boresights[0], cone_length_km, HALF_ANGLE_DEG, CONE_SEGMENTS
    )
    fig.add_trace(
        go.Mesh3d(
            x=cone_vertices[:, 0],
            y=cone_vertices[:, 1],
            z=cone_vertices[:, 2],
            i=cone_i,
            j=cone_j,
            k=cone_k,
            color=colors["accent"],
            opacity=0.25,
            flatshading=True,
            showlegend=True,
            name="Field of View",
            hoverinfo="skip",
        )
    )

    # Trace 4: visible stars (animated)
    frame0_idx = np.nonzero(visible_mask[:, 0])[0]
    fig.add_trace(
        go.Scatter3d(
            x=star_positions_km[frame0_idx, 0],
            y=star_positions_km[frame0_idx, 1],
            z=star_positions_km[frame0_idx, 2],
            mode="markers",
            marker=dict(
                size=marker_sizes(star_vmags[frame0_idx]),
                color=star_color,
            ),
            text=[star_names[i] for i in frame0_idx],
            hovertemplate="%{text}<extra></extra>",
            name="Visible Stars",
        )
    )

    # Animation frames: only the satellite, cone, and visible-star traces change
    frames = []
    for k in range(n_frames):
        idx = np.nonzero(visible_mask[:, k])[0]
        cone_vertices, cone_i, cone_j, cone_k = fov_cone_mesh(
            positions_km[k],
            boresights[k],
            cone_length_km,
            HALF_ANGLE_DEG,
            CONE_SEGMENTS,
        )
        frames.append(
            go.Frame(
                name=str(k),
                traces=[2, 3, 4],
                data=[
                    go.Scatter3d(
                        x=[positions_km[k, 0]],
                        y=[positions_km[k, 1]],
                        z=[positions_km[k, 2]],
                    ),
                    go.Mesh3d(
                        x=cone_vertices[:, 0],
                        y=cone_vertices[:, 1],
                        z=cone_vertices[:, 2],
                        i=cone_i,
                        j=cone_j,
                        k=cone_k,
                    ),
                    go.Scatter3d(
                        x=star_positions_km[idx, 0],
                        y=star_positions_km[idx, 1],
                        z=star_positions_km[idx, 2],
                        marker=dict(size=marker_sizes(star_vmags[idx])),
                        text=[star_names[i] for i in idx],
                    ),
                ],
            )
        )
    fig.frames = frames

    updatemenus, sliders = animation_controls(n_frames)
    axis_range = [-1.05 * star_shell_radius_km, 1.05 * star_shell_radius_km]
    fig.update_layout(
        title="Star-Field Sensor Simulation (SSO, One Orbital Period)",
        scene=dict(
            # Hide the axes, ticks, labels, and grid panes so the stars sit on
            # a clean solid background rather than the default light-grey cube
            xaxis=dict(visible=False, range=axis_range),
            yaxis=dict(visible=False, range=axis_range),
            zaxis=dict(visible=False, range=axis_range),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    return fig


# --8<-- [end:figure]


# --8<-- [start:sensor_view_figure]
def create_sensor_view_figure(theme):
    """2D sensor-frame view: only the visible stars, in boresight-relative
    angular coordinates, with the fixed FOV boundary drawn as a circle.
    """
    colors = get_theme_colors(theme)
    star_color = colors["quaternary"] if theme == "dark" else "#8a6d1f"

    fig = go.Figure()

    # Trace 0: FOV boundary (static) - a constant angular separation from
    # the boresight is an exact circle of this radius in (x, y)
    boundary_theta = np.linspace(0.0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=HALF_ANGLE_DEG * np.cos(boundary_theta),
            y=HALF_ANGLE_DEG * np.sin(boundary_theta),
            mode="lines",
            line=dict(color=colors["accent"], width=2, dash="dot"),
            name=f"Field of View ({HALF_ANGLE_DEG:.0f}°)",
            hoverinfo="skip",
        )
    )

    # Trace 1: visible stars (animated)
    frame0_idx = np.nonzero(visible_mask[:, 0])[0]
    x0, y0 = sensor_frame_angles(star_unit_vectors[frame0_idx], boresights[0])
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=y0,
            mode="markers",
            marker=dict(
                size=marker_sizes(star_vmags[frame0_idx]),
                color=star_color,
            ),
            text=[star_names[i] for i in frame0_idx],
            hovertemplate="%{text}<extra></extra>",
            name="Visible Stars",
        )
    )

    # Animation frames: only the visible-star trace changes
    frames = []
    for k in range(n_frames):
        idx = np.nonzero(visible_mask[:, k])[0]
        x_k, y_k = sensor_frame_angles(star_unit_vectors[idx], boresights[k])
        frames.append(
            go.Frame(
                name=str(k),
                traces=[1],
                data=[
                    go.Scatter(
                        x=x_k,
                        y=y_k,
                        marker=dict(size=marker_sizes(star_vmags[idx])),
                        text=[star_names[i] for i in idx],
                    )
                ],
            )
        )
    fig.frames = frames

    updatemenus, sliders = animation_controls(n_frames)
    axis_range = [-1.05 * HALF_ANGLE_DEG, 1.05 * HALF_ANGLE_DEG]
    fig.update_layout(
        title="Star-Field Sensor View (SSO, One Orbital Period)",
        xaxis=dict(
            title="Cross-Boresight Offset (deg)", range=axis_range, zeroline=False
        ),
        yaxis=dict(
            title="Elevation-Like Offset (deg)",
            range=axis_range,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    return fig


# --8<-- [end:sensor_view_figure]
# --8<-- [end:all]

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import get_theme_colors, save_themed_html  # noqa: E402

light_path, dark_path = save_themed_html(
    create_figure, OUTDIR / SCRIPT_NAME, auto_play=True
)
print(f"Star field simulation (light) saved to: {light_path}")
print(f"Star field simulation (dark) saved to: {dark_path}")

sensor_light_path, sensor_dark_path = save_themed_html(
    create_sensor_view_figure, OUTDIR / "star_field_sensor_view", auto_play=True
)
print(f"Star field sensor view (light) saved to: {sensor_light_path}")
print(f"Star field sensor view (dark) saved to: {sensor_dark_path}")
