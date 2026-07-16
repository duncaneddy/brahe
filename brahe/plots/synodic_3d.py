"""3D trajectory visualization in synodic (two-body rotating) frames."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.bodies import BODY_VISUALS_BY_NAIF_ID, resolve_body
from brahe.plots.trajectory_3d import (
    _normalize_trajectory_groups,
    _plot_body_sphere_matplotlib,
    _plot_body_sphere_plotly,
    _set_axes_equal,
)
from brahe._brahe import OrbitTrajectory


def plot_synodic_3d(
    trajectories,
    frame="EMR",
    reference_epoch=None,
    bodies=None,
    units="km",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=None,
    sphere_resolution_lon=360,
    sphere_resolution_lat=180,
    backend="matplotlib",
    width=None,
    height=None,
) -> object:
    """Plot 3D trajectories in a synodic (two-body rotating) frame.

    Each trajectory is converted to its ECI representation and then
    transformed per-epoch into the requested synodic frame via
    ``state_frame_to_frame``. The frame's primary and secondary bodies are
    drawn as textured spheres at their synodic-frame positions at a single
    reference epoch (the rotating frame keeps both bodies on the x-axis at
    all times, so one epoch is sufficient to place them).

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        frame (str or ReferenceFrame, optional): Synodic frame to plot in.
            Either ``'EMR'``, ``'SER'``, ``'GSE'``, a ``ReferenceFrame`` name
            string, or a ``ReferenceFrame.Synodic(origin, primary, secondary)``.
            Default: ``'EMR'``
        reference_epoch (Epoch, optional): Epoch at which the primary and
            secondary body spheres are placed. Default: the first epoch of
            the first trajectory
        bodies (list of dict, optional): Extra textured spheres to draw in
            addition to the frame's primary/secondary, each with:
            - position (array-like, length 3): Position in meters, in the
              synodic frame
            - radius (float): Radius in meters
            - texture (str, Path, or None, optional): Texture (plotly only)
            - name (str): Label used in the legend/hover text
        units (str, optional): 'm' or 'km'. Default: 'km'
        view_azimuth (float, optional): Camera azimuth angle (degrees). Default: 45.0
        view_elevation (float, optional): Camera elevation angle (degrees). Default: 30.0
        view_distance (float, optional): Camera distance multiplier. Default: 2.5 (larger = further out)
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
        ValueError: If ``frame`` is not a recognized ``ReferenceFrame`` name,
            or resolves to a ``ReferenceFrame`` that is not synodic.
        TypeError: If a trajectory is not an OrbitTrajectory object.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        prop.propagate_to(epoch + 5400.0)

        # Plot a LEO trajectory in the Earth-Moon Rotating frame
        fig = bh.plot_synodic_3d(
            [{"trajectory": prop.trajectory, "label": "LEO"}],
            frame="EMR",
            backend="matplotlib",
        )
        ```
    """
    validate_backend(backend)

    if isinstance(frame, str):
        frame = bh.ReferenceFrame.from_string(frame)
    primary_id = frame.synodic_primary
    secondary_id = frame.synodic_secondary
    if primary_id is None:
        raise ValueError(
            f"{frame} is not a synodic frame. Use EMR/SER/GSE or "
            "ReferenceFrame.Synodic(origin, primary, secondary)."
        )

    traj_groups = _normalize_trajectory_groups(trajectories)
    for group in traj_groups:
        if not isinstance(group.get("trajectory"), OrbitTrajectory):
            raise TypeError(
                "Trajectory must be an OrbitTrajectory object, got "
                f"{type(group.get('trajectory'))}"
            )

    # Transform every trajectory into the synodic frame per-epoch, using
    # the trajectory's own frame metadata (via to_eci -> GCRF -> frame).
    transformed = []
    for group in traj_groups:
        traj = group["trajectory"].to_eci()
        epochs = traj.epochs()
        states = traj.to_matrix()
        xyz = np.array(
            [
                bh.state_frame_to_frame(bh.ReferenceFrame.GCRF, frame, epc, s)[:3]
                for epc, s in zip(epochs, states)
            ]
        )
        transformed.append({**group, "positions": xyz, "epochs": epochs})

    if reference_epoch is None:
        if not transformed:
            raise ValueError(
                "reference_epoch must be provided when trajectories is empty"
            )
        reference_epoch = transformed[0]["epochs"][0]

    # Primaries placed at their synodic-frame positions at reference_epoch.
    body_entries = []
    for naif_id in (primary_id, secondary_id):
        visual = BODY_VISUALS_BY_NAIF_ID.get(naif_id)
        if visual is None:
            logger.warning(
                f"No body visual registered for NAIF ID {naif_id}, skipping sphere"
            )
            continue
        pos = bh.position_frame_to_frame(
            bh.ReferenceFrame.BodyCenteredICRF(naif_id),
            frame,
            reference_epoch,
            np.zeros(3),
        )
        body_entries.append({**visual, "position": pos})
    for extra in bodies or []:  # caller-supplied additions
        resolved = resolve_body(extra)
        body_entries.append({**resolved, "position": extra["position"]})

    scale = 1e-3 if units == "km" else 1.0
    unit_label = "km" if units == "km" else "m"

    if backend == "matplotlib":
        result = _synodic_3d_matplotlib(
            transformed,
            body_entries,
            frame,
            reference_epoch,
            scale,
            unit_label,
            view_azimuth,
            view_elevation,
        )
    else:  # plotly
        result = _synodic_3d_plotly(
            transformed,
            body_entries,
            frame,
            reference_epoch,
            scale,
            unit_label,
            view_azimuth,
            view_elevation,
            view_distance,
            sphere_resolution_lon,
            sphere_resolution_lat,
            width,
            height,
        )

    return result


def _synodic_3d_matplotlib(
    transformed,
    body_entries,
    frame,
    reference_epoch,
    scale,
    unit_label,
    view_azimuth,
    view_elevation,
):
    """Matplotlib implementation of the synodic-frame 3D trajectory plot."""
    apply_scienceplots_style()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for entry in body_entries:
        center = np.asarray(entry["position"], dtype=float) * scale
        _plot_body_sphere_matplotlib(
            ax, entry["radius"], scale, entry["name"], center=tuple(center)
        )

    for group in transformed:
        positions = group["positions"] * scale
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=group.get("color"),
            linewidth=group.get("line_width", 2.0),
            label=group.get("label"),
        )

    ax.view_init(elev=view_elevation, azim=view_azimuth)

    ax.set_xlabel(f"X ({unit_label})")
    ax.set_ylabel(f"Y ({unit_label})")
    ax.set_zlabel(f"Z ({unit_label})")
    ax.set_title(f"3D Trajectory ({frame} Frame, epoch {reference_epoch})")

    _set_axes_equal(ax)

    ax.legend()

    return fig


def _synodic_3d_plotly(
    transformed,
    body_entries,
    frame,
    reference_epoch,
    scale,
    unit_label,
    view_azimuth,
    view_elevation,
    view_distance,
    sphere_resolution_lon,
    sphere_resolution_lat,
    width,
    height,
):
    """Plotly implementation of the synodic-frame 3D trajectory plot."""
    fig = go.Figure()

    for entry in body_entries:
        center = np.asarray(entry["position"], dtype=float) * scale
        _plot_body_sphere_plotly(
            fig,
            entry["radius"],
            scale,
            entry["texture"],
            sphere_resolution_lon,
            sphere_resolution_lat,
            entry["name"],
            center=tuple(center),
        )

    for group in transformed:
        positions = group["positions"] * scale
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines",
                line=dict(color=group.get("color"), width=group.get("line_width", 2.0)),
                name=group.get("label"),
            )
        )

    if view_distance is None:
        view_distance = 2.5  # Default: zoom out to 2.5x distance

    layout_config = {
        "title": f"3D Trajectory ({frame} Frame, epoch {reference_epoch})",
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

    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    return fig


def plot_earth_moon_rotating_3d(trajectories, **kwargs):
    """Plot 3D trajectories in the Earth-Moon Rotating (EMR) frame.

    Alias for ``plot_synodic_3d(trajectories, frame=ReferenceFrame.EMR, ...)``;
    accepts the same keyword arguments.

    Args:
        trajectories: Same trajectory-group input as ``plot_synodic_3d``.
        **kwargs: Forwarded to ``plot_synodic_3d``.

    Returns:
        object: Generated figure object.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        prop.propagate_to(epoch + 5400.0)

        fig = bh.plot_earth_moon_rotating_3d(
            [{"trajectory": prop.trajectory, "label": "LEO"}], backend="matplotlib"
        )
        ```
    """
    kwargs.pop("frame", None)
    return plot_synodic_3d(trajectories, frame=bh.ReferenceFrame.EMR, **kwargs)
