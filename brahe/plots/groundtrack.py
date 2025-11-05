"""
Ground track plotting with communication cones and polygon zones.

Provides ground track visualization with per-group configuration for trajectories,
ground stations, and polygon zones.
"""

import math
import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry
import plotly.graph_objects as go
import shapefile as shp
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, is_scienceplots_available
from shapely.geometry import shape


def split_ground_track_at_antimeridian(
    lons, lats, threshold: float = 180.0
) -> List[Tuple]:
    """Split a ground track into segments at antimeridian crossings.

    When a satellite ground track crosses the antimeridian (±180° longitude), plotting
    libraries may draw an incorrect line across the entire map. This function detects
    such crossings and splits the track into separate segments that can be plotted
    individually.

    Args:
        lons (list or np.ndarray): Longitude values in degrees
        lats (list or np.ndarray): Latitude values in degrees (same length as lons)
        threshold (float, optional): Longitude jump threshold in degrees to detect
            wraparound. Default: 180.0

    Returns:
        List[Tuple]: List of (lon_segment, lat_segment) tuples, where each tuple
            contains arrays for one continuous segment

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Ground track that crosses antimeridian
        lons = [170, 175, 180, -175, -170]
        lats = [10, 15, 20, 25, 30]

        # Split into segments
        segments = bh.split_ground_track_at_antimeridian(lons, lats)

        # Plot each segment
        import matplotlib.pyplot as plt
        for lon_seg, lat_seg in segments:
            plt.plot(lon_seg, lat_seg)
        ```
    """
    lons = np.array(lons)
    lats = np.array(lats)

    if len(lons) != len(lats):
        raise ValueError("lons and lats must have the same length")

    if len(lons) == 0:
        return []

    # Find where jumps occur
    lon_diff = np.diff(lons)
    wrap_indices = np.where(np.abs(lon_diff) > threshold)[0]

    if len(wrap_indices) == 0:
        # No wraparound, return single segment
        return [(lons, lats)]

    # Split into segments
    segments = []
    start_idx = 0

    for wrap_idx in wrap_indices:
        # Add segment up to (and including) the point before the jump
        end_idx = wrap_idx + 1
        segments.append((lons[start_idx:end_idx], lats[start_idx:end_idx]))
        start_idx = end_idx

    # Add final segment
    if start_idx < len(lons):
        segments.append((lons[start_idx:], lats[start_idx:]))

    return segments


def plot_groundtrack(
    trajectories=None,
    ground_stations=None,
    zones=None,
    gs_cone_altitude=500e3,
    gs_min_elevation=10.0,
    basemap="natural_earth",
    show_borders=True,
    show_coastlines=True,
    border_width=0.5,
    show_grid=False,
    show_ticks=True,
    show_legend=False,
    extent=None,
    backend="matplotlib",
) -> object:
    """Plot ground tracks with optional ground stations and polygon zones.

    Args:
        trajectories (list of dict, optional): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - track_length (float, optional): Length of track to display
            - track_units (str, optional): Units for track_length - "orbits" or "seconds". Default: "orbits"

        ground_stations (list of dict, optional): List of ground station groups, each with:
            - stations: List of PointLocation or (lat, lon) tuples
            - color (str, optional): Station and cone color
            - alpha (float, optional): Cone transparency
            - point_size (float, optional): Station marker size
            - show_ring (bool, optional): Show outer ring
            - ring_color (str, optional): Ring color
            - ring_width (float, optional): Ring line width

        zones (list of dict, optional): List of polygon zone groups, each with:
            - zone: PolygonLocation
            - fill (bool, optional): Fill interior
            - fill_alpha (float, optional): Fill transparency
            - fill_color (str, optional): Fill color
            - edge (bool, optional): Show edge
            - edge_color (str, optional): Edge color
            - points (bool, optional): Show vertices

        gs_cone_altitude (float, optional): Assumed satellite altitude for cone calculation (m).
            Default: 500e3
        gs_min_elevation (float, optional): Minimum elevation angle (degrees). Default: 10.0
        basemap (str, optional): Basemap style - "natural_earth", "stock", or None. Default: "natural_earth"
        show_borders (bool, optional): Show country borders. Default: True
        show_coastlines (bool, optional): Show coastlines. Default: True
        border_width (float, optional): Border line width. Default: 0.5
        show_grid (bool, optional): Show lat/lon grid. Default: False
        show_ticks (bool, optional): Show lat/lon tick marks. Default: True
        show_legend (bool, optional): Show legend (plotly only). Default: False
        extent (list, optional): [lon_min, lon_max, lat_min, lat_max] to zoom. Default: None (global)
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'

    Returns:
        Generated figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create a simple LEO trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        traj = prop.propagate(epoch, epoch + 2*bh.orbital_period(oe[0]), 60.0)

        # Define ground stations
        stations_aws = [
            bh.PointLocation(np.radians(40.7128), np.radians(-74.0060), 0.0),  # NYC
            bh.PointLocation(np.radians(37.7749), np.radians(-122.4194), 0.0),  # SF
        ]

        stations_ksat = [
            bh.PointLocation(np.radians(78.2232), np.radians(15.6267), 0.0),  # Svalbard
        ]

        # Plot with per-group configuration
        fig = bh.plot_groundtrack(
            trajectories=[{"trajectory": traj, "color": "red", "track_length": 2, "track_units": "orbits"}],
            ground_stations=[
                {"stations": stations_aws, "color": "orange", "alpha": 0.3},
                {"stations": stations_ksat, "color": "blue", "alpha": 0.3},
            ],
            gs_cone_altitude=500e3,
            gs_min_elevation=10.0,
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting ground track with backend={backend}")
    logger.debug(
        f"Trajectories: {len(trajectories) if trajectories else 0}, Stations: {len(ground_stations) if ground_stations else 0}, Zones: {len(zones) if zones else 0}"
    )

    validate_backend(backend)

    # Normalize inputs to per-group configuration
    trajectory_groups = _normalize_trajectory_groups(trajectories)
    station_groups = _normalize_station_groups(ground_stations)
    zone_groups = _normalize_zone_groups(zones)
    logger.debug(
        f"Normalized: {len(trajectory_groups)} trajectory groups, {len(station_groups)} station groups, {len(zone_groups)} zone groups"
    )

    # Dispatch to backend-specific implementation
    logger.debug(f"Rendering with {backend} backend")
    if backend == "matplotlib":
        result = _groundtrack_matplotlib(
            trajectory_groups,
            station_groups,
            zone_groups,
            gs_cone_altitude,
            gs_min_elevation,
            basemap,
            show_borders,
            show_coastlines,
            border_width,
            show_grid,
            show_ticks,
            extent,
        )
    else:  # plotly
        result = _groundtrack_plotly(
            trajectory_groups,
            station_groups,
            zone_groups,
            gs_cone_altitude,
            gs_min_elevation,
            basemap,
            show_borders,
            show_coastlines,
            border_width,
            show_grid,
            show_ticks,
            show_legend,
            extent,
        )

    elapsed = time.time() - start_time
    logger.info(f"Ground track plot completed in {elapsed:.2f}s")
    return result


def _normalize_trajectory_groups(trajectories):
    """Normalize trajectory input to list of dicts with defaults."""
    defaults = {
        "color": None,
        "line_width": 2.0,
        "track_length": None,
        "track_units": "orbits",
    }

    if trajectories is None:
        return []

    # Single trajectory (not a list of dicts)
    if not isinstance(trajectories, list):
        return [{**defaults, "trajectory": trajectories}]

    if len(trajectories) == 0:
        return []

    if not isinstance(trajectories[0], dict):
        # List of trajectories without config
        return [{**defaults, "trajectory": t} for t in trajectories]

    # List of dicts - apply defaults
    return [{**defaults, **group} for group in trajectories]


def _normalize_station_groups(ground_stations):
    """Normalize ground station input to list of dicts with defaults."""
    defaults = {
        "color": None,
        "alpha": 0.3,
        "point_size": 5.0,
        "show_ring": False,
        "ring_color": None,
        "ring_width": 1.0,
    }

    if ground_stations is None:
        return []

    if len(ground_stations) == 0:
        return []

    # Check if single group (list of stations) or multiple groups (list of dicts)
    if not isinstance(ground_stations[0], dict):
        # Single group - list of PointLocation or tuples
        return [{**defaults, "stations": ground_stations}]

    # Multiple groups - apply defaults
    return [{**defaults, **group} for group in ground_stations]


def _normalize_zone_groups(zones):
    """Normalize polygon zone input to list of dicts with defaults."""
    defaults = {
        "fill": True,
        "fill_alpha": 0.3,
        "fill_color": None,
        "edge": True,
        "edge_color": None,
        "points": False,
    }

    if zones is None:
        return []

    if len(zones) == 0:
        return []

    # Check if single zone or multiple zones (list of dicts)
    if not isinstance(zones[0], dict):
        # Single zone or list of zones without config
        if not isinstance(zones, list):
            return [{**defaults, "zone": zones}]
        return [{**defaults, "zone": z} for z in zones]

    # Multiple groups - apply defaults
    return [{**defaults, **group} for group in zones]


def _groundtrack_matplotlib(
    trajectory_groups,
    station_groups,
    zone_groups,
    gs_cone_altitude,
    gs_min_elevation,
    basemap,
    show_borders,
    show_coastlines,
    border_width,
    show_grid,
    show_ticks,
    extent,
):
    """Matplotlib implementation of ground track plot."""
    # Apply scienceplots if available
    if is_scienceplots_available():
        try:
            plt.style.use(["science", "no-latex"])
        except Exception:
            pass

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # Add basemap
    if basemap == "natural_earth":
        from brahe.plots.basemap import get_natural_earth_land_shapefile

        shapefile = get_natural_earth_land_shapefile()
        ax.add_feature(
            cfeature.ShapelyFeature(
                [shapely.geometry.shape(s) for s in _load_shapefile(shapefile)],
                ccrs.PlateCarree(),
                facecolor="lightgray",
                edgecolor="none",
            )
        )
    elif basemap == "stock":
        ax.stock_img()

    # Add features
    if show_coastlines:
        ax.coastlines(linewidth=border_width)
    if show_borders:
        ax.add_feature(cfeature.BORDERS, linewidth=border_width)
    if show_grid:
        ax.gridlines(draw_labels=show_ticks)

    # Hide axes spines for minimal border
    if border_width == 0:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Plot ground stations with cones
    for group in station_groups:
        _plot_station_group_matplotlib(ax, group, gs_cone_altitude, gs_min_elevation)

    # Plot polygon zones
    for group in zone_groups:
        _plot_zone_group_matplotlib(ax, group)

    # Plot trajectories
    for group in trajectory_groups:
        _plot_trajectory_group_matplotlib(ax, group)

    return fig


def _groundtrack_plotly(
    trajectory_groups,
    station_groups,
    zone_groups,
    gs_cone_altitude,
    gs_min_elevation,
    basemap,
    show_borders,
    show_coastlines,
    border_width,
    show_grid,
    show_ticks,
    show_legend,
    extent,
):
    """Plotly implementation of ground track plot."""
    fig = go.Figure()

    # Configure geo layout
    geo_config = {
        "projection_type": "equirectangular",
        "showcoastlines": show_coastlines,
        "coastlinecolor": "black",
        "coastlinewidth": border_width,
        "showland": basemap is not None,
        "landcolor": "lightgray" if basemap == "natural_earth" else "tan",
        "showcountries": show_borders,
        "countrycolor": "black",
        "countrywidth": border_width,
        "showlakes": False,
        "showrivers": False,
    }

    if show_grid:
        geo_config["lataxis"] = {"showgrid": True}
        geo_config["lonaxis"] = {"showgrid": True}

    if extent is not None:
        geo_config["lonaxis"] = {"range": [extent[0], extent[1]]}
        geo_config["lataxis"] = {"range": [extent[2], extent[3]]}

    fig.update_geos(**geo_config)

    # Plot ground stations (simplified - full implementation would include cones)
    for group in station_groups:
        _plot_station_group_plotly(
            fig, group, gs_cone_altitude, gs_min_elevation, show_legend
        )

    # Plot polygon zones
    for group in zone_groups:
        _plot_zone_group_plotly(fig, group, show_legend)

    # Plot trajectories
    for group in trajectory_groups:
        _plot_trajectory_group_plotly(fig, group, show_legend)

    return fig


# Helper functions


def _compute_communication_cone_radius(elevation_deg, altitude_m):
    """Compute the ground range of a communication cone.

    Uses spherical geometry to compute the maximum angular distance from
    a ground station to the satellite horizon.

    Args:
        elevation_deg: Minimum elevation angle in degrees
        altitude_m: Satellite altitude in meters

    Returns:
        float: Angular radius in radians
    """
    ele_rad = math.radians(elevation_deg)
    rho = math.asin(bh.R_EARTH / (bh.R_EARTH + altitude_m))
    eta = math.asin(math.cos(ele_rad) * math.sin(rho))
    lam = math.pi / 2.0 - eta - ele_rad
    return lam


def _color_to_rgba(color, alpha):
    """Convert a color (name, hex, or RGB tuple) to RGBA string for plotly.

    Args:
        color: Color as string name, hex code, or RGB/RGBA tuple
        alpha: Alpha transparency value (0.0 to 1.0)

    Returns:
        str: RGBA color string like 'rgba(255, 0, 0, 0.5)'
    """
    import matplotlib.colors as mcolors

    # Convert color to RGB
    if isinstance(color, str):
        # Handle named colors or hex codes
        rgb = mcolors.to_rgb(color)
    elif isinstance(color, (list, tuple)):
        # Already RGB or RGBA
        rgb = color[:3]
    else:
        # Default to blue if unknown format
        rgb = (0, 0, 1)

    # Convert to 0-255 range and create RGBA string
    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _plot_station_group_matplotlib(ax, group, gs_cone_altitude, gs_min_elevation):
    """Plot a group of ground stations with communication cones (matplotlib)."""
    import cartopy.crs as ccrs
    from cartopy.geodesic import Geodesic
    import shapely.geometry

    stations = group.get("stations", [])
    color = group.get("color")
    alpha = group.get("alpha", 0.3)
    point_size = group.get("point_size", 5.0)
    show_ring = group.get("show_ring", True)
    ring_color = group.get("ring_color") or color
    ring_width = group.get("ring_width", 1.0)

    # Compute cone radius
    cone_radius_rad = _compute_communication_cone_radius(
        gs_min_elevation, gs_cone_altitude
    )
    cone_radius_m = cone_radius_rad * bh.R_EARTH

    for station in stations:
        # Extract lat/lon
        if hasattr(station, "latitude") and hasattr(station, "longitude"):
            lat_deg = math.degrees(station.latitude(bh.AngleFormat.RADIANS))
            lon_deg = math.degrees(station.longitude(bh.AngleFormat.RADIANS))
        else:
            lat_deg, lon_deg = station[0], station[1]
            if abs(lat_deg) > math.pi or abs(lon_deg) > math.pi:
                # Already in degrees
                pass
            else:
                # Convert from radians
                lat_deg = math.degrees(lat_deg)
                lon_deg = math.degrees(lon_deg)

        # Plot station point
        ax.plot(
            lon_deg,
            lat_deg,
            marker="o",
            markersize=point_size,
            color=color,
            transform=ccrs.Geodetic(),
            zorder=10,
        )

        # Plot communication cone
        circle_points = Geodesic().circle(
            lon=lon_deg,
            lat=lat_deg,
            radius=cone_radius_m,
            n_samples=100,
            endpoint=False,
        )
        geom = shapely.geometry.Polygon(circle_points)

        ax.add_geometries(
            [geom],
            crs=ccrs.Geodetic(),
            facecolor=color,
            alpha=alpha,
            edgecolor="none",
            linewidth=0,
            zorder=5,
        )

        # Plot outer ring if requested
        if show_ring:
            # Extract coordinates for ring
            coords = np.array(circle_points)
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color=ring_color,
                linewidth=ring_width,
                transform=ccrs.Geodetic(),
                zorder=6,
            )


def _plot_zone_group_matplotlib(ax, group):
    """Plot a polygon zone group (matplotlib)."""
    import cartopy.crs as ccrs
    import shapely.geometry

    zone = group.get("zone")
    fill = group.get("fill", True)
    fill_alpha = group.get("fill_alpha", 0.3)
    fill_color = group.get("fill_color")
    edge = group.get("edge", True)
    edge_color = group.get("edge_color")
    points = group.get("points", False)

    if zone is None:
        return

    # Extract vertices from PolygonLocation
    if hasattr(zone, "vertices"):
        vertices = zone.vertices()
    else:
        vertices = zone

    # Convert to lat/lon in degrees
    lats = []
    lons = []
    for vertex in vertices:
        if hasattr(vertex, "latitude") and hasattr(vertex, "longitude"):
            lat = math.degrees(vertex.latitude(bh.AngleFormat.RADIANS))
            lon = math.degrees(vertex.longitude(bh.AngleFormat.RADIANS))
        else:
            lat, lon = vertex[0], vertex[1]
            if abs(lat) <= math.pi and abs(lon) <= math.pi:
                lat = math.degrees(lat)
                lon = math.degrees(lon)
        lats.append(lat)
        lons.append(lon)

    # Create polygon
    coords = list(zip(lons, lats))
    geom = shapely.geometry.Polygon(coords)

    # Plot filled polygon
    if fill:
        ax.add_geometries(
            [geom],
            crs=ccrs.Geodetic(),
            facecolor=fill_color,
            alpha=fill_alpha,
            edgecolor="none",
            linewidth=0,
            zorder=4,
        )

    # Plot edge
    if edge:
        lons_closed = lons + [lons[0]]
        lats_closed = lats + [lats[0]]
        ax.plot(
            lons_closed,
            lats_closed,
            color=edge_color,
            linewidth=1.0,
            transform=ccrs.Geodetic(),
            zorder=5,
        )

    # Plot points
    if points:
        ax.plot(
            lons,
            lats,
            "o",
            color=edge_color or fill_color,
            markersize=3,
            transform=ccrs.Geodetic(),
            zorder=6,
        )


def _filter_track(states, epochs, track_length, track_units):
    """Filter trajectory to show only the most recent track_length.

    Args:
        states: List of state vectors
        epochs: List of Epoch objects
        track_length: Length of track to display
        track_units: "orbits" or "seconds"

    Returns:
        (filtered_states, filtered_epochs): Filtered lists
    """
    if len(states) == 0 or track_length is None or track_length <= 0:
        return states, epochs

    # Get the last epoch (most recent time)
    last_epoch = epochs[-1]

    if track_units == "orbits":
        # Calculate orbital period from the last state
        last_state = states[-1]
        period = bh.orbital_period_from_state(last_state, bh.GM_EARTH)

        # Convert orbits to seconds
        duration = track_length * period
    elif track_units == "seconds":
        duration = track_length
    else:
        # Unknown units, return full trajectory
        return states, epochs

    # Find the cutoff epoch
    cutoff_epoch = last_epoch - duration

    # Filter states and epochs
    filtered_states = []
    filtered_epochs = []
    for i, epoch in enumerate(epochs):
        if epoch >= cutoff_epoch:
            filtered_states.append(states[i])
            filtered_epochs.append(epoch)

    return filtered_states, filtered_epochs


def _plot_trajectory_group_matplotlib(ax, group):
    """Plot a trajectory group (matplotlib)."""
    import cartopy.crs as ccrs

    trajectory = group.get("trajectory")
    color = group.get("color")
    line_width = group.get("line_width", 2.0)
    track_length = group.get("track_length")
    track_units = group.get("track_units", "orbits")

    if trajectory is None:
        return

    # Extract ECI states from trajectory
    if hasattr(trajectory, "to_matrix"):
        # OrbitTrajectory - use to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        epochs = trajectory.epochs()

        # Apply track filtering if requested
        if track_length is not None and track_length > 0:
            states, epochs = _filter_track(states, epochs, track_length, track_units)

        lats = []
        lons = []
        for i, state in enumerate(states):
            epoch = epochs[i]
            # Convert ECI to ECEF
            ecef_state = bh.state_eci_to_ecef(epoch, state)
            # Convert ECEF to geodetic (returns [lon, lat, alt])
            lon, lat, alt = bh.position_ecef_to_geodetic(
                ecef_state[:3], bh.AngleFormat.RADIANS
            )
            lats.append(math.degrees(lat))
            lons.append(math.degrees(lon))

    elif isinstance(trajectory, np.ndarray):
        # Assume already lat/lon or ECI positions
        if trajectory.shape[1] >= 2:
            # Assume [lat, lon] or first two columns are lat/lon
            lats = trajectory[:, 0]
            lons = trajectory[:, 1]
            # Check if in radians
            if np.max(np.abs(lats)) <= np.pi:
                lats = np.degrees(lats)
                lons = np.degrees(lons)
        else:
            return
    else:
        return

    # Plot ground track
    ax.plot(
        lons,
        lats,
        color=color,
        linewidth=line_width,
        transform=ccrs.Geodetic(),
        zorder=7,
    )


def _plot_station_group_plotly(
    fig, group, gs_cone_altitude, gs_min_elevation, show_legend
):
    """Plot a group of ground stations with communication cones (plotly)."""
    from cartopy.geodesic import Geodesic

    stations = group.get("stations", [])
    color = group.get("color", "blue")
    alpha = group.get("alpha", 0.3)
    point_size = group.get("point_size", 8)

    # Compute cone radius
    cone_radius_rad = _compute_communication_cone_radius(
        gs_min_elevation, gs_cone_altitude
    )
    cone_radius_m = cone_radius_rad * bh.R_EARTH

    # Collect station markers
    station_lats = []
    station_lons = []

    for station in stations:
        # Extract lat/lon
        if hasattr(station, "latitude") and hasattr(station, "longitude"):
            lat_deg = math.degrees(station.latitude(bh.AngleFormat.RADIANS))
            lon_deg = math.degrees(station.longitude(bh.AngleFormat.RADIANS))
        else:
            lat_deg, lon_deg = station[0], station[1]
            if abs(lat_deg) <= math.pi and abs(lon_deg) <= math.pi:
                lat_deg = math.degrees(lat_deg)
                lon_deg = math.degrees(lon_deg)

        station_lats.append(lat_deg)
        station_lons.append(lon_deg)

        # Plot communication cone for this station
        circle_points = Geodesic().circle(
            lon=lon_deg,
            lat=lat_deg,
            radius=cone_radius_m,
            n_samples=100,
            endpoint=False,
        )

        # Convert circle points to lat/lon lists
        cone_lons = [pt[0] for pt in circle_points]
        cone_lats = [pt[1] for pt in circle_points]

        # Reverse the order to fix winding direction for Plotly fill
        # Plotly's fill="toself" is sensitive to winding order
        cone_lons = cone_lons[::-1]
        cone_lats = cone_lats[::-1]

        # Close the polygon
        cone_lons_closed = cone_lons + [cone_lons[0]]
        cone_lats_closed = cone_lats + [cone_lats[0]]

        # Convert color to RGBA for alpha transparency
        rgba_color = _color_to_rgba(color, alpha)

        # For closed circles, don't split at antimeridian - Plotly handles it
        # Only check if circle extends beyond valid lat/lon ranges
        max_lat = max(cone_lats_closed)
        min_lat = min(cone_lats_closed)

        # Skip circles that extend to extreme latitudes (polar regions)
        # These can cause rendering artifacts
        if max_lat < 85 and min_lat > -85:
            fig.add_trace(
                go.Scattergeo(
                    lat=cone_lats_closed,
                    lon=cone_lons_closed,
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=rgba_color,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Plot all station markers as a single trace
    if station_lats:
        fig.add_trace(
            go.Scattergeo(
                lat=station_lats,
                lon=station_lons,
                mode="markers",
                marker=dict(size=point_size, color=color),
                name="Ground Stations",
                showlegend=show_legend,
            )
        )


def _plot_zone_group_plotly(fig, group, show_legend):
    """Plot a polygon zone group (plotly)."""
    zone = group.get("zone")
    fill_color = group.get("fill_color", "blue")
    fill_alpha = group.get("fill_alpha", 0.3)
    edge = group.get("edge", True)
    edge_color = group.get("edge_color", fill_color)

    if zone is None:
        return

    # Extract vertices
    if hasattr(zone, "vertices"):
        vertices = zone.vertices
    else:
        vertices = zone

    lats = []
    lons = []
    for vertex in vertices:
        if hasattr(vertex, "latitude") and hasattr(vertex, "longitude"):
            lat = math.degrees(vertex.latitude(bh.AngleFormat.RADIANS))
            lon = math.degrees(vertex.longitude(bh.AngleFormat.RADIANS))
        else:
            # vertices are [lon, lat, alt] tuples
            lon, lat = vertex[0], vertex[1]
            if abs(lat) <= math.pi and abs(lon) <= math.pi:
                lat = math.degrees(lat)
                lon = math.degrees(lon)
        lats.append(lat)
        lons.append(lon)

    # Reverse the order to fix winding direction for Plotly fill
    # Plotly's fill="toself" is sensitive to winding order
    lons = lons[::-1]
    lats = lats[::-1]

    # Split at antimeridian if needed
    segments = split_ground_track_at_antimeridian(lons, lats)

    # Plot each segment
    for seg_idx, (seg_lons, seg_lats) in enumerate(segments):
        # Close the polygon
        seg_lons_closed = list(seg_lons) + [seg_lons[0]]
        seg_lats_closed = list(seg_lats) + [seg_lats[0]]

        # Convert fill color to RGBA
        rgba_fill = _color_to_rgba(fill_color, fill_alpha)

        fig.add_trace(
            go.Scattergeo(
                lat=seg_lats_closed,
                lon=seg_lons_closed,
                mode="lines",
                line=dict(
                    color=edge_color if edge else rgba_fill, width=2 if edge else 0
                ),
                fill="toself",
                fillcolor=rgba_fill,
                name="Zone" if seg_idx == 0 else None,  # Only show in legend once
                showlegend=show_legend and seg_idx == 0,
            )
        )


def _plot_trajectory_group_plotly(fig, group, show_legend):
    """Plot a trajectory group (plotly)."""
    trajectory = group.get("trajectory")
    color = group.get("color", "red")
    line_width = group.get("line_width", 2.0)
    track_length = group.get("track_length")
    track_units = group.get("track_units", "orbits")

    if trajectory is None:
        return

    # Extract lat/lon
    if hasattr(trajectory, "to_matrix"):
        # OrbitTrajectory - use to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        epochs = trajectory.epochs()

        # Apply track filtering if requested
        if track_length is not None and track_length > 0:
            states, epochs = _filter_track(states, epochs, track_length, track_units)

        lats = []
        lons = []
        for i, state in enumerate(states):
            epoch = epochs[i]
            ecef_state = bh.state_eci_to_ecef(epoch, state)
            # Convert ECEF to geodetic (returns [lon, lat, alt])
            lon, lat, alt = bh.position_ecef_to_geodetic(
                ecef_state[:3], bh.AngleFormat.RADIANS
            )
            lats.append(math.degrees(lat))
            lons.append(math.degrees(lon))

    elif isinstance(trajectory, np.ndarray):
        if trajectory.shape[1] >= 2:
            lats = trajectory[:, 0]
            lons = trajectory[:, 1]
            if np.max(np.abs(lats)) <= np.pi:
                lats = np.degrees(lats)
                lons = np.degrees(lons)
        else:
            return
    else:
        return

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(color=color, width=line_width),
            name="Ground Track",
            showlegend=show_legend,
        )
    )


def _load_shapefile(shapefile_path):
    """Load shapefile geometries.

    Args:
        shapefile_path: Path to .shp file

    Returns:
        list: List of shapely geometries
    """

    sf = shp.Reader(shapefile_path)
    return [shape(record.shape.__geo_interface__) for record in sf.shapeRecords()]
