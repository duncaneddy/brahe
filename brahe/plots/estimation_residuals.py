"""
Measurement residual visualization — array API and solver API.

Provides functions to plot measurement residuals from raw numpy arrays or
directly from solved estimator objects, supporting both matplotlib and plotly
backends.
"""

import time as _time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.estimation_common import (
    compute_time_axis,
    extract_residuals,
    resolve_colors,
    resolve_labels,
    compute_grid_layout,
)


# =============================================================================
# Public API — array API: single-panel residual scatter
# =============================================================================


def plot_measurement_residual_from_arrays(
    times,
    residuals,
    labels=None,
    colors=None,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot measurement residuals as a scatter overlay from raw numpy arrays.

    Each column of ``residuals`` is one measurement component and is drawn as
    a separate scatter series with a horizontal zero-reference line.

    Args:
        times (numpy.ndarray): 1-D array of time values, length N.
        residuals (numpy.ndarray): 2-D array of shape ``(N, n_components)``.
        labels (list[str] or None): Label per component.  Defaults to
            "Series 0", "Series 1", …
        colors (list[str] or None): Colour per component.  Defaults to the
            internal colour cycle.
        time_label (str): X-axis label.  Default: "Time [s]".
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``, ``legend_loc``, ``dark_mode``,
            ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual_from_arrays backend={backend}")
    validate_backend(backend)

    n_components = residuals.shape[1] if residuals.ndim == 2 else 1
    resolved_colors = resolve_colors(n_components, colors)
    resolved_labels = resolve_labels(n_components, labels)
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _residual_single_matplotlib(
            times, residuals, resolved_labels, resolved_colors, time_label, cfg
        )
    else:
        fig = _residual_single_plotly(
            times, residuals, resolved_labels, resolved_colors, time_label, cfg
        )

    logger.info(
        f"plot_measurement_residual_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — array API: grid residual scatter
# =============================================================================


def plot_measurement_residual_grid_from_arrays(
    times,
    residuals,
    labels=None,
    colors=None,
    ncols=3,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot measurement residuals in a subplot grid from raw numpy arrays.

    One subplot is created per measurement component.  Each subplot contains
    scatter markers for that component plus a zero-reference line.  Unused
    subplots (when n_components is not a multiple of ncols) are hidden.

    Args:
        times (numpy.ndarray): 1-D array of time values, length N.
        residuals (numpy.ndarray): 2-D array of shape ``(N, n_components)``.
        labels (list[str] or None): Title/label per component.  Defaults to
            "Series 0", "Series 1", …
        colors (list[str] or None): Colour per component.  Defaults to the
            internal colour cycle.
        ncols (int): Number of subplot columns.  Default: 3.
        time_label (str): X-axis label.  Default: "Time [s]".
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``, ``legend_loc``, ``dark_mode``,
            ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual_grid_from_arrays backend={backend}")
    validate_backend(backend)

    n_components = residuals.shape[1] if residuals.ndim == 2 else 1
    resolved_colors = resolve_colors(n_components, colors)
    resolved_labels = resolve_labels(n_components, labels)
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _residual_grid_matplotlib(
            times,
            residuals,
            resolved_labels,
            resolved_colors,
            n_components,
            ncols,
            time_label,
            cfg,
        )
    else:
        fig = _residual_grid_plotly(
            times,
            residuals,
            resolved_labels,
            resolved_colors,
            n_components,
            ncols,
            time_label,
            cfg,
        )

    logger.info(
        f"plot_measurement_residual_grid_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — array API: RMS line
# =============================================================================


def plot_measurement_residual_rms_from_arrays(
    times,
    residuals,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot the per-epoch RMS of measurement residuals from raw numpy arrays.

    Computes ``rms[i] = sqrt(mean(residuals[i, :]**2))`` and draws a single
    line plot.

    Args:
        times (numpy.ndarray): 1-D array of time values, length N.
        residuals (numpy.ndarray): 2-D array of shape ``(N, n_components)``.
        time_label (str): X-axis label.  Default: "Time [s]".
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``, ``legend_loc``, ``dark_mode``,
            ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual_rms_from_arrays backend={backend}")
    validate_backend(backend)

    rms = np.sqrt(np.mean(residuals**2, axis=1))
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _residual_rms_matplotlib(times, rms, time_label, cfg)
    else:
        fig = _residual_rms_plotly(times, rms, time_label, cfg)

    logger.info(
        f"plot_measurement_residual_rms_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — solver API: single-panel residual scatter
# =============================================================================


def plot_measurement_residual(
    solver,
    iteration=-1,
    residual_type="postfit",
    model_name=None,
    labels=None,
    colors=None,
    time_units="seconds",
    orbital_period=None,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot measurement residuals from a solved estimator.

    Extracts residuals from the solver and delegates to
    :func:`plot_measurement_residual_from_arrays`.  When
    ``residual_type="both"``, prefit and postfit residuals are overlaid on the
    same axes using different markers and alpha levels.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        iteration (int): Iteration index for BLS (default -1, last).
            Ignored for sequential filters.
        residual_type (str): "prefit", "postfit", or "both".
            Default: "postfit".
        model_name (str or None): Filter residuals by measurement model name.
            None means include all models.  Default: None.
        labels (list[str] or None): Label per residual component.  Default:
            generated labels.
        colors (list[str] or None): Colour per component.  Default: colour
            cycle.
        time_units (str or callable): Time axis units.  Default: "seconds".
        orbital_period (float or None): Orbital period in seconds.  Required
            when time_units="orbits".  Default: None.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
        **kwargs: Forwarded to the array-API function.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual backend={backend} type={residual_type}")
    validate_backend(backend)

    if residual_type == "both":
        fig = _residual_both_overlay(
            solver,
            iteration,
            model_name,
            labels,
            colors,
            time_units,
            orbital_period,
            backend,
            backend_config or {},
        )
    else:
        epochs, residuals, _ = extract_residuals(
            solver, iteration, residual_type, model_name
        )
        time_vals, time_lbl = compute_time_axis(epochs, time_units, orbital_period)
        fig = plot_measurement_residual_from_arrays(
            time_vals,
            residuals,
            labels=labels,
            colors=colors,
            time_label=time_lbl,
            backend=backend,
            backend_config=backend_config,
            **kwargs,
        )

    logger.info(f"plot_measurement_residual done in {_time.time() - start:.2f}s")
    return fig


# =============================================================================
# Public API — solver API: grid residual scatter
# =============================================================================


def plot_measurement_residual_grid(
    solver,
    iteration=-1,
    residual_type="postfit",
    model_name=None,
    labels=None,
    ncols=3,
    time_units="seconds",
    orbital_period=None,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot measurement residuals in a subplot grid from a solved estimator.

    Extracts residuals from the solver and delegates to
    :func:`plot_measurement_residual_grid_from_arrays`.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        iteration (int): Iteration index for BLS (default -1, last).
            Ignored for sequential filters.
        residual_type (str): "prefit" or "postfit".  Default: "postfit".
        model_name (str or None): Filter residuals by measurement model name.
            None means include all.  Default: None.
        labels (list[str] or None): Label per component.  Default: generated.
        ncols (int): Number of subplot columns.  Default: 3.
        time_units (str or callable): Time axis units.  Default: "seconds".
        orbital_period (float or None): Orbital period in seconds.  Default: None.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
        **kwargs: Forwarded to the array-API function.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual_grid backend={backend}")
    validate_backend(backend)

    epochs, residuals, _ = extract_residuals(
        solver, iteration, residual_type, model_name
    )
    time_vals, time_lbl = compute_time_axis(epochs, time_units, orbital_period)

    fig = plot_measurement_residual_grid_from_arrays(
        time_vals,
        residuals,
        labels=labels,
        ncols=ncols,
        time_label=time_lbl,
        backend=backend,
        backend_config=backend_config,
        **kwargs,
    )

    logger.info(f"plot_measurement_residual_grid done in {_time.time() - start:.2f}s")
    return fig


# =============================================================================
# Public API — solver API: RMS line
# =============================================================================


def plot_measurement_residual_rms(
    solver,
    residual_type="postfit",
    model_name=None,
    time_units="seconds",
    orbital_period=None,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot the per-epoch RMS of measurement residuals from a solved estimator.

    When ``residual_type="both"``, prefit and postfit RMS lines are overlaid on
    the same axes.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        residual_type (str): "prefit", "postfit", or "both".
            Default: "postfit".
        model_name (str or None): Filter residuals by measurement model name.
            None means include all.  Default: None.
        time_units (str or callable): Time axis units.  Default: "seconds".
        orbital_period (float or None): Orbital period in seconds.  Default: None.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_measurement_residual_rms backend={backend} type={residual_type}")
    validate_backend(backend)

    cfg = backend_config or {}

    if residual_type == "both":
        fig = _residual_rms_both(
            solver, model_name, time_units, orbital_period, backend, cfg
        )
    else:
        epochs, residuals, _ = extract_residuals(solver, -1, residual_type, model_name)
        time_vals, time_lbl = compute_time_axis(epochs, time_units, orbital_period)
        fig = plot_measurement_residual_rms_from_arrays(
            time_vals,
            residuals,
            time_label=time_lbl,
            backend=backend,
            backend_config=backend_config,
            **kwargs,
        )

    logger.info(f"plot_measurement_residual_rms done in {_time.time() - start:.2f}s")
    return fig


# =============================================================================
# Private — helper: "both" overlay for single-panel scatter
# =============================================================================


def _residual_both_overlay(
    solver,
    iteration,
    model_name,
    labels,
    colors,
    time_units,
    orbital_period,
    backend,
    cfg,
):
    """Overlay prefit (circles, alpha=0.5) and postfit (x, alpha=0.9) scatter."""
    epochs_pre, residuals_pre, n_components = extract_residuals(
        solver, iteration, "prefit", model_name
    )
    epochs_post, residuals_post, _ = extract_residuals(
        solver, iteration, "postfit", model_name
    )
    time_pre, time_lbl = compute_time_axis(epochs_pre, time_units, orbital_period)
    time_post, _ = compute_time_axis(epochs_post, time_units, orbital_period)

    resolved_colors = resolve_colors(n_components, colors)
    resolved_labels = resolve_labels(n_components, labels)

    if backend == "matplotlib":
        return _residual_both_matplotlib(
            time_pre,
            residuals_pre,
            time_post,
            residuals_post,
            resolved_labels,
            resolved_colors,
            n_components,
            time_lbl,
            cfg,
        )
    else:
        return _residual_both_plotly(
            time_pre,
            residuals_pre,
            time_post,
            residuals_post,
            resolved_labels,
            resolved_colors,
            n_components,
            time_lbl,
            cfg,
        )


# =============================================================================
# Private — helper: "both" overlay for RMS
# =============================================================================


def _residual_rms_both(solver, model_name, time_units, orbital_period, backend, cfg):
    """Overlay prefit RMS (#1f77b4) and postfit RMS (#d62728) lines."""
    epochs_pre, residuals_pre, _ = extract_residuals(solver, -1, "prefit", model_name)
    epochs_post, residuals_post, _ = extract_residuals(
        solver, -1, "postfit", model_name
    )
    time_pre, time_lbl = compute_time_axis(epochs_pre, time_units, orbital_period)
    time_post, _ = compute_time_axis(epochs_post, time_units, orbital_period)

    rms_pre = np.sqrt(np.mean(residuals_pre**2, axis=1))
    rms_post = np.sqrt(np.mean(residuals_post**2, axis=1))

    if backend == "matplotlib":
        return _rms_both_matplotlib(
            time_pre, rms_pre, time_post, rms_post, time_lbl, cfg
        )
    else:
        return _rms_both_plotly(time_pre, rms_pre, time_post, rms_post, time_lbl, cfg)


# =============================================================================
# Private — matplotlib implementations
# =============================================================================


def _residual_single_matplotlib(times, residuals, labels, colors, time_label, cfg):
    """Matplotlib implementation for single-panel residual scatter."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    n_components = residuals.shape[1] if residuals.ndim == 2 else 1
    for comp_idx in range(n_components):
        col_data = residuals[:, comp_idx] if residuals.ndim == 2 else residuals
        ax.scatter(
            times,
            col_data,
            color=colors[comp_idx],
            label=labels[comp_idx],
            s=15,
            alpha=0.7,
        )

    ax.set_xlabel(time_label)
    ax.set_ylabel("Residual")
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _residual_grid_matplotlib(
    times, residuals, labels, colors, n_components, ncols, time_label, cfg
):
    """Matplotlib implementation for grid residual scatter."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    nrows, ncols = compute_grid_layout(n_components, ncols)
    figsize = cfg.get("figsize", (15, 10))

    fig, axes_2d = plt.subplots(nrows, ncols, figsize=figsize)
    axes_2d = np.atleast_2d(axes_2d)

    for comp_idx in range(n_components):
        row = comp_idx // ncols
        col = comp_idx % ncols
        ax = axes_2d[row, col]

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        col_data = residuals[:, comp_idx] if residuals.ndim == 2 else residuals
        ax.scatter(
            times,
            col_data,
            color=colors[comp_idx],
            s=15,
            alpha=0.7,
        )
        ax.set_title(labels[comp_idx])
        ax.set_xlabel(time_label)
        ax.set_ylabel("Residual")
        ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)

    # Hide unused subplots
    total_subplots = nrows * ncols
    for idx in range(n_components, total_subplots):
        r = idx // ncols
        c = idx % ncols
        axes_2d[r, c].set_visible(False)

    plt.tight_layout()
    return fig


def _residual_rms_matplotlib(times, rms, time_label, cfg):
    """Matplotlib implementation for RMS line plot."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, rms, color="#1f77b4", label="RMS")

    ax.set_xlabel(time_label)
    ax.set_ylabel("Residual RMS")
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _residual_both_matplotlib(
    time_pre,
    residuals_pre,
    time_post,
    residuals_post,
    labels,
    colors,
    n_components,
    time_label,
    cfg,
):
    """Matplotlib implementation for prefit/postfit overlay scatter."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    for comp_idx in range(n_components):
        color = colors[comp_idx]
        lbl = labels[comp_idx]

        pre_data = (
            residuals_pre[:, comp_idx] if residuals_pre.ndim == 2 else residuals_pre
        )
        post_data = (
            residuals_post[:, comp_idx] if residuals_post.ndim == 2 else residuals_post
        )

        ax.scatter(
            time_pre,
            pre_data,
            color=color,
            marker="o",
            s=15,
            alpha=0.5,
            label=f"{lbl} (prefit)",
        )
        ax.scatter(
            time_post,
            post_data,
            color=color,
            marker="x",
            s=20,
            alpha=0.9,
            label=f"{lbl} (postfit)",
        )

    ax.set_xlabel(time_label)
    ax.set_ylabel("Residual")
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _rms_both_matplotlib(time_pre, rms_pre, time_post, rms_post, time_label, cfg):
    """Matplotlib implementation for prefit + postfit RMS overlay."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_pre, rms_pre, color="#1f77b4", label="Prefit RMS")
    ax.plot(time_post, rms_post, color="#d62728", label="Postfit RMS")

    ax.set_xlabel(time_label)
    ax.set_ylabel("Residual RMS")
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


# =============================================================================
# Private — plotly implementations
# =============================================================================


def _residual_single_plotly(times, residuals, labels, colors, time_label, cfg):
    """Plotly implementation for single-panel residual scatter."""
    fig = go.Figure()

    # Zero reference line
    if len(times) > 0:
        fig.add_trace(
            go.Scatter(
                x=[times[0], times[-1]],
                y=[0, 0],
                mode="lines",
                line={"color": "gray", "dash": "dash", "width": 0.5},
                showlegend=False,
            )
        )

    n_components = residuals.shape[1] if residuals.ndim == 2 else 1
    for comp_idx in range(n_components):
        col_data = residuals[:, comp_idx] if residuals.ndim == 2 else residuals
        fig.add_trace(
            go.Scatter(
                x=times,
                y=col_data,
                mode="markers",
                marker={"color": colors[comp_idx], "size": 5, "opacity": 0.7},
                name=labels[comp_idx],
            )
        )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title="Residual",
        width=width,
        height=height,
    )
    return fig


def _residual_grid_plotly(
    times, residuals, labels, colors, n_components, ncols, time_label, cfg
):
    """Plotly implementation for grid residual scatter."""
    nrows, ncols = compute_grid_layout(n_components, ncols)

    subplot_titles = labels[:n_components]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for comp_idx in range(n_components):
        row = comp_idx // ncols + 1
        col = comp_idx % ncols + 1

        col_data = residuals[:, comp_idx] if residuals.ndim == 2 else residuals

        # Zero reference
        if len(times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[times[0], times[-1]],
                    y=[0, 0],
                    mode="lines",
                    line={"color": "gray", "dash": "dash", "width": 0.5},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=col_data,
                mode="markers",
                marker={"color": colors[comp_idx], "size": 5, "opacity": 0.7},
                name=labels[comp_idx],
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(width=width, height=height)
    return fig


def _residual_rms_plotly(times, rms, time_label, cfg):
    """Plotly implementation for RMS line plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=rms,
            mode="lines",
            name="RMS",
            line={"color": "#1f77b4"},
        )
    )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title="Residual RMS",
        width=width,
        height=height,
    )
    return fig


def _residual_both_plotly(
    time_pre,
    residuals_pre,
    time_post,
    residuals_post,
    labels,
    colors,
    n_components,
    time_label,
    cfg,
):
    """Plotly implementation for prefit/postfit overlay scatter."""
    fig = go.Figure()

    # Zero reference line
    if len(time_post) > 0:
        fig.add_trace(
            go.Scatter(
                x=[time_post[0], time_post[-1]],
                y=[0, 0],
                mode="lines",
                line={"color": "gray", "dash": "dash", "width": 0.5},
                showlegend=False,
            )
        )

    for comp_idx in range(n_components):
        color = colors[comp_idx]
        lbl = labels[comp_idx]

        pre_data = (
            residuals_pre[:, comp_idx] if residuals_pre.ndim == 2 else residuals_pre
        )
        post_data = (
            residuals_post[:, comp_idx] if residuals_post.ndim == 2 else residuals_post
        )

        fig.add_trace(
            go.Scatter(
                x=time_pre,
                y=pre_data,
                mode="markers",
                marker={"color": color, "symbol": "circle", "size": 5, "opacity": 0.5},
                name=f"{lbl} (prefit)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_post,
                y=post_data,
                mode="markers",
                marker={"color": color, "symbol": "x", "size": 6, "opacity": 0.9},
                name=f"{lbl} (postfit)",
            )
        )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title="Residual",
        width=width,
        height=height,
    )
    return fig


def _rms_both_plotly(time_pre, rms_pre, time_post, rms_post, time_label, cfg):
    """Plotly implementation for prefit + postfit RMS overlay."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_pre,
            y=rms_pre,
            mode="lines",
            name="Prefit RMS",
            line={"color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_post,
            y=rms_post,
            mode="lines",
            name="Postfit RMS",
            line={"color": "#d62728"},
        )
    )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title="Residual RMS",
        width=width,
        height=height,
    )
    return fig
