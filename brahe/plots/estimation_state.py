"""
Estimation state visualization — array API.

Provides functions to plot state errors and state values from raw numpy arrays,
supporting both matplotlib and plotly backends.  Solver-API wrappers (Task 3)
will be layered on top of these array-API functions.
"""

import time as _time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.estimation_common import (
    resolve_colors,
    resolve_labels,
    compute_grid_layout,
    compute_time_axis,  # noqa: F401 – used by Task 3 solver-API wrappers
    extract_state_errors,  # noqa: F401
    extract_state_history,  # noqa: F401
    extract_covariance_sigmas,  # noqa: F401
)


# =============================================================================
# Public API — single-state error
# =============================================================================


def plot_estimator_state_error_from_arrays(
    times,
    errors,
    sigmas=None,
    labels=None,
    colors=None,
    state_label=None,
    time_label="Time [s]",
    measurements=None,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot estimator state error time series from raw numpy arrays.

    Draws one error line per series with optional ±sigma covariance bands and
    an optional measurement scatter overlay.

    Args:
        times (list[numpy.ndarray]): List of 1-D time arrays, one per series.
        errors (list[numpy.ndarray]): List of 1-D error arrays, one per series.
        sigmas (list[numpy.ndarray] or None): List of 1-D sigma arrays (the
            half-width of the covariance band), one per series.  None means no
            bands are drawn.  Default: None.
        labels (list[str] or None): Legend label for each series.  Default
            labels are "Series 0", "Series 1", …
        colors (list[str] or None): Colour for each series.  Defaults to the
            internal colour cycle.
        state_label (str or None): Y-axis label.  Default: "State Error".
        time_label (str): X-axis label.  Default: "Time [s]".
        measurements (tuple or None): Optional ``(times_array, values_array)``
            pair to overlay as scatter markers.  Default: None.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration keys.
            Matplotlib keys: ``figsize``, ``legend_loc``, ``dark_mode``,
            ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored (reserved for forward-compatibility).

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_estimator_state_error_from_arrays backend={backend}")
    validate_backend(backend)

    n = len(errors)
    resolved_colors = resolve_colors(n, colors)
    resolved_labels = resolve_labels(n, labels)
    resolved_state_label = state_label if state_label is not None else "State Error"
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _state_error_single_matplotlib(
            times,
            errors,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_label,
            time_label,
            measurements,
            cfg,
        )
    else:
        fig = _state_error_single_plotly(
            times,
            errors,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_label,
            time_label,
            measurements,
            cfg,
        )

    logger.info(
        f"plot_estimator_state_error_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — single-state value
# =============================================================================


def plot_estimator_state_value_from_arrays(
    times,
    values,
    true_values=None,
    sigmas=None,
    labels=None,
    colors=None,
    state_label=None,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot estimator state value time series from raw numpy arrays.

    Draws one estimated-value line per series with an optional truth reference
    line and optional ±sigma covariance bands.

    Args:
        times (list[numpy.ndarray]): List of 1-D time arrays, one per series.
        values (list[numpy.ndarray]): List of 1-D value arrays, one per series.
        true_values (numpy.ndarray or None): 1-D truth reference array drawn
            as a black dashed line.  Must share the same time axis as the first
            series.  Default: None.
        sigmas (list[numpy.ndarray] or None): List of 1-D sigma arrays, one per
            series.  Default: None.
        labels (list[str] or None): Legend labels.  Default: "Series 0", …
        colors (list[str] or None): Colours for each series.  Default: colour
            cycle.
        state_label (str or None): Y-axis label.  Default: "State Value".
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
    logger.info(f"plot_estimator_state_value_from_arrays backend={backend}")
    validate_backend(backend)

    n = len(values)
    resolved_colors = resolve_colors(n, colors)
    resolved_labels = resolve_labels(n, labels)
    resolved_state_label = state_label if state_label is not None else "State Value"
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _state_value_single_matplotlib(
            times,
            values,
            true_values,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_label,
            time_label,
            cfg,
        )
    else:
        fig = _state_value_single_plotly(
            times,
            values,
            true_values,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_label,
            time_label,
            cfg,
        )

    logger.info(
        f"plot_estimator_state_value_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — multi-state error grid
# =============================================================================


def plot_estimator_state_error_grid_from_arrays(
    times,
    errors,
    sigmas=None,
    labels=None,
    colors=None,
    state_labels=None,
    ncols=3,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot a grid of state error time series from raw numpy arrays.

    Each subplot corresponds to one state component.  One line (and optional
    sigma band) is drawn per series.

    Args:
        times (list[numpy.ndarray]): List of 1-D time arrays, one per series.
        errors (list[numpy.ndarray]): List of 2-D arrays of shape ``(N, n_states)``,
            one per series.
        sigmas (list[numpy.ndarray] or None): List of 2-D arrays of shape
            ``(N, n_states)``, one per series.  Default: None.
        labels (list[str] or None): Legend labels.  Default: "Series 0", …
        colors (list[str] or None): Colours.  Default: colour cycle.
        state_labels (list[str] or None): Y-axis label for each state subplot.
            Default: "State 0", "State 1", …
        ncols (int): Number of columns in the subplot grid.  Default: 3.
        time_label (str): X-axis label.  Default: "Time [s]".
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``, ``legend_subplot``, ``legend_loc``,
            ``dark_mode``, ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_estimator_state_error_grid_from_arrays backend={backend}")
    validate_backend(backend)

    n_series = len(errors)
    n_states = errors[0].shape[1]
    resolved_colors = resolve_colors(n_series, colors)
    resolved_labels = resolve_labels(n_series, labels)
    resolved_state_labels = (
        state_labels
        if state_labels is not None
        else [f"State {i}" for i in range(n_states)]
    )
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _state_error_grid_matplotlib(
            times,
            errors,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            ncols,
            time_label,
            cfg,
        )
    else:
        fig = _state_error_grid_plotly(
            times,
            errors,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            ncols,
            time_label,
            cfg,
        )

    logger.info(
        f"plot_estimator_state_error_grid_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — multi-state value grid
# =============================================================================


def plot_estimator_state_value_grid_from_arrays(
    times,
    values,
    true_values=None,
    sigmas=None,
    labels=None,
    colors=None,
    state_labels=None,
    ncols=3,
    time_label="Time [s]",
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot a grid of state value time series from raw numpy arrays.

    Each subplot corresponds to one state component.  An optional truth
    reference line (black dashed) and optional sigma bands can be overlaid.

    Args:
        times (list[numpy.ndarray]): List of 1-D time arrays, one per series.
        values (list[numpy.ndarray]): List of 2-D arrays of shape
            ``(N, n_states)``, one per series.
        true_values (numpy.ndarray or None): 2-D array of shape
            ``(N, n_states)`` for the truth reference.  Default: None.
        sigmas (list[numpy.ndarray] or None): List of 2-D arrays of shape
            ``(N, n_states)``, one per series.  Default: None.
        labels (list[str] or None): Legend labels.  Default: "Series 0", …
        colors (list[str] or None): Colours.  Default: colour cycle.
        state_labels (list[str] or None): Y-axis labels per state.  Default:
            "State 0", "State 1", …
        ncols (int): Number of columns in the subplot grid.  Default: 3.
        time_label (str): X-axis label.  Default: "Time [s]".
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``, ``legend_subplot``, ``legend_loc``,
            ``dark_mode``, ``ylabel_pad``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_estimator_state_value_grid_from_arrays backend={backend}")
    validate_backend(backend)

    n_series = len(values)
    n_states = values[0].shape[1]
    resolved_colors = resolve_colors(n_series, colors)
    resolved_labels = resolve_labels(n_series, labels)
    resolved_state_labels = (
        state_labels
        if state_labels is not None
        else [f"State {i}" for i in range(n_states)]
    )
    cfg = backend_config or {}

    if backend == "matplotlib":
        fig = _state_value_grid_matplotlib(
            times,
            values,
            true_values,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            ncols,
            time_label,
            cfg,
        )
    else:
        fig = _state_value_grid_plotly(
            times,
            values,
            true_values,
            sigmas,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            ncols,
            time_label,
            cfg,
        )

    logger.info(
        f"plot_estimator_state_value_grid_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Private — matplotlib implementations
# =============================================================================


def _state_error_single_matplotlib(
    times,
    errors,
    sigmas,
    labels,
    colors,
    state_label,
    time_label,
    measurements,
    cfg,
):
    """Matplotlib implementation for single-state error plot."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    for i, (t, e) in enumerate(zip(times, errors)):
        color = colors[i]
        label = labels[i]
        ax.plot(t, e, color=color, label=label)
        if sigmas is not None and i < len(sigmas):
            s = sigmas[i]
            ax.fill_between(t, e - s, e + s, color=color, alpha=0.2)

    if measurements is not None:
        meas_t, meas_v = measurements
        ax.scatter(
            meas_t,
            meas_v,
            color="black",
            marker="x",
            s=20,
            zorder=5,
            label="Measurements",
        )

    ax.set_xlabel(time_label)
    ax.set_ylabel(state_label)
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _state_value_single_matplotlib(
    times,
    values,
    true_values,
    sigmas,
    labels,
    colors,
    state_label,
    time_label,
    cfg,
):
    """Matplotlib implementation for single-state value plot."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    figsize = cfg.get("figsize", (10, 6))
    fig, ax = plt.subplots(figsize=figsize)

    for i, (t, v) in enumerate(zip(times, values)):
        color = colors[i]
        label = labels[i]
        ax.plot(t, v, color=color, label=label)
        if sigmas is not None and i < len(sigmas):
            s = sigmas[i]
            ax.fill_between(t, v - s, v + s, color=color, alpha=0.2)

    if true_values is not None:
        # Use the time axis of the first series for truth
        ax.plot(
            times[0],
            true_values,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Truth",
        )

    ax.set_xlabel(time_label)
    ax.set_ylabel(state_label)
    ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
    legend_loc = cfg.get("legend_loc", "best")
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _state_error_grid_matplotlib(
    times,
    errors,
    sigmas,
    labels,
    colors,
    state_labels,
    ncols,
    time_label,
    cfg,
):
    """Matplotlib implementation for multi-state error grid."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    n_states = len(state_labels)
    nrows, ncols = compute_grid_layout(n_states, ncols)
    figsize = cfg.get("figsize", (15, 10))
    legend_subplot = cfg.get("legend_subplot", (0, 0))
    legend_loc = cfg.get("legend_loc", "best")

    fig, axes_2d = plt.subplots(nrows, ncols, figsize=figsize)
    # Normalise to 2-D array regardless of nrows/ncols
    axes_2d = np.atleast_2d(axes_2d)

    for state_idx in range(n_states):
        row = state_idx // ncols
        col = state_idx % ncols
        ax = axes_2d[row, col]

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        for series_idx, (t, e) in enumerate(zip(times, errors)):
            color = colors[series_idx]
            label = labels[series_idx]
            ax.plot(t, e[:, state_idx], color=color, label=label)
            if sigmas is not None and series_idx < len(sigmas):
                s = sigmas[series_idx][:, state_idx]
                ax.fill_between(
                    t, e[:, state_idx] - s, e[:, state_idx] + s, color=color, alpha=0.2
                )

        ax.set_ylabel(state_labels[state_idx])
        ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
        ax.set_xlabel(time_label)

    # Add legend to the designated subplot only
    legend_row, legend_col = legend_subplot
    if legend_row < nrows and legend_col < ncols:
        axes_2d[legend_row, legend_col].legend(loc=legend_loc)

    # Hide unused subplots
    total_subplots = nrows * ncols
    for idx in range(n_states, total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes_2d[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def _state_value_grid_matplotlib(
    times,
    values,
    true_values,
    sigmas,
    labels,
    colors,
    state_labels,
    ncols,
    time_label,
    cfg,
):
    """Matplotlib implementation for multi-state value grid."""
    apply_scienceplots_style()
    if cfg.get("dark_mode", False):
        plt.style.use("dark_background")

    n_states = len(state_labels)
    nrows, ncols = compute_grid_layout(n_states, ncols)
    figsize = cfg.get("figsize", (15, 10))
    legend_subplot = cfg.get("legend_subplot", (0, 0))
    legend_loc = cfg.get("legend_loc", "best")

    fig, axes_2d = plt.subplots(nrows, ncols, figsize=figsize)
    axes_2d = np.atleast_2d(axes_2d)

    for state_idx in range(n_states):
        row = state_idx // ncols
        col = state_idx % ncols
        ax = axes_2d[row, col]

        for series_idx, (t, v) in enumerate(zip(times, values)):
            color = colors[series_idx]
            label = labels[series_idx]
            ax.plot(t, v[:, state_idx], color=color, label=label)
            if sigmas is not None and series_idx < len(sigmas):
                s = sigmas[series_idx][:, state_idx]
                ax.fill_between(
                    t, v[:, state_idx] - s, v[:, state_idx] + s, color=color, alpha=0.2
                )

        if true_values is not None:
            ax.plot(
                times[0],
                true_values[:, state_idx],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Truth",
            )

        ax.set_ylabel(state_labels[state_idx])
        ax.yaxis.labelpad = cfg.get("ylabel_pad", 10)
        ax.set_xlabel(time_label)

    # Add legend to the designated subplot only
    legend_row, legend_col = legend_subplot
    if legend_row < nrows and legend_col < ncols:
        axes_2d[legend_row, legend_col].legend(loc=legend_loc)

    # Hide unused subplots
    total_subplots = nrows * ncols
    for idx in range(n_states, total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes_2d[row, col].set_visible(False)

    plt.tight_layout()
    return fig


# =============================================================================
# Private — plotly implementations
# =============================================================================


def _make_plotly_band(t, lower, upper, color, name, legendgroup, show_legend):
    """Return a single filled-band Scatter trace for plotly."""
    t_closed = np.concatenate([t, t[::-1]])
    y_closed = np.concatenate([upper, lower[::-1]])
    return go.Scatter(
        x=t_closed,
        y=y_closed,
        fill="toself",
        fillcolor=color,
        line={"color": "rgba(255,255,255,0)"},
        opacity=0.2,
        name=name,
        legendgroup=legendgroup,
        showlegend=show_legend,
        hoverinfo="skip",
    )


def _state_error_single_plotly(
    times,
    errors,
    sigmas,
    labels,
    colors,
    state_label,
    time_label,
    measurements,
    cfg,
):
    """Plotly implementation for single-state error plot."""
    fig = go.Figure()

    for i, (t, e) in enumerate(zip(times, errors)):
        color = colors[i]
        label = labels[i]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e,
                mode="lines",
                name=label,
                line={"color": color},
                legendgroup=label,
            )
        )
        if sigmas is not None and i < len(sigmas):
            s = sigmas[i]
            fig.add_trace(
                _make_plotly_band(t, e - s, e + s, color, f"{label} ±σ", label, False)
            )

    # Zero reference line
    if len(times) > 0:
        t_all = times[0]
        fig.add_trace(
            go.Scatter(
                x=[t_all[0], t_all[-1]],
                y=[0, 0],
                mode="lines",
                line={"color": "gray", "dash": "dash", "width": 0.5},
                showlegend=False,
            )
        )

    if measurements is not None:
        meas_t, meas_v = measurements
        fig.add_trace(
            go.Scatter(
                x=meas_t,
                y=meas_v,
                mode="markers",
                marker={"color": "black", "symbol": "x", "size": 6},
                name="Measurements",
            )
        )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title=state_label,
        width=width,
        height=height,
    )
    return fig


def _state_value_single_plotly(
    times,
    values,
    true_values,
    sigmas,
    labels,
    colors,
    state_label,
    time_label,
    cfg,
):
    """Plotly implementation for single-state value plot."""
    fig = go.Figure()

    for i, (t, v) in enumerate(zip(times, values)):
        color = colors[i]
        label = labels[i]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=v,
                mode="lines",
                name=label,
                line={"color": color},
                legendgroup=label,
            )
        )
        if sigmas is not None and i < len(sigmas):
            s = sigmas[i]
            fig.add_trace(
                _make_plotly_band(t, v - s, v + s, color, f"{label} ±σ", label, False)
            )

    if true_values is not None:
        fig.add_trace(
            go.Scatter(
                x=times[0],
                y=true_values,
                mode="lines",
                name="Truth",
                line={"color": "black", "dash": "dash", "width": 1.5},
            )
        )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(
        xaxis_title=time_label,
        yaxis_title=state_label,
        width=width,
        height=height,
    )
    return fig


def _state_error_grid_plotly(
    times,
    errors,
    sigmas,
    labels,
    colors,
    state_labels,
    ncols,
    time_label,
    cfg,
):
    """Plotly implementation for multi-state error grid."""
    n_states = len(state_labels)
    nrows, ncols = compute_grid_layout(n_states, ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=state_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for state_idx in range(n_states):
        row = state_idx // ncols + 1  # plotly is 1-indexed
        col = state_idx % ncols + 1
        is_first_state = state_idx == 0

        for series_idx, (t, e) in enumerate(zip(times, errors)):
            color = colors[series_idx]
            label = labels[series_idx]
            show_legend = is_first_state
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=e[:, state_idx],
                    mode="lines",
                    name=label,
                    line={"color": color},
                    legendgroup=label,
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
            if sigmas is not None and series_idx < len(sigmas):
                s = sigmas[series_idx][:, state_idx]
                band = _make_plotly_band(
                    t,
                    e[:, state_idx] - s,
                    e[:, state_idx] + s,
                    color,
                    f"{label} ±σ",
                    label,
                    False,
                )
                fig.add_trace(band, row=row, col=col)

        # Zero reference
        if len(times) > 0:
            t_ref = times[0]
            fig.add_trace(
                go.Scatter(
                    x=[t_ref[0], t_ref[-1]],
                    y=[0, 0],
                    mode="lines",
                    line={"color": "gray", "dash": "dash", "width": 0.5},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(width=width, height=height)
    return fig


def _state_value_grid_plotly(
    times,
    values,
    true_values,
    sigmas,
    labels,
    colors,
    state_labels,
    ncols,
    time_label,
    cfg,
):
    """Plotly implementation for multi-state value grid."""
    n_states = len(state_labels)
    nrows, ncols = compute_grid_layout(n_states, ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=state_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for state_idx in range(n_states):
        row = state_idx // ncols + 1
        col = state_idx % ncols + 1
        is_first_state = state_idx == 0

        for series_idx, (t, v) in enumerate(zip(times, values)):
            color = colors[series_idx]
            label = labels[series_idx]
            show_legend = is_first_state
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=v[:, state_idx],
                    mode="lines",
                    name=label,
                    line={"color": color},
                    legendgroup=label,
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
            if sigmas is not None and series_idx < len(sigmas):
                s = sigmas[series_idx][:, state_idx]
                band = _make_plotly_band(
                    t,
                    v[:, state_idx] - s,
                    v[:, state_idx] + s,
                    color,
                    f"{label} ±σ",
                    label,
                    False,
                )
                fig.add_trace(band, row=row, col=col)

        if true_values is not None:
            show_truth_legend = is_first_state
            fig.add_trace(
                go.Scatter(
                    x=times[0],
                    y=true_values[:, state_idx],
                    mode="lines",
                    name="Truth",
                    line={"color": "black", "dash": "dash", "width": 1.5},
                    legendgroup="Truth",
                    showlegend=show_truth_legend,
                ),
                row=row,
                col=col,
            )

    width = cfg.get("width", None)
    height = cfg.get("height", None)
    fig.update_layout(width=width, height=height)
    return fig
