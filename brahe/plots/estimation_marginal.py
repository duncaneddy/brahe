"""
Marginal distribution visualization — array API and solver API.

Provides functions to plot 2D covariance ellipses with optional marginal density
curves from raw numpy arrays or directly from solved estimator objects, supporting
both matplotlib and plotly backends.
"""

import time as _time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe.plots.estimation_common import (
    extract_sub_covariance,
    resolve_colors,
    resolve_labels,
)


# =============================================================================
# Private helpers
# =============================================================================


def _ellipse_points(mean, cov, sigma, n_points=100):
    """Compute covariance ellipse points using eigendecomposition.

    Args:
        mean (numpy.ndarray): Shape (2,) center of the ellipse.
        cov (numpy.ndarray): Shape (2, 2) covariance matrix.
        sigma (float): N-sigma for ellipse scaling, based on chi-squared quantile.
        n_points (int): Number of points on the ellipse. Default: 100.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: x and y arrays of ellipse points.
    """
    # Chi-squared quantile for df=2: chi2.ppf(p, df=2) = -2*ln(1-p) exactly
    # Probability enclosed by n-sigma ellipse in 2D: p = 1 - exp(-sigma^2 / 2)
    p = 1.0 - np.exp(-(sigma**2) / 2.0)
    s = np.sqrt(-2.0 * np.log(1.0 - p))

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Clamp negative eigenvalues to zero for numerical stability
    eigenvalues = np.maximum(eigenvalues, 0.0)

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues) * s)
    ellipse = circle @ transform.T + mean

    return ellipse[:, 0], ellipse[:, 1]


def _marginal_density(mean, variance, sigma, n_points=200):
    """Compute analytical Gaussian marginal density curve.

    Args:
        mean (float): Mean of the Gaussian.
        variance (float): Variance of the Gaussian.
        sigma (float): N-sigma for the plot range extent.
        n_points (int): Number of points in the density curve. Default: 200.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: x values and corresponding density values.
    """
    std = np.sqrt(max(variance, 0.0))
    x_min = mean - 4 * std
    x_max = mean + 4 * std
    x_vals = np.linspace(x_min, x_max, n_points)
    # Gaussian PDF: (1 / (std * sqrt(2*pi))) * exp(-0.5 * ((x - mean) / std)^2)
    if std == 0.0:
        density = np.zeros_like(x_vals)
    else:
        density = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x_vals - mean) / std) ** 2
        )
    return x_vals, density


# =============================================================================
# Private backend implementations — matplotlib
# =============================================================================


def _marginal_matplotlib(
    means,
    covariances,
    sigma,
    resolved_labels,
    resolved_colors,
    state_labels,
    scatter_points,
    show_marginals,
    cfg,
):
    """Matplotlib implementation for marginal distribution plot."""
    apply_scienceplots_style()
    figsize = cfg.get("figsize", (8, 8))

    if show_marginals:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)

    # Scatter overlay
    if scatter_points is not None:
        ax_main.scatter(
            scatter_points[:, 0],
            scatter_points[:, 1],
            color="gray",
            s=8,
            alpha=0.3,
            zorder=1,
            label="_nolegend_",
        )

    # Draw per-series ellipses and mean markers
    x_label, y_label = state_labels
    for mean, cov, label, color in zip(
        means, covariances, resolved_labels, resolved_colors
    ):
        # Mean marker
        ax_main.plot(mean[0], mean[1], "o", color=color, markersize=5, zorder=3)

        if sigma is not None:
            ex, ey = _ellipse_points(mean, cov, sigma)
            ax_main.plot(ex, ey, color=color, linewidth=1.5, label=label, zorder=2)

            if show_marginals:
                # Top axis: X marginal density
                xv, xd = _marginal_density(mean[0], cov[0, 0], sigma)
                ax_top.plot(xv, xd, color=color, linewidth=1.5)

                # Right axis: Y marginal density (density on X-axis, y_vals on Y-axis)
                yv, yd = _marginal_density(mean[1], cov[1, 1], sigma)
                ax_right.plot(yd, yv, color=color, linewidth=1.5)
        else:
            # No ellipses — only mean markers; use label for legend
            ax_main.plot([], [], "o", color=color, label=label)

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.legend()

    if show_marginals:
        ax_top.set_ylabel("Density")
        ax_right.set_xlabel("Density")

    return fig


# =============================================================================
# Private backend implementations — plotly
# =============================================================================


def _marginal_plotly(
    means,
    covariances,
    sigma,
    resolved_labels,
    resolved_colors,
    state_labels,
    scatter_points,
    show_marginals,
    cfg,
):
    """Plotly implementation for marginal distribution plot."""
    width = cfg.get("width", 800)
    height = cfg.get("height", 800)

    x_label, y_label = state_labels

    if show_marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
        top_row, top_col = 1, 1
        right_row, right_col = 2, 2
    else:
        fig = go.Figure()
        main_row = main_col = None

    # Scatter overlay
    if scatter_points is not None:
        scatter_trace = go.Scatter(
            x=scatter_points[:, 0],
            y=scatter_points[:, 1],
            mode="markers",
            marker=dict(color="gray", size=4, opacity=0.3),
            showlegend=False,
            name="_scatter",
        )
        if show_marginals:
            fig.add_trace(scatter_trace, row=main_row, col=main_col)
        else:
            fig.add_trace(scatter_trace)

    # Draw per-series ellipses and mean markers
    for mean, cov, label, color in zip(
        means, covariances, resolved_labels, resolved_colors
    ):
        # Mean marker
        mean_trace = go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode="markers",
            marker=dict(color=color, size=8),
            showlegend=sigma is None,
            name=label,
            legendgroup=label,
        )

        if show_marginals:
            fig.add_trace(mean_trace, row=main_row, col=main_col)
        else:
            fig.add_trace(mean_trace)

        if sigma is not None:
            ex, ey = _ellipse_points(mean, cov, sigma)
            ellipse_trace = go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=True,
                name=label,
                legendgroup=label,
            )
            if show_marginals:
                fig.add_trace(ellipse_trace, row=main_row, col=main_col)
            else:
                fig.add_trace(ellipse_trace)

            if show_marginals:
                # Top axis: X marginal density
                xv, xd = _marginal_density(mean[0], cov[0, 0], sigma)
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=xd,
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        name=f"{label} X marginal",
                        legendgroup=label,
                    ),
                    row=top_row,
                    col=top_col,
                )

                # Right axis: Y marginal density (density on X-axis, y_vals on Y-axis)
                yv, yd = _marginal_density(mean[1], cov[1, 1], sigma)
                fig.add_trace(
                    go.Scatter(
                        x=yd,
                        y=yv,
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        name=f"{label} Y marginal",
                        legendgroup=label,
                    ),
                    row=right_row,
                    col=right_col,
                )

    if show_marginals:
        fig.update_xaxes(title_text=x_label, row=main_row, col=main_col)
        fig.update_yaxes(title_text=y_label, row=main_row, col=main_col)
    else:
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
        )

    fig.update_layout(width=width, height=height)
    return fig


# =============================================================================
# Public API — array API
# =============================================================================


def plot_estimator_marginal_from_arrays(
    means,
    covariances,
    sigma=None,
    labels=None,
    colors=None,
    state_labels=None,
    scatter_points=None,
    show_marginals=True,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot 2D covariance ellipses with optional marginal density curves from arrays.

    Each entry in ``means`` / ``covariances`` represents one estimator series.
    Covariance ellipses are scaled using the chi-squared quantile for the
    requested sigma level.  Optional marginal (1D Gaussian) density curves can
    be shown on top and right axes.

    Args:
        means (list[numpy.ndarray]): List of shape-(2,) mean vectors, one per series.
        covariances (list[numpy.ndarray]): List of shape-(2, 2) covariance matrices,
            one per series.
        sigma (float or None): N-sigma level for ellipse scaling.  None means only
            mean markers are drawn (no ellipses or marginals).  Default: None.
        labels (list[str] or None): Legend label per series.  Default: "Series 0", …
        colors (list[str] or None): Colour per series.  Default: colour cycle.
        state_labels (tuple[str, str] or None): ``(x_label, y_label)`` axis labels.
            Default: ("State 0", "State 1").
        scatter_points (numpy.ndarray or None): Shape ``(N, 2)`` array of Monte
            Carlo samples overlaid as gray dots.  Default: None.
        show_marginals (bool): If True, add marginal density curves on top/right axes.
            Only used when ``sigma`` is not None.  Default: True.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
            Matplotlib keys: ``figsize``.  Plotly keys: ``width``, ``height``.
        **kwargs: Ignored (reserved for forward-compatibility).

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_estimator_marginal_from_arrays backend={backend}")
    validate_backend(backend)

    n = len(means)
    resolved_colors = resolve_colors(n, colors)
    resolved_labels = resolve_labels(n, labels)
    resolved_state_labels = (
        state_labels if state_labels is not None else ("State 0", "State 1")
    )
    cfg = backend_config or {}

    # Marginals only make sense when ellipses are drawn
    effective_marginals = show_marginals and (sigma is not None)

    if backend == "matplotlib":
        fig = _marginal_matplotlib(
            means,
            covariances,
            sigma,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            scatter_points,
            effective_marginals,
            cfg,
        )
    else:
        fig = _marginal_plotly(
            means,
            covariances,
            sigma,
            resolved_labels,
            resolved_colors,
            resolved_state_labels,
            scatter_points,
            effective_marginals,
            cfg,
        )

    logger.info(
        f"plot_estimator_marginal_from_arrays done in {_time.time() - start:.2f}s"
    )
    return fig


# =============================================================================
# Public API — solver API
# =============================================================================


def plot_estimator_marginal(
    solvers,
    state_indices=(0, 1),
    sigma=None,
    labels=None,
    colors=None,
    state_labels=None,
    scatter_points=None,
    show_marginals=True,
    backend="matplotlib",
    backend_config=None,
    **kwargs,
):
    """Plot 2D covariance ellipses with optional marginals from solved estimators.

    Extracts the 2x2 sub-covariance and 2D sub-state from each solver using the
    given ``state_indices``, then delegates to
    :func:`plot_estimator_marginal_from_arrays`.

    Args:
        solvers (list): List of solved estimator objects (BatchLeastSquares,
            ExtendedKalmanFilter, or UnscentedKalmanFilter).
        state_indices (tuple[int, int]): Two state indices to extract.
            Default: (0, 1).
        sigma (float or None): N-sigma level for ellipse scaling.  None means
            only mean markers are drawn.  Default: None.
        labels (list[str] or None): Legend label per solver.  Default: "Series 0", …
        colors (list[str] or None): Colour per solver.  Default: colour cycle.
        state_labels (tuple[str, str] or None): ``(x_label, y_label)`` axis labels.
            Default: ("State 0", "State 1").
        scatter_points (numpy.ndarray or None): Shape ``(N, 2)`` MC sample overlay.
            Default: None.
        show_marginals (bool): If True, add marginal density curves on top/right axes.
            Default: True.
        backend (str): "matplotlib" or "plotly".  Default: "matplotlib".
        backend_config (dict or None): Backend-specific configuration.
        **kwargs: Ignored.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated
        figure object.
    """
    start = _time.time()
    logger.info(f"plot_estimator_marginal backend={backend} n_solvers={len(solvers)}")

    means = []
    covariances = []
    for solver in solvers:
        mean_2d, cov_2x2 = extract_sub_covariance(solver, state_indices)
        means.append(mean_2d)
        covariances.append(cov_2x2)

    fig = plot_estimator_marginal_from_arrays(
        means,
        covariances,
        sigma=sigma,
        labels=labels,
        colors=colors,
        state_labels=state_labels,
        scatter_points=scatter_points,
        show_marginals=show_marginals,
        backend=backend,
        backend_config=backend_config,
        **kwargs,
    )

    logger.info(f"plot_estimator_marginal done in {_time.time() - start:.2f}s")
    return fig
