"""Tests for estimation_marginal array-API and solver-API plotting functions."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import brahe as bh

from brahe.plots.estimation_marginal import (
    plot_estimator_marginal_from_arrays,
    plot_estimator_marginal,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def marginal_arrays():
    means = [np.array([1.0, -1.0]), np.array([0.5, -0.5])]
    covs = [
        np.array([[2.0, 0.5], [0.5, 1.0]]),
        np.array([[1.5, -0.3], [-0.3, 0.8]]),
    ]
    return means, covs


@pytest.fixture(scope="module")
def solved_bls():
    """Solve a BLS with a perturbed LEO initial state against position observations."""
    bh.initialize_eop()

    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    r = bh.R_EARTH + 500e3
    v = (bh.GM_EARTH / r) ** 0.5
    true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    prop_truth = bh.NumericalOrbitPropagator(
        epoch,
        true_state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
    )

    observations = []
    for i in range(1, 21):
        t = epoch + i * 30.0
        prop_truth.propagate_to(t)
        state_at_t = prop_truth.current_state()
        pos = state_at_t[:3]
        observations.append(bh.Observation(t, pos, 0))

    initial_state = true_state.copy()
    initial_state[0] += 1000.0
    p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    bls = bh.BatchLeastSquares(
        epoch,
        initial_state,
        p0,
        propagation_config=bh.NumericalPropagationConfig.default(),
        force_config=bh.ForceModelConfig.two_body(),
        measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    )
    bls.solve(observations)

    return bls


# =============================================================================
# plot_estimator_marginal_from_arrays — matplotlib
# =============================================================================


class TestMarginalFromArraysMatplotlib:
    def test_basic_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_scatter_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        rng = np.random.default_rng(42)
        scatter = rng.multivariate_normal([0.75, -0.75], [[1.5, 0.2], [0.2, 0.9]], 100)
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, scatter_points=scatter, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_marginals_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, show_marginals=False, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_sigma_returns_figure(self, marginal_arrays):
        """sigma=None means only mean markers are plotted (no ellipses)."""
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=None, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_marginals_has_multiple_axes(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, show_marginals=True, backend="matplotlib"
        )
        # Should have 3 axes: main, top marginal, right marginal
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_without_marginals_has_one_axis(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, show_marginals=False, backend="matplotlib"
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_custom_labels(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means,
            covs,
            sigma=2,
            labels=["Filter A", "Filter B"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_state_labels_axis(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means,
            covs,
            sigma=2,
            state_labels=("X Position [m]", "Y Position [m]"),
            backend="matplotlib",
        )
        ax_main = fig.axes[0]  # main is first added (gs[1, 0]) when marginals=True
        assert ax_main.get_xlabel() == "X Position [m]"
        assert ax_main.get_ylabel() == "Y Position [m]"
        plt.close(fig)

    def test_custom_colors(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means,
            covs,
            sigma=2,
            colors=["red", "blue"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backend_config_figsize(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means,
            covs,
            sigma=2,
            backend="matplotlib",
            backend_config={"figsize": (10, 10)},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_series(self):
        means = [np.array([0.0, 0.0])]
        covs = [np.array([[1.0, 0.0], [0.0, 1.0]])]
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=3, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_estimator_marginal_from_arrays — plotly
# =============================================================================


class TestMarginalFromArraysPlotly:
    def test_basic_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_scatter_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        rng = np.random.default_rng(42)
        scatter = rng.multivariate_normal([0.75, -0.75], [[1.5, 0.2], [0.2, 0.9]], 100)
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, scatter_points=scatter, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_no_sigma_returns_figure(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=None, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_labels(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means, covs, sigma=2, labels=["A", "B"], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, marginal_arrays):
        means, covs = marginal_arrays
        fig = plot_estimator_marginal_from_arrays(
            means,
            covs,
            sigma=2,
            backend="plotly",
            backend_config={"width": 900, "height": 900},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_estimator_marginal — solver API
# =============================================================================


class TestMarginalSolverMatplotlib:
    def test_matplotlib_basic(self, solved_bls):
        fig = plot_estimator_marginal(
            [solved_bls],
            state_indices=(0, 1),
            sigma=3,
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_matplotlib_no_sigma(self, solved_bls):
        fig = plot_estimator_marginal(
            [solved_bls],
            state_indices=(0, 1),
            sigma=None,
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMarginalSolverPlotly:
    def test_plotly_basic(self, solved_bls):
        fig = plot_estimator_marginal(
            [solved_bls],
            state_indices=(0, 1),
            sigma=3,
            backend="plotly",
        )
        assert isinstance(fig, go.Figure)
