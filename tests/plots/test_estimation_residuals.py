"""Tests for estimation_residuals array-API and solver-API plotting functions."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import brahe as bh

from brahe.plots.estimation_residuals import (
    plot_measurement_residual_from_arrays,
    plot_measurement_residual_grid_from_arrays,
    plot_measurement_residual_rms_from_arrays,
    plot_measurement_residual,
    plot_measurement_residual_grid,
    plot_measurement_residual_rms,
)


# =============================================================================
# Fixtures — array API
# =============================================================================


@pytest.fixture
def residual_arrays():
    """50 observations with 3-component residuals."""
    rng = np.random.default_rng(42)
    times = np.linspace(0, 600, 50)
    residuals = rng.normal(0, 5, (50, 3))
    return times, residuals


@pytest.fixture
def residual_arrays_1d():
    """50 observations with 1-component residuals."""
    rng = np.random.default_rng(7)
    times = np.linspace(0, 600, 50)
    residuals = rng.normal(0, 3, (50, 1))
    return times, residuals


@pytest.fixture
def residual_arrays_6d():
    """30 observations with 6-component residuals."""
    rng = np.random.default_rng(99)
    times = np.linspace(0, 900, 30)
    residuals = rng.normal(0, 2, (30, 6))
    return times, residuals


# =============================================================================
# Fixtures — solver API
# =============================================================================


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
# plot_measurement_residual_from_arrays — matplotlib
# =============================================================================


class TestResidualFromArraysMatplotlib:
    def test_returns_figure(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times, residuals, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times,
            residuals,
            labels=["X", "Y", "Z"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_colors(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times,
            residuals,
            colors=["red", "green", "blue"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_time_label(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times,
            residuals,
            time_label="Time [min]",
            backend="matplotlib",
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time [min]"
        plt.close(fig)

    def test_default_time_label(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times, residuals, backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time [s]"
        plt.close(fig)

    def test_single_component(self, residual_arrays_1d):
        times, residuals = residual_arrays_1d
        fig = plot_measurement_residual_from_arrays(
            times, residuals, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backend_config_figsize(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times,
            residuals,
            backend="matplotlib",
            backend_config={"figsize": (8, 4)},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_measurement_residual_from_arrays — plotly
# =============================================================================


class TestResidualFromArraysPlotly:
    def test_returns_figure(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(times, residuals, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_labels(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times, residuals, labels=["X", "Y", "Z"], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_from_arrays(
            times,
            residuals,
            backend="plotly",
            backend_config={"width": 800, "height": 400},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_measurement_residual_grid_from_arrays — matplotlib
# =============================================================================


class TestResidualGridFromArraysMatplotlib:
    def test_returns_figure_3_components(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_6_components(self, residual_arrays_6d):
        times, residuals = residual_arrays_6d
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, labels=["X", "Y", "Z"], backend="matplotlib"
        )
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert axes[0].get_title() == "X"
        plt.close(fig)

    def test_default_labels(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, backend="matplotlib"
        )
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert axes[0].get_title() == "Series 0"
        plt.close(fig)

    def test_custom_ncols(self, residual_arrays_6d):
        times, residuals = residual_arrays_6d
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, ncols=2, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unused_subplots_hidden(self, residual_arrays):
        """With 3 components and ncols=3, no unused subplots; with ncols=2, one hidden."""
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, ncols=2, backend="matplotlib"
        )
        # 3 components, ncols=2 → 2 rows × 2 cols = 4 subplots, 1 unused
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) == 1
        plt.close(fig)

    def test_backend_config_figsize(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times,
            residuals,
            backend="matplotlib",
            backend_config={"figsize": (12, 8)},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_measurement_residual_grid_from_arrays — plotly
# =============================================================================


class TestResidualGridFromArraysPlotly:
    def test_returns_figure_3_components(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_returns_figure_6_components(self, residual_arrays_6d):
        times, residuals = residual_arrays_6d
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_labels(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, labels=["X", "Y", "Z"], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_ncols(self, residual_arrays_6d):
        times, residuals = residual_arrays_6d
        fig = plot_measurement_residual_grid_from_arrays(
            times, residuals, ncols=2, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_grid_from_arrays(
            times,
            residuals,
            backend="plotly",
            backend_config={"width": 1200, "height": 600},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_measurement_residual_rms_from_arrays — matplotlib
# =============================================================================


class TestResidualRmsFromArraysMatplotlib:
    def test_returns_figure(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times, residuals, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_time_label(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times, residuals, time_label="Elapsed [s]", backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Elapsed [s]"
        plt.close(fig)

    def test_default_time_label(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times, residuals, backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time [s]"
        plt.close(fig)

    def test_rms_is_non_negative(self, residual_arrays):
        """The RMS line values must be non-negative."""
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times, residuals, backend="matplotlib"
        )
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0
        rms_data = lines[0].get_ydata()
        assert np.all(rms_data >= 0)
        plt.close(fig)

    def test_backend_config_figsize(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times,
            residuals,
            backend="matplotlib",
            backend_config={"figsize": (10, 4)},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_measurement_residual_rms_from_arrays — plotly
# =============================================================================


class TestResidualRmsFromArraysPlotly:
    def test_returns_figure(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times, residuals, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, residual_arrays):
        times, residuals = residual_arrays
        fig = plot_measurement_residual_rms_from_arrays(
            times,
            residuals,
            backend="plotly",
            backend_config={"width": 800, "height": 400},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# Invalid backend — array API
# =============================================================================


class TestInvalidBackendArrayApi:
    def test_residual_raises(self, residual_arrays):
        times, residuals = residual_arrays
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_measurement_residual_from_arrays(times, residuals, backend="bokeh")

    def test_residual_grid_raises(self, residual_arrays):
        times, residuals = residual_arrays
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_measurement_residual_grid_from_arrays(
                times, residuals, backend="seaborn"
            )

    def test_residual_rms_raises(self, residual_arrays):
        times, residuals = residual_arrays
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_measurement_residual_rms_from_arrays(
                times, residuals, backend="plotly_express"
            )


# =============================================================================
# plot_measurement_residual — solver API
# =============================================================================


class TestSolverResidual:
    def test_matplotlib_returns_figure(self, solved_bls):
        fig = plot_measurement_residual(solved_bls, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotly_returns_figure(self, solved_bls):
        fig = plot_measurement_residual(solved_bls, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_prefit_type(self, solved_bls):
        fig = plot_measurement_residual(
            solved_bls, residual_type="prefit", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_both_type_matplotlib(self, solved_bls):
        fig = plot_measurement_residual(
            solved_bls, residual_type="both", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_both_type_plotly(self, solved_bls):
        fig = plot_measurement_residual(
            solved_bls, residual_type="both", backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_labels(self, solved_bls):
        fig = plot_measurement_residual(
            solved_bls, labels=["X [m]", "Y [m]", "Z [m]"], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_time_units_minutes(self, solved_bls):
        fig = plot_measurement_residual(
            solved_bls, time_units="minutes", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "min" in ax.get_xlabel()
        plt.close(fig)

    def test_last_iteration(self, solved_bls):
        fig = plot_measurement_residual(solved_bls, iteration=-1, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_measurement_residual_grid — solver API
# =============================================================================


class TestSolverResidualGrid:
    def test_matplotlib_returns_figure(self, solved_bls):
        fig = plot_measurement_residual_grid(solved_bls, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotly_returns_figure(self, solved_bls):
        fig = plot_measurement_residual_grid(solved_bls, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_prefit_type(self, solved_bls):
        fig = plot_measurement_residual_grid(
            solved_bls, residual_type="prefit", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_ncols(self, solved_bls):
        fig = plot_measurement_residual_grid(solved_bls, ncols=2, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_time_units_minutes(self, solved_bls):
        fig = plot_measurement_residual_grid(
            solved_bls, time_units="minutes", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_measurement_residual_rms — solver API
# =============================================================================


class TestSolverResidualRms:
    def test_matplotlib_returns_figure(self, solved_bls):
        fig = plot_measurement_residual_rms(solved_bls, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotly_returns_figure(self, solved_bls):
        fig = plot_measurement_residual_rms(solved_bls, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_prefit_type(self, solved_bls):
        fig = plot_measurement_residual_rms(
            solved_bls, residual_type="prefit", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_both_type_matplotlib(self, solved_bls):
        fig = plot_measurement_residual_rms(
            solved_bls, residual_type="both", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        # Both type should produce 2 lines (prefit + postfit)
        ax = fig.axes[0]
        lines = [ln for ln in ax.get_lines() if ln.get_label() != "_nolegend_"]
        assert len(lines) == 2
        plt.close(fig)

    def test_both_type_plotly(self, solved_bls):
        fig = plot_measurement_residual_rms(
            solved_bls, residual_type="both", backend="plotly"
        )
        assert isinstance(fig, go.Figure)
        # Should have 2 traces: prefit RMS and postfit RMS
        assert len(fig.data) == 2

    def test_time_units_minutes(self, solved_bls):
        fig = plot_measurement_residual_rms(
            solved_bls, time_units="minutes", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "min" in ax.get_xlabel()
        plt.close(fig)

    def test_rms_non_negative(self, solved_bls):
        fig = plot_measurement_residual_rms(solved_bls, backend="matplotlib")
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert np.all(lines[0].get_ydata() >= 0)
        plt.close(fig)
