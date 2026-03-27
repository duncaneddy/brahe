"""Tests for estimation_state array-API plotting functions."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from brahe.plots.estimation_state import (
    plot_estimator_state_error_from_arrays,
    plot_estimator_state_value_from_arrays,
    plot_estimator_state_error_grid_from_arrays,
    plot_estimator_state_value_grid_from_arrays,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def single_series_error():
    """Single time series with errors and sigmas."""
    n = 50
    times = np.linspace(0, 100, n)
    errors = np.random.default_rng(0).normal(0, 1, n)
    sigmas = np.abs(np.random.default_rng(1).normal(2, 0.5, n))
    return times, errors, sigmas


@pytest.fixture
def multi_series_error():
    """Two time series with errors and sigmas."""
    n = 40
    rng = np.random.default_rng(42)
    times_a = np.linspace(0, 80, n)
    times_b = np.linspace(0, 80, n)
    errors_a = rng.normal(0, 1, n)
    errors_b = rng.normal(0, 1.5, n)
    sigmas_a = np.abs(rng.normal(2, 0.3, n))
    sigmas_b = np.abs(rng.normal(3, 0.4, n))
    return (
        [times_a, times_b],
        [errors_a, errors_b],
        [sigmas_a, sigmas_b],
    )


@pytest.fixture
def single_series_value():
    """Single time series of estimated values with truth."""
    n = 50
    rng = np.random.default_rng(7)
    times = np.linspace(0, 100, n)
    values = rng.normal(10, 1, n)
    true_values = np.full(n, 10.0)
    sigmas = np.abs(rng.normal(1, 0.2, n))
    return times, values, true_values, sigmas


@pytest.fixture
def grid_3state_data():
    """Multi-series 2D error data for 3-state grid (single series)."""
    n = 40
    n_states = 3
    rng = np.random.default_rng(99)
    times = np.linspace(0, 100, n)
    errors = rng.normal(0, 1, (n, n_states))
    sigmas = np.abs(rng.normal(2, 0.5, (n, n_states)))
    return times, errors, sigmas


@pytest.fixture
def grid_6state_data():
    """Multi-series 2D error data for 6-state grid."""
    n = 30
    n_states = 6
    rng = np.random.default_rng(5)
    times = np.linspace(0, 60, n)
    errors = rng.normal(0, 1, (n, n_states))
    sigmas = np.abs(rng.normal(1.5, 0.3, (n, n_states)))
    true_values = rng.normal(5, 0.1, (n, n_states))
    return times, errors, sigmas, true_values


# =============================================================================
# plot_estimator_state_error_from_arrays — matplotlib
# =============================================================================


class TestStateErrorMatplotlib:
    def test_single_series_returns_figure(self, single_series_error):
        times, errors, sigmas = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_series_with_sigmas(self, single_series_error):
        times, errors, sigmas = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], sigmas=[sigmas], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_series(self, multi_series_error):
        times_list, errors_list, sigmas_list = multi_series_error
        fig = plot_estimator_state_error_from_arrays(
            times_list, errors_list, sigmas=sigmas_list, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_state_label(self, single_series_error):
        times, errors, _ = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], state_label="Position Error [m]", backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        # Check ylabel is set
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Position Error [m]"
        plt.close(fig)

    def test_default_state_label(self, single_series_error):
        times, errors, _ = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_ylabel() == "State Error"
        plt.close(fig)

    def test_with_measurements_overlay(self, single_series_error):
        times, errors, _ = single_series_error
        meas_times = np.linspace(0, 100, 10)
        meas_values = np.random.default_rng(3).normal(0, 0.5, 10)
        fig = plot_estimator_state_error_from_arrays(
            [times],
            [errors],
            measurements=(meas_times, meas_values),
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self, multi_series_error):
        times_list, errors_list, _ = multi_series_error
        fig = plot_estimator_state_error_from_arrays(
            times_list,
            errors_list,
            labels=["Filter A", "Filter B"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backend_config_figsize(self, single_series_error):
        times, errors, _ = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times],
            [errors],
            backend="matplotlib",
            backend_config={"figsize": (8, 4)},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_estimator_state_error_from_arrays — plotly
# =============================================================================


class TestStateErrorPlotly:
    def test_single_series_returns_figure(self, single_series_error):
        times, errors, sigmas = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_single_series_with_sigmas(self, single_series_error):
        times, errors, sigmas = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], sigmas=[sigmas], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_multi_series(self, multi_series_error):
        times_list, errors_list, sigmas_list = multi_series_error
        fig = plot_estimator_state_error_from_arrays(
            times_list, errors_list, sigmas=sigmas_list, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_state_label(self, single_series_error):
        times, errors, _ = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times], [errors], state_label="Velocity Error [m/s]", backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_measurements_overlay(self, single_series_error):
        times, errors, _ = single_series_error
        meas_times = np.linspace(0, 100, 10)
        meas_values = np.zeros(10)
        fig = plot_estimator_state_error_from_arrays(
            [times],
            [errors],
            measurements=(meas_times, meas_values),
            backend="plotly",
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, single_series_error):
        times, errors, _ = single_series_error
        fig = plot_estimator_state_error_from_arrays(
            [times],
            [errors],
            backend="plotly",
            backend_config={"width": 800, "height": 400},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_estimator_state_value_from_arrays — matplotlib
# =============================================================================


class TestStateValueMatplotlib:
    def test_single_series_returns_figure(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_true_values(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], true_values=true_values, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_sigmas(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], sigmas=[sigmas], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_true_values_and_sigmas(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times],
            [values],
            true_values=true_values,
            sigmas=[sigmas],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_state_label(self, single_series_value):
        times, values, _, _ = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], state_label="X Position [m]", backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_ylabel() == "X Position [m]"
        plt.close(fig)

    def test_default_state_label(self, single_series_value):
        times, values, _, _ = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], backend="matplotlib"
        )
        ax = fig.axes[0]
        assert ax.get_ylabel() == "State Value"
        plt.close(fig)

    def test_multi_series(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times, times],
            [values, values * 1.01],
            labels=["EKF", "UKF"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_estimator_state_value_from_arrays — plotly
# =============================================================================


class TestStateValuePlotly:
    def test_single_series_returns_figure(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_values(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], true_values=true_values, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_sigmas(self, single_series_value):
        times, values, true_values, sigmas = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], sigmas=[sigmas], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_without_true_values(self, single_series_value):
        times, values, _, _ = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_state_label(self, single_series_value):
        times, values, _, _ = single_series_value
        fig = plot_estimator_state_value_from_arrays(
            [times], [values], state_label="Y Position [m]", backend="plotly"
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_estimator_state_error_grid_from_arrays — matplotlib
# =============================================================================


class TestStateErrorGridMatplotlib:
    def test_single_series_3_states(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_series_with_sigmas(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], sigmas=[sigmas], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_6_states_default_ncols(self, grid_6state_data):
        times, errors, sigmas, _ = grid_6state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        # 6 states with ncols=3 -> 2x3 grid
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_ncols(self, grid_6state_data):
        times, errors, sigmas, _ = grid_6state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], ncols=2, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_state_labels(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        state_labels = ["Pos X [m]", "Pos Y [m]", "Pos Z [m]"]
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], state_labels=state_labels, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        # Check that state labels appear as y-axis labels
        axes = fig.axes
        for i, label in enumerate(state_labels):
            assert axes[i].get_ylabel() == label
        plt.close(fig)

    def test_default_state_labels(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        axes = fig.axes
        assert axes[0].get_ylabel() == "State 0"
        assert axes[1].get_ylabel() == "State 1"
        assert axes[2].get_ylabel() == "State 2"
        plt.close(fig)

    def test_multi_series(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times, times],
            [errors, errors * 0.9],
            labels=["BLS", "EKF"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_backend_config_figsize(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="matplotlib", backend_config={"figsize": (12, 8)}
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_estimator_state_error_grid_from_arrays — plotly
# =============================================================================


class TestStateErrorGridPlotly:
    def test_single_series_3_states(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_single_series_with_sigmas(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], sigmas=[sigmas], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_6_states_default_ncols(self, grid_6state_data):
        times, errors, sigmas, _ = grid_6state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_ncols(self, grid_6state_data):
        times, errors, sigmas, _ = grid_6state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], ncols=2, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_state_labels(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        state_labels = ["Pos X [m]", "Pos Y [m]", "Pos Z [m]"]
        fig = plot_estimator_state_error_grid_from_arrays(
            [times], [errors], state_labels=state_labels, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_multi_series(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times, times],
            [errors, errors * 0.95],
            labels=["BLS", "EKF"],
            backend="plotly",
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        fig = plot_estimator_state_error_grid_from_arrays(
            [times],
            [errors],
            backend="plotly",
            backend_config={"width": 1200, "height": 600},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# plot_estimator_state_value_grid_from_arrays — matplotlib
# =============================================================================


class TestStateValueGridMatplotlib:
    def test_single_series_3_states(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [errors], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_true_values(self, grid_6state_data):
        times, errors, sigmas, true_values = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [errors], true_values=true_values, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_sigmas(self, grid_3state_data):
        times, errors, sigmas = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [errors], sigmas=[sigmas], backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_true_values_and_sigmas(self, grid_6state_data):
        times, values, sigmas, true_values = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times],
            [values],
            true_values=true_values,
            sigmas=[sigmas],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_ncols(self, grid_6state_data):
        times, values, _, _ = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], ncols=2, backend="matplotlib"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_state_labels(self, grid_3state_data):
        times, values, _ = grid_3state_data
        state_labels = ["x", "y", "z"]
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], state_labels=state_labels, backend="matplotlib"
        )
        axes = fig.axes
        assert axes[0].get_ylabel() == "x"
        assert axes[1].get_ylabel() == "y"
        assert axes[2].get_ylabel() == "z"
        plt.close(fig)

    def test_default_state_labels(self, grid_3state_data):
        times, values, _ = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], backend="matplotlib"
        )
        axes = fig.axes
        assert axes[0].get_ylabel() == "State 0"
        plt.close(fig)

    def test_multi_series(self, grid_3state_data):
        times, values, _ = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times, times],
            [values, values * 1.01],
            labels=["EKF", "UKF"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# plot_estimator_state_value_grid_from_arrays — plotly
# =============================================================================


class TestStateValueGridPlotly:
    def test_single_series_3_states(self, grid_3state_data):
        times, values, _ = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_values(self, grid_6state_data):
        times, values, _, true_values = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], true_values=true_values, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_sigmas(self, grid_3state_data):
        times, values, sigmas = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], sigmas=[sigmas], backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_values_and_sigmas(self, grid_6state_data):
        times, values, sigmas, true_values = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times],
            [values],
            true_values=true_values,
            sigmas=[sigmas],
            backend="plotly",
        )
        assert isinstance(fig, go.Figure)

    def test_custom_ncols(self, grid_6state_data):
        times, values, _, _ = grid_6state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times], [values], ncols=2, backend="plotly"
        )
        assert isinstance(fig, go.Figure)

    def test_multi_series(self, grid_3state_data):
        times, values, _ = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times, times],
            [values, values * 0.99],
            labels=["Solver A", "Solver B"],
            backend="plotly",
        )
        assert isinstance(fig, go.Figure)

    def test_backend_config_dimensions(self, grid_3state_data):
        times, values, _ = grid_3state_data
        fig = plot_estimator_state_value_grid_from_arrays(
            [times],
            [values],
            backend="plotly",
            backend_config={"width": 1000, "height": 700},
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# Invalid backend
# =============================================================================


class TestInvalidBackend:
    def test_error_raises_on_invalid_backend(self, single_series_error):
        times, errors, _ = single_series_error
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_estimator_state_error_from_arrays([times], [errors], backend="bokeh")

    def test_value_raises_on_invalid_backend(self, single_series_value):
        times, values, _, _ = single_series_value
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_estimator_state_value_from_arrays([times], [values], backend="seaborn")

    def test_error_grid_raises_on_invalid_backend(self, grid_3state_data):
        times, errors, _ = grid_3state_data
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_estimator_state_error_grid_from_arrays(
                [times], [errors], backend="bokeh"
            )

    def test_value_grid_raises_on_invalid_backend(self, grid_3state_data):
        times, values, _ = grid_3state_data
        with pytest.raises(ValueError, match="Invalid backend"):
            plot_estimator_state_value_grid_from_arrays(
                [times], [values], backend="bokeh"
            )
