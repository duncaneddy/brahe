"""Tests for 3D trajectory plotting with texture support."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import brahe as bh

pytestmark = pytest.mark.ci


@pytest.fixture
def test_trajectory():
    """Create a simple test trajectory."""
    # Create simple LEO orbit
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    duration = bh.orbital_period(oe[0])
    prop.propagate_to(epoch + duration)
    traj = prop.trajectory

    return traj


def test_plot_trajectory_3d_matplotlib_simple(test_trajectory):
    """Test 3D trajectory plotting with matplotlib and simple texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="simple",
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_matplotlib_blue_marble(test_trajectory):
    """Test 3D trajectory plotting with matplotlib and blue marble texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="blue_marble",
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_matplotlib_default(test_trajectory):
    """Test that matplotlib defaults to 'simple' texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_plotly_simple(test_trajectory):
    """Test 3D trajectory plotting with plotly and simple texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="simple",
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_plotly_blue_marble(test_trajectory):
    """Test 3D trajectory plotting with plotly and blue marble texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="blue_marble",
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_plotly_default(test_trajectory):
    """Test that plotly defaults to 'blue_marble' texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.ci
def test_plot_trajectory_3d_natural_earth_50m(test_trajectory):
    """Test 3D trajectory plotting with Natural Earth 50m texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="natural_earth_50m",
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_no_earth(test_trajectory):
    """Test 3D trajectory plotting without Earth sphere."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=False,
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_invalid_texture(test_trajectory):
    """Test that invalid texture name raises ValueError (plotly only)."""
    with pytest.raises(ValueError, match="Unknown texture name"):
        bh.plot_trajectory_3d(
            [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
            units="km",
            show_earth=True,
            earth_texture="invalid_texture",
            backend="plotly",
        )


def test_plot_trajectory_3d_multiple_trajectories(test_trajectory):
    """Test plotting multiple trajectories with textures."""
    # Create second trajectory at different altitude
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe2 = np.array([bh.R_EARTH + 800e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
    state2 = bh.state_koe_to_eci(oe2, bh.AngleFormat.RADIANS)
    prop2 = bh.KeplerianPropagator.from_eci(epoch, state2, 60.0)
    duration2 = bh.orbital_period(oe2[0])
    prop2.propagate_to(epoch + duration2)
    traj2 = prop2.trajectory

    fig = bh.plot_trajectory_3d(
        [
            {"trajectory": test_trajectory, "color": "red", "label": "Orbit 1"},
            {"trajectory": traj2, "color": "blue", "label": "Orbit 2"},
        ],
        units="km",
        show_earth=True,
        earth_texture="blue_marble",
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_normalized_units(test_trajectory):
    """Test plotting with normalized Earth radii units."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        normalize=True,
        show_earth=True,
        earth_texture="blue_marble",
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_custom_view(test_trajectory):
    """Test plotting with custom view angles."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_earth=True,
        earth_texture="simple",
        view_azimuth=60.0,
        view_elevation=20.0,
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
