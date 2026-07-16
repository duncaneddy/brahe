"""Tests for 3D trajectory plotting with texture support."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import brahe as bh


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


@pytest.fixture
def eci_trajectory(test_trajectory):
    """Alias of test_trajectory for tests exercising the central_body API."""
    return test_trajectory


@pytest.fixture
def lunar_trajectory():
    """Small circular trajectory in the Moon-centered inertial frame (NAIF 301)."""
    epoch0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    n = 20
    radius = bh.R_MOON + 100e3
    speed = 1600.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    epochs = [epoch0 + i * 60.0 for i in range(n)]
    states = np.zeros((n, 6))
    states[:, 0] = radius * np.cos(angles)
    states[:, 1] = radius * np.sin(angles)
    states[:, 3] = -speed * np.sin(angles)
    states[:, 4] = speed * np.cos(angles)

    return bh.OrbitTrajectory.from_orbital_data(
        epochs,
        states,
        bh.OrbitFrame.BodyCenteredInertial(301),
        bh.OrbitRepresentation.CARTESIAN,
        None,
        None,
    )


def test_plot_trajectory_3d_matplotlib_simple(test_trajectory):
    """Test 3D trajectory plotting with matplotlib and simple texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_body=True,
        texture="simple",
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
        show_body=True,
        texture="blue_marble",
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
        show_body=True,
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
        show_body=True,
        texture="simple",
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_plotly_blue_marble(test_trajectory):
    """Test 3D trajectory plotting with plotly and blue marble texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_body=True,
        texture="blue_marble",
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_plotly_default(test_trajectory):
    """Test that plotly defaults to 'blue_marble' texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_body=True,
        backend="plotly",
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_plot_trajectory_3d_natural_earth_50m(test_trajectory):
    """Test 3D trajectory plotting with Natural Earth 50m texture."""
    fig = bh.plot_trajectory_3d(
        [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
        units="km",
        show_body=True,
        texture="natural_earth_50m",
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
        show_body=False,
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d_invalid_texture(test_trajectory):
    """Test that invalid texture name raises ValueError (plotly only)."""
    with pytest.raises(ValueError, match="Unknown texture"):
        bh.plot_trajectory_3d(
            [{"trajectory": test_trajectory, "color": "red", "label": "Test"}],
            units="km",
            show_body=True,
            texture="invalid_texture",
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
        show_body=True,
        texture="blue_marble",
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
        show_body=True,
        texture="blue_marble",
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
        show_body=True,
        texture="simple",
        view_azimuth=60.0,
        view_elevation=20.0,
        backend="matplotlib",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_resolve_body_registry_and_custom():
    from brahe.plots.bodies import resolve_body

    moon = resolve_body("moon")
    assert moon["radius"] == pytest.approx(bh.R_MOON)
    assert moon["naif_id"] == 301
    custom = resolve_body({"name": "Ceres", "radius": 469.7e3, "texture": "ceres"})
    assert custom["radius"] == 469.7e3
    with pytest.raises(ValueError):
        resolve_body("kerbin")


def test_plot_trajectory_3d_moon_centered(lunar_trajectory):
    fig = bh.plot_trajectory_3d(
        [{"trajectory": lunar_trajectory, "label": "LLO"}],
        central_body="moon",
        texture="simple",
        backend="matplotlib",
    )
    assert fig is not None
    plt.close(fig)


def test_plot_trajectory_3d_additional_bodies(eci_trajectory):
    fig = bh.plot_trajectory_3d(
        [{"trajectory": eci_trajectory}],
        texture="simple",
        additional_bodies=[
            {
                "position": [384.4e6, 0.0, 0.0],
                "radius": bh.R_MOON,
                "texture": "simple",
                "name": "Moon",
            }
        ],
        backend="plotly",
    )
    names = [t.name for t in fig.data]
    assert "Moon" in names


def test_plot_trajectory_3d_rejects_removed_kwargs(eci_trajectory):
    with pytest.raises(TypeError):
        bh.plot_trajectory_3d([{"trajectory": eci_trajectory}], earth_texture="simple")
