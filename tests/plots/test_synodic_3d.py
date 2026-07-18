"""Tests for synodic-frame 3D trajectory plotting."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture
def leo_trajectory():
    epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    prop.propagate_to(epoch + 5400.0)
    return prop.trajectory


@pytest.mark.integration  # requires the DE440s SPICE kernel
def test_plot_synodic_3d_emr_alias(leo_trajectory, naif_cache_setup):
    fig = bh.plot_synodic_3d(
        [{"trajectory": leo_trajectory, "label": "LEO"}],
        frame="EMR",
        backend="plotly",
    )
    names = [t.name for t in fig.data]
    assert "Earth" in names and "Moon" in names


@pytest.mark.integration
def test_plot_earth_moon_rotating_3d(leo_trajectory, naif_cache_setup):
    fig = bh.plot_earth_moon_rotating_3d([leo_trajectory], backend="matplotlib")
    assert fig is not None


def test_plot_synodic_3d_rejects_non_synodic_frame(leo_trajectory):
    with pytest.raises(ValueError):
        bh.plot_synodic_3d([leo_trajectory], frame=bh.ReferenceFrame.GCRF)


def test_plot_synodic_3d_rejects_non_trajectory():
    with pytest.raises(TypeError):
        bh.plot_synodic_3d([np.zeros((10, 6))], frame="EMR")


def test_plot_synodic_3d_rejects_empty_trajectories_without_reference_epoch():
    with pytest.raises(ValueError):
        bh.plot_synodic_3d([], frame="EMR")


@pytest.mark.integration  # requires the DE440s SPICE kernel
def test_plot_synodic_3d_extra_body_without_texture(leo_trajectory, naif_cache_setup):
    fig = bh.plot_synodic_3d(
        [{"trajectory": leo_trajectory, "label": "LEO"}],
        frame="EMR",
        bodies=[{"position": [1.0e7, 0.0, 0.0], "radius": 1.0e6, "name": "Custom"}],
        backend="plotly",
    )
    names = [t.name for t in fig.data]
    assert "Custom" in names
