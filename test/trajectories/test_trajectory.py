"""
Tests for the base Trajectory class in brahe.

These tests mirror the Rust test suite to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture(scope="session", autouse=True)
def setup_eop_provider():
    """Set up EOP provider for tests that require ECEF frame conversions."""
    provider = brahe.StaticEOPProvider.from_zero()
    brahe.set_global_eop_provider_from_static_provider(provider)


@pytest.fixture
def sample_epochs():
    """Create sample epochs for testing."""
    base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    return [
        base_epoch,
        brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC"),
        brahe.Epoch.from_datetime(2023, 1, 1, 14, 0, 0.0, 0.0, "UTC"),
    ]


@pytest.fixture
def sample_states():
    """Create sample 6-element state vectors."""
    return [
        np.array([1000.0, 2000.0, 3000.0, 1.0, 2.0, 3.0]),
        np.array([1100.0, 2100.0, 3100.0, 1.1, 2.1, 3.1]),
        np.array([1200.0, 2200.0, 3200.0, 1.2, 2.2, 3.2]),
    ]


class TestTrajectoryCreation:
    """Test trajectory creation that mirrors Rust tests."""

    def test_trajectory_creation(self):
        """Test basic trajectory creation."""
        trajectory = brahe.Trajectory()
        assert len(trajectory) == 0

    def test_trajectory_with_interpolation(self):
        """Test trajectory creation with specific interpolation."""
        linear_interp = brahe.InterpolationMethod.linear
        trajectory = brahe.Trajectory(linear_interp)
        assert len(trajectory) == 0


class TestTrajectoryStateManagement:
    """Test trajectory state management that mirrors Rust tests."""

    def test_trajectory_add_state(self, sample_epochs, sample_states):
        """Test adding states to trajectory."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        assert len(trajectory) == 3

    def test_trajectory_indexing(self, sample_epochs, sample_states):
        """Test trajectory indexing."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test indexing
        first_epoch = trajectory.epoch_at_index(0)
        assert first_epoch.jd() == sample_epochs[0].jd()

    def test_trajectory_nearest_state(self, sample_epochs, sample_states):
        """Test finding nearest state."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test nearest state
        nearest_epoch, nearest_state = trajectory.nearest_state(sample_epochs[1])
        assert nearest_epoch.jd() == sample_epochs[1].jd()
        np.testing.assert_array_almost_equal(nearest_state, sample_states[1])


class TestTrajectoryInterpolation:
    """Test trajectory interpolation that mirrors Rust tests."""

    def test_trajectory_linear_interpolation(self, sample_epochs, sample_states):
        """Test linear interpolation."""
        trajectory = brahe.Trajectory(brahe.InterpolationMethod.linear)

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        # Test interpolation at midpoint
        mid_time_jd = (sample_epochs[0].jd() + sample_epochs[1].jd()) / 2.0
        mid_epoch = brahe.Epoch.from_jd(mid_time_jd, "UTC")
        interpolated_state = trajectory.state_at_epoch(mid_epoch)

        # Should be roughly halfway between first two states
        expected_state = (sample_states[0] + sample_states[1]) / 2.0
        np.testing.assert_array_almost_equal(interpolated_state, expected_state, decimal=1)


class TestTrajectoryProperties:
    """Test trajectory properties that mirror Rust tests."""

    def test_trajectory_to_matrix(self, sample_epochs, sample_states):
        """Test converting trajectory to matrix."""
        trajectory = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            trajectory.add_state(epoch, state)

        matrix = trajectory.to_matrix()
        assert matrix.shape == (6, 3)  # 6 elements, 3 states

        # Check that we can access matrix elements (implementation-specific organization)
        assert matrix[0, 0] == sample_states[0][0]  # First element of first state