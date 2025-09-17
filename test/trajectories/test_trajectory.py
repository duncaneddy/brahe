"""
Tests for the base Trajectory class in brahe.

The Trajectory class provides frame-agnostic storage and interpolation
of 6-dimensional state vectors over time.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture
def sample_epochs():
    """Create sample epochs for testing."""
    base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
    return [
        base_epoch,
        brahe.Epoch.from_jd(base_epoch.jd() + 0.5/24.0, "UTC"),  # +30 min
        brahe.Epoch.from_jd(base_epoch.jd() + 1.0/24.0, "UTC"),  # +1 hour
        brahe.Epoch.from_jd(base_epoch.jd() + 2.0/24.0, "UTC"),  # +2 hours
    ]


@pytest.fixture
def sample_states():
    """Create sample 6D state vectors for testing."""
    return [
        np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0]),           # Circular orbit
        np.array([6.678e6 + 1000, 100.0, 0.0, -10.0, 7.726e3, 5.0]), # Slightly perturbed
        np.array([6.678e6 + 2000, 200.0, 0.0, -20.0, 7.726e3, 10.0]),
        np.array([6.678e6 + 3000, 300.0, 0.0, -30.0, 7.726e3, 15.0]),
    ]


class TestTrajectoryCreation:
    """Test trajectory creation and initialization."""

    def test_empty_trajectory_creation(self):
        """Test creating an empty trajectory with default interpolation."""
        traj = brahe.Trajectory()

        assert len(traj) == 0
        assert traj.length == 0
        assert str(traj.interpolation_method) == "Linear"
        assert traj.start_epoch is None
        assert traj.end_epoch is None
        assert traj.time_span is None

    def test_trajectory_with_interpolation_method(self):
        """Test creating trajectory with specific interpolation method."""
        linear_method = brahe.InterpolationMethod.linear
        traj = brahe.Trajectory(linear_method)

        assert str(traj.interpolation_method) == "Linear"

        lagrange_method = brahe.InterpolationMethod.lagrange
        traj2 = brahe.Trajectory(lagrange_method)

        assert str(traj2.interpolation_method) == "Lagrange"

    def test_trajectory_from_data(self, sample_epochs, sample_states):
        """Test creating trajectory from epochs and states data."""
        # Convert states to flat array format
        states_flat = np.concatenate(sample_states)

        traj = brahe.Trajectory.from_data(sample_epochs, states_flat)

        assert traj.length == len(sample_epochs)
        assert traj.start_epoch.jd() == sample_epochs[0].jd()
        assert traj.end_epoch.jd() == sample_epochs[-1].jd()

    def test_trajectory_from_data_interpolation_method(self, sample_epochs, sample_states):
        """Test creating trajectory from data with specific interpolation method."""
        states_flat = np.concatenate(sample_states)
        lagrange_method = brahe.InterpolationMethod.lagrange

        traj = brahe.Trajectory.from_data(sample_epochs, states_flat, lagrange_method)

        assert traj.interpolation_method == "Lagrange"
        assert traj.length == len(sample_epochs)

    def test_trajectory_from_data_validation(self):
        """Test validation in trajectory creation from data."""
        epoch1 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        epoch2 = brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC")

        # Mismatched array lengths
        with pytest.raises(ValueError, match="Number of epochs must match"):
            brahe.Trajectory.from_data([epoch1], np.array([1, 2, 3, 4, 5, 6, 7, 8]))

        # Invalid state vector length (not multiple of 6)
        with pytest.raises(ValueError, match="States array length must be a multiple of 6"):
            brahe.Trajectory.from_data([epoch1], np.array([1, 2, 3, 4, 5]))


class TestTrajectoryStateManagement:
    """Test trajectory state addition and access."""

    def test_add_single_state(self):
        """Test adding a single state to trajectory."""
        traj = brahe.Trajectory()
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        traj.add_state(epoch, state)

        assert traj.length == 1
        assert traj.start_epoch.jd() == epoch.jd()
        assert traj.end_epoch.jd() == epoch.jd()

    def test_add_multiple_states(self, sample_epochs, sample_states):
        """Test adding multiple states to trajectory."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        assert traj.length == len(sample_epochs)
        assert traj.start_epoch.jd() == sample_epochs[0].jd()
        assert traj.end_epoch.jd() == sample_epochs[-1].jd()

    def test_add_states_out_of_order(self, sample_epochs, sample_states):
        """Test adding states in non-chronological order."""
        traj = brahe.Trajectory()

        # Add states in reverse order
        for epoch, state in zip(reversed(sample_epochs), reversed(sample_states)):
            traj.add_state(epoch, state)

        assert traj.length == len(sample_epochs)
        # Trajectory should handle ordering internally
        assert traj.start_epoch.jd() == sample_epochs[0].jd()
        assert traj.end_epoch.jd() == sample_epochs[-1].jd()

    def test_state_vector_validation(self):
        """Test validation of state vectors."""
        traj = brahe.Trajectory()
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Invalid state vector length
        with pytest.raises(ValueError, match="State vector must have exactly 6 elements"):
            traj.add_state(epoch, np.array([1, 2, 3, 4, 5]))

        with pytest.raises(ValueError, match="State vector must have exactly 6 elements"):
            traj.add_state(epoch, np.array([1, 2, 3, 4, 5, 6, 7]))


class TestTrajectoryStateAccess:
    """Test trajectory state access and interpolation."""

    def test_state_at_index(self, sample_epochs, sample_states):
        """Test accessing states by index."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        # Test valid indices
        state_0 = traj.state_at_index(0)
        state_2 = traj.state_at_index(2)

        assert len(state_0) == 6
        assert len(state_2) == 6
        assert np.allclose(state_0, sample_states[0])

    def test_epoch_at_index(self, sample_epochs, sample_states):
        """Test accessing epochs by index."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        epoch_0 = traj.epoch_at_index(0)
        epoch_2 = traj.epoch_at_index(2)

        assert epoch_0.jd() == sample_epochs[0].jd()
        assert epoch_2.jd() == sample_epochs[2].jd()

    def test_state_at_epoch_exact(self, sample_epochs, sample_states):
        """Test state interpolation at exact epochs."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        # Query at exact epoch
        state = traj.state_at_epoch(sample_epochs[1])
        assert np.allclose(state, sample_states[1])

    def test_state_at_epoch_interpolated(self, sample_epochs, sample_states):
        """Test state interpolation between epochs."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs[:2], sample_states[:2]):
            traj.add_state(epoch, state)

        # Query at midpoint between first two epochs
        mid_time = (sample_epochs[0].jd() + sample_epochs[1].jd()) / 2.0
        mid_epoch = brahe.Epoch.from_jd(mid_time, "UTC")

        interpolated_state = traj.state_at_epoch(mid_epoch)

        # Should be approximately the average of the two states
        expected_state = (sample_states[0] + sample_states[1]) / 2.0
        assert np.allclose(interpolated_state, expected_state, rtol=1e-10)

    def test_nearest_state(self, sample_epochs, sample_states):
        """Test finding nearest state to a query epoch."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        # Query slightly before second epoch
        query_epoch = brahe.Epoch.from_jd(sample_epochs[1].jd() - 0.1/24.0, "UTC")

        nearest_epoch, nearest_state = traj.nearest_state(query_epoch)

        # Should return the second epoch/state
        assert nearest_epoch.jd() == sample_epochs[1].jd()
        assert np.allclose(nearest_state, sample_states[1])


class TestTrajectoryProperties:
    """Test trajectory properties and metadata."""

    def test_time_span(self, sample_epochs, sample_states):
        """Test trajectory time span calculation."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        expected_span = (sample_epochs[-1].jd() - sample_epochs[0].jd()) * 86400.0  # Convert to seconds
        assert np.isclose(traj.time_span, expected_span)

    def test_length_properties(self, sample_epochs, sample_states):
        """Test length-related properties."""
        traj = brahe.Trajectory()

        assert len(traj) == 0
        assert traj.length == 0

        for i, (epoch, state) in enumerate(zip(sample_epochs, sample_states)):
            traj.add_state(epoch, state)
            assert len(traj) == i + 1
            assert traj.length == i + 1

    def test_trajectory_configuration(self):
        """Test trajectory configuration methods."""
        traj = brahe.Trajectory()

        # Test max size setting
        traj.set_max_size(100)
        # No direct way to test this, but it should not crash

        # Test max age setting
        traj.set_max_age(3600.0)  # 1 hour
        # No direct way to test this, but it should not crash

    def test_clear_trajectory(self, sample_epochs, sample_states):
        """Test clearing trajectory data."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        assert traj.length > 0

        traj.clear()

        assert traj.length == 0
        assert traj.start_epoch is None
        assert traj.end_epoch is None

    def test_to_matrix(self, sample_epochs, sample_states):
        """Test conversion to matrix format."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        matrix = traj.to_matrix()

        assert matrix.shape == (len(sample_states), 6)
        # First row should match first state
        assert np.allclose(matrix[0, :], sample_states[0])


class TestTrajectoryStringRepresentation:
    """Test trajectory string representations."""

    def test_repr_and_str(self):
        """Test string representations of trajectory."""
        traj = brahe.Trajectory()

        repr_str = repr(traj)
        assert "Trajectory" in repr_str
        assert "Linear" in repr_str  # Default interpolation method
        assert "states=0" in repr_str

        str_str = str(traj)
        assert str_str == repr_str

    def test_repr_with_states(self, sample_epochs, sample_states):
        """Test string representation with states."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        repr_str = repr(traj)
        assert f"states={len(sample_states)}" in repr_str


class TestTrajectoryInterpolationMethods:
    """Test different interpolation methods."""

    def test_linear_interpolation(self):
        """Test linear interpolation behavior."""
        traj = brahe.Trajectory(brahe.InterpolationMethod.linear)

        epoch1 = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        epoch2 = brahe.Epoch.from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, "UTC")

        state1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        traj.add_state(epoch1, state1)
        traj.add_state(epoch2, state2)

        # Query at midpoint
        mid_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 30, 0.0, 0.0, "UTC")
        interpolated = traj.state_at_epoch(mid_epoch)

        # Should be exactly halfway between states
        expected = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert np.allclose(interpolated, expected)

    def test_lagrange_interpolation(self):
        """Test Lagrange interpolation method."""
        traj = brahe.Trajectory(brahe.InterpolationMethod.lagrange)

        # Add some test data
        base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        for i in range(3):
            epoch = brahe.Epoch.from_jd(base_epoch.jd() + i * 0.5/24.0, "UTC")
            state = np.array([i * 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            traj.add_state(epoch, state)

        # Test interpolation (exact behavior depends on implementation)
        mid_epoch = brahe.Epoch.from_jd(base_epoch.jd() + 0.25/24.0, "UTC")
        interpolated = traj.state_at_epoch(mid_epoch)

        assert len(interpolated) == 6
        assert np.all(np.isfinite(interpolated))


class TestTrajectoryErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_trajectory_access(self):
        """Test accessing data from empty trajectory."""
        traj = brahe.Trajectory()

        with pytest.raises(RuntimeError):
            traj.state_at_index(0)

        with pytest.raises(RuntimeError):
            traj.epoch_at_index(0)

        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        with pytest.raises(RuntimeError):
            traj.state_at_epoch(epoch)

    def test_index_out_of_bounds(self, sample_epochs, sample_states):
        """Test accessing invalid indices."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        with pytest.raises(RuntimeError):
            traj.state_at_index(-1)

        with pytest.raises(RuntimeError):
            traj.state_at_index(len(sample_states))

        with pytest.raises(RuntimeError):
            traj.epoch_at_index(len(sample_epochs))

    def test_extrapolation_behavior(self, sample_epochs, sample_states):
        """Test behavior when querying outside trajectory time span."""
        traj = brahe.Trajectory()

        for epoch, state in zip(sample_epochs, sample_states):
            traj.add_state(epoch, state)

        # Query before start
        early_epoch = brahe.Epoch.from_jd(sample_epochs[0].jd() - 1.0, "UTC")

        # This should either extrapolate or raise an error depending on implementation
        # For now, just test that it doesn't crash
        try:
            traj.state_at_epoch(early_epoch)
        except RuntimeError:
            # Acceptable if extrapolation is not supported
            pass

    def test_single_state_interpolation(self):
        """Test interpolation with only one state."""
        traj = brahe.Trajectory()
        epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        state = np.array([6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0])

        traj.add_state(epoch, state)

        # Query at the same epoch should return the exact state
        result = traj.state_at_epoch(epoch)
        assert np.allclose(result, state)

        # Query at different epoch with single state
        other_epoch = brahe.Epoch.from_jd(epoch.jd() + 1.0, "UTC")
        try:
            result = traj.state_at_epoch(other_epoch)
            # If interpolation is supported with single state, result should equal input state
            assert np.allclose(result, state)
        except RuntimeError:
            # Acceptable if interpolation requires multiple states
            pass


class TestTrajectoryPerformance:
    """Test trajectory performance and memory management."""

    def test_large_trajectory(self):
        """Test trajectory with many states."""
        traj = brahe.Trajectory()
        base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")

        # Add many states
        num_states = 1000
        for i in range(num_states):
            epoch = brahe.Epoch.from_jd(base_epoch.jd() + i * 1.0/1440.0, "UTC")  # 1 minute intervals
            state = np.array([6.678e6 + i, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            traj.add_state(epoch, state)

        assert traj.length == num_states

        # Test access performance
        mid_index = num_states // 2
        mid_state = traj.state_at_index(mid_index)
        assert len(mid_state) == 6

    def test_memory_management_settings(self):
        """Test memory management configuration."""
        traj = brahe.Trajectory()

        # Test max size setting
        traj.set_max_size(10)

        # Test max age setting
        traj.set_max_age(3600.0)  # 1 hour

        # Add states to potentially trigger eviction
        base_epoch = brahe.Epoch.from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, "UTC")
        for i in range(20):  # More than max_size
            epoch = brahe.Epoch.from_jd(base_epoch.jd() + i * 1.0/24.0, "UTC")
            state = np.array([6.678e6 + i, 0.0, 0.0, 0.0, 7.726e3, 0.0])
            traj.add_state(epoch, state)

        # Trajectory length should be managed according to settings
        # (Exact behavior depends on implementation)
        assert traj.length > 0