"""
Tests for threading utilities.

NOTE: These tests cannot be run in the same process as other tests that use
the thread pool, as the thread pool is a global singleton that cannot be
reinitialized once set. Each test should be run in isolation.
"""

import pytest
import brahe as bh


def test_get_max_threads_default():
    """Test getting default thread count."""
    # Note: This may initialize the thread pool with defaults if not already initialized
    threads = bh.get_max_threads()

    # Should be at least 1
    assert threads >= 1
    # Default should be reasonable (not more than CPU count)
    # We can't test exact value as it depends on when pool was initialized


def test_set_num_threads():
    """Test setting specific number of threads.

    Note: This test will only work if run before any parallel operations.
    If the thread pool is already initialized, it should raise RuntimeError.
    """
    try:
        bh.set_num_threads(4)
        assert bh.get_max_threads() == 4
    except RuntimeError:
        # Thread pool already initialized - this is expected if other tests ran first
        pytest.skip("Thread pool already initialized - cannot test set_num_threads")


def test_set_num_threads_invalid():
    """Test that setting 0 threads raises an error."""
    with pytest.raises(RuntimeError):
        bh.set_num_threads(0)


def test_set_max_threads():
    """Test setting threads to maximum (all CPU cores).

    Note: This test will only work if run before any parallel operations.
    If the thread pool is already initialized, it should raise RuntimeError.
    """
    try:
        bh.set_max_threads()
        threads = bh.get_max_threads()
        # Should be at least 1
        assert threads >= 1
    except RuntimeError:
        # Thread pool already initialized - this is expected if other tests ran first
        pytest.skip("Thread pool already initialized - cannot test set_max_threads")


def test_set_ludicrous_speed():
    """Test setting threads to ludicrous speed (alias for set_max_threads).

    Note: This test will only work if run before any parallel operations.
    If the thread pool is already initialized, it should raise RuntimeError.
    """
    try:
        bh.set_ludicrous_speed()
        threads = bh.get_max_threads()
        # Should be at least 1
        assert threads >= 1
    except RuntimeError:
        # Thread pool already initialized - this is expected if other tests ran first
        pytest.skip("Thread pool already initialized - cannot test set_ludicrous_speed")
