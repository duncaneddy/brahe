"""
Tests for threading utilities.
"""

import pytest
import brahe as bh


def test_get_max_threads_default():
    """Test getting default thread count."""
    threads = bh.get_max_threads()

    # Should be at least 1
    assert threads >= 1


def test_set_num_threads():
    """Test setting specific number of threads."""
    bh.set_num_threads(4)
    assert bh.get_max_threads() == 4


def test_set_num_threads_reinitialize():
    """Test that thread pool can be reinitialized with different thread counts."""
    # Set to 2 threads
    bh.set_num_threads(2)
    assert bh.get_max_threads() == 2

    # Reinitialize with 4 threads
    bh.set_num_threads(4)
    assert bh.get_max_threads() == 4

    # Reinitialize with 1 thread
    bh.set_num_threads(1)
    assert bh.get_max_threads() == 1


def test_set_num_threads_invalid():
    """Test that setting 0 threads raises an error."""
    with pytest.raises(ValueError, match="Number of threads must be at least 1"):
        bh.set_num_threads(0)


def test_set_max_threads():
    """Test setting threads to maximum (all CPU cores)."""
    bh.set_max_threads()
    threads = bh.get_max_threads()
    # Should be at least 1
    assert threads >= 1


def test_set_max_threads_reinitialize():
    """Test that set_max_threads can be called multiple times."""
    bh.set_max_threads()
    threads1 = bh.get_max_threads()

    # Call again - should not raise
    bh.set_max_threads()
    threads2 = bh.get_max_threads()

    assert threads1 == threads2


def test_set_ludicrous_speed():
    """Test setting threads to ludicrous speed (alias for set_max_threads)."""
    bh.set_ludicrous_speed()
    threads = bh.get_max_threads()
    # Should be at least 1
    assert threads >= 1


def test_set_ludicrous_speed_reinitialize():
    """Test that set_ludicrous_speed can be called multiple times."""
    bh.set_ludicrous_speed()
    threads1 = bh.get_max_threads()

    # Call again - should not raise
    bh.set_ludicrous_speed()
    threads2 = bh.get_max_threads()

    assert threads1 == threads2


def test_mixed_function_reinitialization():
    """Test that different thread configuration functions work together."""
    # Start with specific thread count
    bh.set_num_threads(2)
    assert bh.get_max_threads() == 2

    # Switch to max threads
    bh.set_max_threads()
    max_threads = bh.get_max_threads()
    assert max_threads >= 2  # Should be >= our previous setting

    # Switch to ludicrous speed (should be same as max)
    bh.set_ludicrous_speed()
    assert bh.get_max_threads() == max_threads

    # Go back to specific count
    bh.set_num_threads(1)
    assert bh.get_max_threads() == 1

    # Back to max again
    bh.set_max_threads()
    assert bh.get_max_threads() == max_threads


def test_get_max_threads_after_set():
    """Test that get_max_threads returns the value set by set_num_threads."""
    test_values = [1, 2, 4, 8]

    for n in test_values:
        bh.set_num_threads(n)
        assert bh.get_max_threads() == n, (
            f"Expected {n} threads, got {bh.get_max_threads()}"
        )
