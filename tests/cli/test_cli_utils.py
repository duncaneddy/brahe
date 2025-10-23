"""
Tests for brahe CLI utility functions.

Note: epoch_from_epochlike() was removed in favor of direct Epoch(value) usage.
Those tests are now in tests/time/test_epoch_float_init.py
"""

from brahe.cli.utils import get_time_string


def test_get_time_string_seconds():
    """Test time string formatting for seconds."""
    assert get_time_string(45.5) == "45.50 seconds"
    assert get_time_string(0.5) == "0.50 seconds"


def test_get_time_string_minutes():
    """Test time string formatting for minutes."""
    assert get_time_string(90.0) == "1 minutes and 30.00 seconds"
    assert get_time_string(125.75) == "2 minutes and 5.75 seconds"


def test_get_time_string_hours():
    """Test time string formatting for hours."""
    assert get_time_string(3665.0) == "1 hours, 1 minutes, and 5.00 seconds"
    assert get_time_string(7200.0) == "2 hours, 0 minutes, and 0.00 seconds"


def test_get_time_string_days():
    """Test time string formatting for days."""
    assert get_time_string(86400.0) == "1 days, 0 hours, 0 minutes, and 0.00 seconds"
    assert get_time_string(90061.5) == "1 days, 1 hours, 1 minutes, and 1.50 seconds"


def test_get_time_string_short_seconds():
    """Test short format for seconds."""
    assert get_time_string(45.5, short=True) == "46s"
    assert get_time_string(0.5, short=True) == "0s"
    assert get_time_string(30.0, short=True) == "30s"


def test_get_time_string_short_minutes():
    """Test short format for minutes."""
    assert get_time_string(90.0, short=True) == "1m 30s"
    assert get_time_string(125.75, short=True) == "2m 6s"


def test_get_time_string_short_hours():
    """Test short format for hours."""
    assert get_time_string(3665.0, short=True) == "1h 1m 5s"
    assert get_time_string(7200.0, short=True) == "2h"


def test_get_time_string_short_days():
    """Test short format for days."""
    assert get_time_string(86400.0, short=True) == "1d"
    assert get_time_string(90061.5, short=True) == "1d 1h 1m 2s"
