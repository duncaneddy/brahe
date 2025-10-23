"""
Tests for brahe CLI utility functions and brahe.format_time_string.
"""

import brahe as bh


def test_format_time_string_seconds():
    """Test time string formatting for seconds."""
    assert bh.format_time_string(45.5) == "45.50 seconds"
    assert bh.format_time_string(0.5) == "0.50 seconds"


def test_format_time_string_minutes():
    """Test time string formatting for minutes."""
    assert bh.format_time_string(90.0) == "1 minutes and 30.00 seconds"
    assert bh.format_time_string(125.75) == "2 minutes and 5.75 seconds"


def test_format_time_string_hours():
    """Test time string formatting for hours."""
    assert bh.format_time_string(3665.0) == "1 hours, 1 minutes, and 5.00 seconds"
    assert bh.format_time_string(7200.0) == "2 hours, 0 minutes, and 0.00 seconds"


def test_format_time_string_days():
    """Test time string formatting for days."""
    assert (
        bh.format_time_string(86400.0) == "1 days, 0 hours, 0 minutes, and 0.00 seconds"
    )
    assert (
        bh.format_time_string(90061.5) == "1 days, 1 hours, 1 minutes, and 1.50 seconds"
    )


def test_format_time_string_short_seconds():
    """Test short format for seconds."""
    assert bh.format_time_string(45.5, short=True) == "45s"
    assert bh.format_time_string(0.5, short=True) == "0s"
    assert bh.format_time_string(30.0, short=True) == "30s"


def test_format_time_string_short_minutes():
    """Test short format for minutes."""
    assert bh.format_time_string(90.0, short=True) == "1m 30s"
    assert bh.format_time_string(125.75, short=True) == "2m 5s"


def test_format_time_string_short_hours():
    """Test short format for hours."""
    assert bh.format_time_string(3665.0, short=True) == "1h 1m 5s"
    assert bh.format_time_string(7200.0, short=True) == "2h"


def test_format_time_string_short_days():
    """Test short format for days."""
    assert bh.format_time_string(86400.0, short=True) == "1d"
    assert bh.format_time_string(90061.5, short=True) == "1d 1h 1m 1s"
