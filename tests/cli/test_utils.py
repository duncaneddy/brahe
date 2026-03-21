"""
Tests for brahe CLI utility functions and brahe.format_time_string.
"""

import pytest
import brahe as bh
from brahe.cli.utils import parse_numeric_expression, set_cli_eop


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


# =============================================================================
# parse_numeric_expression tests
# =============================================================================


def test_parse_numeric_expression_simple_float():
    assert parse_numeric_expression("123.45") == 123.45


def test_parse_numeric_expression_scientific_notation():
    assert parse_numeric_expression("500e3") == 500000.0


def test_parse_numeric_expression_int_passthrough():
    assert parse_numeric_expression(42) == 42.0


def test_parse_numeric_expression_float_passthrough():
    assert parse_numeric_expression(3.14) == 3.14


def test_parse_numeric_expression_brahe_constant():
    result = parse_numeric_expression("R_EARTH")
    assert abs(result - bh.R_EARTH) < 1.0


def test_parse_numeric_expression_expression():
    result = parse_numeric_expression("R_EARTH+500e3")
    assert abs(result - (bh.R_EARTH + 500e3)) < 1.0


def test_parse_numeric_expression_math_function():
    result = parse_numeric_expression("sqrt(4)")
    assert abs(result - 2.0) < 1e-10


def test_parse_numeric_expression_complex():
    result = parse_numeric_expression("2*R_EARTH")
    assert abs(result - 2 * bh.R_EARTH) < 1.0


def test_parse_numeric_expression_with_parentheses():
    result = parse_numeric_expression("(R_EARTH+500e3)*2")
    assert abs(result - (bh.R_EARTH + 500e3) * 2) < 1.0


def test_parse_numeric_expression_pi():
    result = parse_numeric_expression("pi")
    assert abs(result - 3.141592653589793) < 1e-10


def test_parse_numeric_expression_invalid_chars():
    with pytest.raises(ValueError, match="Invalid characters"):
        parse_numeric_expression("R_EARTH; rm -rf /")


def test_parse_numeric_expression_unsafe_import():
    with pytest.raises(ValueError):
        parse_numeric_expression("__import__('os')")


def test_parse_numeric_expression_unsafe_exec():
    with pytest.raises(ValueError):
        parse_numeric_expression("exec('print(1)')")


def test_parse_numeric_expression_nonfinite():
    with pytest.raises(ValueError):
        parse_numeric_expression("log(0)")


# =============================================================================
# set_cli_eop tests
# =============================================================================


def test_set_cli_eop():
    """Verify set_cli_eop runs without error."""
    set_cli_eop()
