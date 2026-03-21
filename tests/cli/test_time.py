"""Tests for brahe CLI time commands."""

from typer.testing import CliRunner
from brahe.cli.__main__ import app

app.rich_markup_mode = None
runner = CliRunner()


# =============================================================================
# convert command
# =============================================================================


def test_convert_string_to_mjd():
    result = runner.invoke(
        app, ["time", "convert", "2024-01-01 00:00:00 UTC", "string", "mjd"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # MJD for 2024-01-01 is ~60310
    assert 60309 < val < 60311


def test_convert_mjd_to_string():
    result = runner.invoke(app, ["time", "convert", "60310.0", "mjd", "string"])
    assert result.exit_code == 0
    assert "2024" in result.stdout


def test_convert_jd_to_string():
    result = runner.invoke(app, ["time", "convert", "2460310.5", "jd", "string"])
    assert result.exit_code == 0
    assert "2024" in result.stdout


def test_convert_string_to_gps_date():
    result = runner.invoke(
        app, ["time", "convert", "2024-01-01 00:00:00 UTC", "string", "gps_date"]
    )
    assert result.exit_code == 0
    # Returns a tuple (week, seconds)
    assert "2295" in result.stdout


def test_convert_string_to_gps_nanoseconds():
    result = runner.invoke(
        app,
        ["time", "convert", "2024-01-01 00:00:00 UTC", "string", "gps_nanoseconds"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0


def test_convert_with_input_time_system():
    result = runner.invoke(
        app,
        [
            "time",
            "convert",
            "60310.0",
            "mjd",
            "string",
            "--input-time-system",
            "UTC",
        ],
    )
    assert result.exit_code == 0
    assert "2024" in result.stdout


def test_convert_with_output_time_system():
    result = runner.invoke(
        app,
        [
            "time",
            "convert",
            "2024-01-01 00:00:00 UTC",
            "string",
            "mjd",
            "--output-time-system",
            "GPS",
        ],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0


def test_convert_mjd_to_jd():
    result = runner.invoke(app, ["time", "convert", "60310.0", "mjd", "jd"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # JD = MJD + 2400000.5
    assert abs(val - 2460310.5) < 0.001


# =============================================================================
# add command
# =============================================================================


def test_add_basic():
    result = runner.invoke(app, ["time", "add", "2024-01-01 00:00:00 UTC", "3600"])
    assert result.exit_code == 0
    # Should be one hour later
    assert "01:00:00" in result.stdout or "1:00:00" in result.stdout


def test_add_with_output_format_mjd():
    result = runner.invoke(
        app,
        [
            "time",
            "add",
            "2024-01-01 00:00:00 UTC",
            "86400",
            "--output-format",
            "mjd",
        ],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # Should be ~MJD for 2024-01-02
    assert val > 60310


def test_add_with_output_time_system():
    result = runner.invoke(
        app,
        [
            "time",
            "add",
            "2024-01-01 00:00:00 UTC",
            "0",
            "--output-time-system",
            "TAI",
        ],
    )
    assert result.exit_code == 0
    assert "2024" in result.stdout


# =============================================================================
# time-system-offset command
# =============================================================================


def test_time_system_offset_utc_to_tai():
    result = runner.invoke(
        app,
        ["time", "time-system-offset", "2024-01-01 00:00:00 UTC", "UTC", "TAI"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # UTC-TAI offset is ~37 seconds in 2024
    assert 36 < val < 38


def test_time_system_offset_utc_to_gps():
    result = runner.invoke(
        app,
        ["time", "time-system-offset", "2024-01-01 00:00:00 UTC", "UTC", "GPS"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # GPS is ahead of UTC by ~18 seconds
    assert 17 < val < 19


# =============================================================================
# range command
# =============================================================================


def test_range_basic():
    result = runner.invoke(
        app,
        [
            "time",
            "range",
            "2024-01-01 00:00:00 UTC",
            "2024-01-01 00:05:00 UTC",
            "60",
        ],
    )
    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
    # 5 minutes with 60s step = 6 epochs (inclusive of start, exclusive or inclusive of end)
    assert len(lines) >= 5


def test_range_small_step():
    result = runner.invoke(
        app,
        [
            "time",
            "range",
            "2024-01-01 00:00:00 UTC",
            "2024-01-01 00:00:10 UTC",
            "5",
        ],
    )
    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
    assert len(lines) >= 2
    # Each line should contain epoch-like text
    assert "2024" in lines[0]
