"""Tests for brahe CLI orbits commands."""

from typer.testing import CliRunner
from brahe.cli.__main__ import app

app.rich_markup_mode = None
runner = CliRunner()


# =============================================================================
# orbital-period command
# =============================================================================


def test_orbital_period_basic():
    result = runner.invoke(app, ["orbits", "orbital-period", "6878137"])
    assert result.exit_code == 0
    # Should output a float value (period in seconds for ~500km LEO orbit)
    val = float(result.stdout.strip())
    assert 5000 < val < 6000


def test_orbital_period_with_expression():
    result = runner.invoke(app, ["orbits", "orbital-period", "R_EARTH+500e3"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 5000 < val < 6000


def test_orbital_period_units_minutes():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "R_EARTH+500e3", "--units", "minutes"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 80 < val < 100


def test_orbital_period_units_hours():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "R_EARTH+500e3", "--units", "hours"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 1.0 < val < 2.0


def test_orbital_period_units_days():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "R_EARTH+500e3", "--units", "days"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 0.0 < val < 0.1


def test_orbital_period_units_years():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "R_EARTH+500e3", "--units", "years"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_orbital_period_with_gm():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "6878137", "--gm", "GM_EARTH"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 5000 < val < 6000


def test_orbital_period_format():
    result = runner.invoke(
        app, ["orbits", "orbital-period", "R_EARTH+500e3", "--format", ".3f"]
    )
    assert result.exit_code == 0
    # Should have exactly 3 decimal places
    assert "." in result.stdout.strip()
    decimals = result.stdout.strip().split(".")[1]
    assert len(decimals) == 3


def test_orbital_period_invalid_expression():
    result = runner.invoke(app, ["orbits", "orbital-period", "invalid!!expr"])
    assert result.exit_code == 1


# =============================================================================
# sma-from-period command
# =============================================================================


def test_sma_from_period_basic():
    result = runner.invoke(app, ["orbits", "sma-from-period", "5400"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 6000000 < val < 7500000


def test_sma_from_period_units_minutes():
    result = runner.invoke(
        app, ["orbits", "sma-from-period", "90", "--units", "minutes"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 6000000 < val < 7500000


def test_sma_from_period_units_hours():
    result = runner.invoke(
        app, ["orbits", "sma-from-period", "1.5", "--units", "hours"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 6000000 < val < 7500000


def test_sma_from_period_with_gm():
    result = runner.invoke(
        app, ["orbits", "sma-from-period", "5400", "--gm", "GM_EARTH"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 6000000 < val < 7500000


def test_sma_from_period_invalid_expression():
    result = runner.invoke(app, ["orbits", "sma-from-period", "__import__('os')"])
    assert result.exit_code == 1


# =============================================================================
# mean-motion command
# =============================================================================


def test_mean_motion_basic():
    result = runner.invoke(app, ["orbits", "mean-motion", "R_EARTH+500e3"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # Mean motion in rad/s for LEO
    assert 0.0 < val < 0.01


def test_mean_motion_with_gm():
    result = runner.invoke(
        app, ["orbits", "mean-motion", "R_EARTH+500e3", "--gm", "GM_EARTH"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert 0.0 < val < 0.01


def test_mean_motion_invalid_expression():
    result = runner.invoke(app, ["orbits", "mean-motion", "bad@expr"])
    assert result.exit_code == 1


# =============================================================================
# anomaly-conversion command
# =============================================================================


def test_anomaly_conversion_mean_to_eccentric():
    result = runner.invoke(
        app,
        ["orbits", "anomaly-conversion", "1.0", "0.01", "mean", "eccentric"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_mean_to_true():
    result = runner.invoke(
        app, ["orbits", "anomaly-conversion", "1.0", "0.01", "mean", "true"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_eccentric_to_mean():
    result = runner.invoke(
        app,
        ["orbits", "anomaly-conversion", "1.0", "0.01", "eccentric", "mean"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_eccentric_to_true():
    result = runner.invoke(
        app,
        ["orbits", "anomaly-conversion", "1.0", "0.01", "eccentric", "true"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_true_to_mean():
    result = runner.invoke(
        app, ["orbits", "anomaly-conversion", "1.0", "0.01", "true", "mean"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_true_to_eccentric():
    result = runner.invoke(
        app,
        ["orbits", "anomaly-conversion", "1.0", "0.01", "true", "eccentric"],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_same_anomaly():
    result = runner.invoke(
        app, ["orbits", "anomaly-conversion", "1.5", "0.01", "mean", "mean"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert abs(val - 1.5) < 1e-10


def test_anomaly_conversion_as_degrees():
    result = runner.invoke(
        app,
        [
            "orbits",
            "anomaly-conversion",
            "45.0",
            "0.01",
            "mean",
            "eccentric",
            "--as-degrees",
        ],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    assert val > 0.0


def test_anomaly_conversion_invalid_expression():
    result = runner.invoke(
        app, ["orbits", "anomaly-conversion", "bad!", "0.01", "mean", "true"]
    )
    assert result.exit_code == 1


# =============================================================================
# sun-sync-inclination command
# =============================================================================


def test_sun_sync_inclination_basic():
    result = runner.invoke(
        app, ["orbits", "sun-sync-inclination", "R_EARTH+500e3", "0.01"]
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # Sun-sync inclination for 500km LEO is ~97 degrees (default is --as-degrees)
    assert 90 < val < 110


def test_sun_sync_inclination_no_degrees():
    result = runner.invoke(
        app,
        [
            "orbits",
            "sun-sync-inclination",
            "R_EARTH+500e3",
            "0.01",
            "--no-as-degrees",
        ],
    )
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # In radians, should be ~1.7
    assert 1.5 < val < 2.0


def test_sun_sync_inclination_invalid_expression():
    result = runner.invoke(app, ["orbits", "sun-sync-inclination", "bad!", "0.01"])
    assert result.exit_code == 1


# =============================================================================
# perigee-velocity command
# =============================================================================


def test_perigee_velocity_basic():
    result = runner.invoke(app, ["orbits", "perigee-velocity", "R_EARTH+500e3", "0.01"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # Perigee velocity for LEO ~7600 m/s
    assert 7000 < val < 8500


def test_perigee_velocity_invalid_expression():
    result = runner.invoke(app, ["orbits", "perigee-velocity", "bad!", "0.01"])
    assert result.exit_code == 1


# =============================================================================
# apogee-velocity command
# =============================================================================


def test_apogee_velocity_basic():
    result = runner.invoke(app, ["orbits", "apogee-velocity", "R_EARTH+500e3", "0.01"])
    assert result.exit_code == 0
    val = float(result.stdout.strip())
    # Apogee velocity for near-circular LEO ~7500 m/s
    assert 7000 < val < 8500


def test_apogee_velocity_invalid_expression():
    result = runner.invoke(app, ["orbits", "apogee-velocity", "bad!", "0.01"])
    assert result.exit_code == 1
