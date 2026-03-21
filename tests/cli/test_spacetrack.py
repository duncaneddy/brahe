"""Tests for SpaceTrack CLI commands (top-level brahe spacetrack)"""

from typer.testing import CliRunner
from brahe.cli.spacetrack import app

runner = CliRunner()


def test_help():
    """Test top-level spacetrack help lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gp" in result.stdout
    assert "gp-history" in result.stdout
    assert "satcat" in result.stdout


def test_gp_help():
    """Test gp command help shows all options."""
    result = runner.invoke(app, ["gp", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--name" in result.stdout
    assert "--epoch-range" in result.stdout
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--order-by" in result.stdout
    assert "--descending" in result.stdout
    assert "--columns" in result.stdout
    assert "--output-format" in result.stdout
    assert "--output-file" in result.stdout


def test_gp_requires_selector():
    """Test gp requires at least one selector (and fails without credentials)."""
    result = runner.invoke(app, ["gp"])
    # Should fail - either missing selector or missing credentials
    assert result.exit_code == 1


def test_gp_history_help():
    """Test gp-history command help."""
    result = runner.invoke(app, ["gp-history", "--help"])
    assert result.exit_code == 0
    assert "catnr" in result.stdout.lower()
    assert "--epoch-range" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_help():
    """Test satcat command help shows all options."""
    result = runner.invoke(app, ["satcat", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--name" in result.stdout
    assert "--country" in result.stdout
    assert "--object-type" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_requires_selector():
    """Test satcat requires at least one selector."""
    result = runner.invoke(app, ["satcat"])
    assert result.exit_code == 1
