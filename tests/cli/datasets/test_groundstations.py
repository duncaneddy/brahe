"""Tests for groundstation CLI commands"""

from typer.testing import CliRunner
from brahe.cli.datasets.groundstations import app

runner = CliRunner(mix_stderr=False)


def test_help():
    """Test help command"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list-providers" in result.stdout
    assert "list-stations" in result.stdout
    assert "show" in result.stdout
    assert "show-all" in result.stdout


def test_list_providers():
    """Test list-providers command"""
    result = runner.invoke(app, ["list-providers"])
    assert result.exit_code == 0
    assert "ksat" in result.stdout.lower()
    assert "atlas" in result.stdout.lower()
    assert "aws" in result.stdout.lower()


def test_list_providers_table():
    """Test list-providers command with table output"""
    result = runner.invoke(app, ["list-providers", "-t"])
    assert result.exit_code == 0
    assert "Provider" in result.stdout
    assert "Stations" in result.stdout
    assert "Frequency Bands" in result.stdout
    assert "ksat" in result.stdout.lower() or "KSAT" in result.stdout


def test_show():
    """Test show command with ksat provider"""
    result = runner.invoke(app, ["show", "ksat"])
    assert result.exit_code == 0
    assert "KSAT" in result.stdout
    assert "Groundstations" in result.stdout


def test_show_with_properties():
    """Test show command with properties flag"""
    result = runner.invoke(app, ["show", "ksat", "--properties"])
    assert result.exit_code == 0
    assert "KSAT" in result.stdout
    assert "Groundstations" in result.stdout


def test_show_invalid_provider():
    """Test show command with invalid provider"""
    result = runner.invoke(app, ["show", "nonexistent"])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Error" in result.stderr


def test_show_all():
    """Test show-all command"""
    result = runner.invoke(app, ["show-all"])
    assert result.exit_code == 0
    assert "All Groundstations" in result.stdout


def test_show_all_with_properties():
    """Test show-all command with properties flag"""
    result = runner.invoke(app, ["show-all", "--properties"])
    assert result.exit_code == 0
    assert "All Groundstations" in result.stdout


def test_show_help():
    """Test show command help"""
    result = runner.invoke(app, ["show", "--help"])
    assert result.exit_code == 0
    assert "provider" in result.stdout.lower()
    assert "properties" in result.stdout.lower()


def test_show_all_help():
    """Test show-all command help"""
    result = runner.invoke(app, ["show-all", "--help"])
    assert result.exit_code == 0
    assert "properties" in result.stdout.lower()


def test_list_stations():
    """Test list-stations command"""
    result = runner.invoke(app, ["list-stations"])
    assert result.exit_code == 0
    assert "Groundstations" in result.stdout


def test_list_stations_with_provider():
    """Test list-stations command with provider filter"""
    result = runner.invoke(app, ["list-stations", "--provider", "ksat"])
    assert result.exit_code == 0
    assert "KSAT" in result.stdout or "ksat" in result.stdout.lower()


def test_list_stations_table():
    """Test list-stations command with table output"""
    result = runner.invoke(app, ["list-stations", "-t"])
    assert result.exit_code == 0
    assert "Name" in result.stdout
    assert "Lat" in result.stdout
    assert "Lon" in result.stdout
    assert "Alt" in result.stdout
    # Header may be abbreviated due to column width
    assert "Prov" in result.stdout or "Provider" in result.stdout
    assert "Freque" in result.stdout or "Frequency Bands" in result.stdout


def test_list_stations_table_with_provider():
    """Test list-stations command with table output and provider filter"""
    result = runner.invoke(app, ["list-stations", "--provider", "ksat", "-t"])
    assert result.exit_code == 0
    assert "KSAT" in result.stdout
    assert "Name" in result.stdout


def test_list_stations_invalid_provider():
    """Test list-stations command with invalid provider"""
    result = runner.invoke(app, ["list-stations", "--provider", "nonexistent"])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Error" in result.stderr


def test_list_providers_help():
    """Test list-providers command help"""
    result = runner.invoke(app, ["list-providers", "--help"])
    assert result.exit_code == 0
    assert "--table" in result.stdout or "-t" in result.stdout


def test_list_stations_help():
    """Test list-stations command help"""
    result = runner.invoke(app, ["list-stations", "--help"])
    assert result.exit_code == 0
    assert "provider" in result.stdout.lower()
    assert "table" in result.stdout.lower()
