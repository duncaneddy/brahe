"""Tests for groundstation CLI commands"""

from typer.testing import CliRunner
from brahe.cli.datasets.groundstations import app

runner = CliRunner(mix_stderr=False)


def test_help():
    """Test help command"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "show" in result.stdout
    assert "show-all" in result.stdout


def test_list():
    """Test list command"""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "ksat" in result.stdout.lower()
    assert "atlas" in result.stdout.lower()
    assert "aws" in result.stdout.lower()


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
    assert "--properties" in result.stdout


def test_show_all_help():
    """Test show-all command help"""
    result = runner.invoke(app, ["show-all", "--help"])
    assert result.exit_code == 0
    assert "--properties" in result.stdout
