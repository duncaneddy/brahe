"""Tests for CelesTrak CLI commands (top-level brahe celestrak)"""

import pytest
from typer.testing import CliRunner
from brahe.cli.celestrak import app

runner = CliRunner()


def test_help():
    """Test top-level celestrak help lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gp" in result.stdout
    assert "sup-gp" in result.stdout
    assert "satcat" in result.stdout
    assert "groups" in result.stdout
    assert "download" in result.stdout


def test_gp_help():
    """Test gp command help shows all options."""
    result = runner.invoke(app, ["gp", "--help"])
    assert result.exit_code == 0
    assert "--group" in result.stdout
    assert "--name" in result.stdout
    assert "--catnr" in result.stdout
    assert "--intdes" in result.stdout
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--order-by" in result.stdout
    assert "--descending" in result.stdout
    assert "--columns" in result.stdout
    assert "--output-format" in result.stdout
    assert "--output-file" in result.stdout


def test_gp_requires_selector():
    """Test gp command requires at least one selector."""
    result = runner.invoke(app, ["gp"])
    assert result.exit_code == 1
    assert (
        "at least one" in result.stdout.lower() or "required" in result.stdout.lower()
    )


def test_sup_gp_help():
    """Test sup-gp command help."""
    result = runner.invoke(app, ["sup-gp", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_sup_gp_invalid_source():
    """Test sup-gp rejects invalid source names."""
    result = runner.invoke(app, ["sup-gp", "nonexistent_source"])
    assert result.exit_code == 1
    assert "unknown source" in result.stdout.lower() or "error" in result.stdout.lower()


def test_satcat_help():
    """Test satcat command help."""
    result = runner.invoke(app, ["satcat", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--active" in result.stdout
    assert "--payloads" in result.stdout
    assert "--on-orbit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_requires_filter():
    """Test satcat command requires at least one filter option."""
    result = runner.invoke(app, ["satcat"])
    assert result.exit_code == 1


def test_groups():
    """Test groups command lists satellite groups (no network)."""
    result = runner.invoke(app, ["groups"])
    assert result.exit_code == 0
    assert "active" in result.stdout
    assert "stations" in result.stdout
    assert "starlink" in result.stdout
    assert "gnss" in result.stdout
    assert "gps-ops" in result.stdout


def test_download_help():
    """Test download command help."""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "filepath" in result.stdout.lower()
    assert "--group" in result.stdout
    assert "--content-format" in result.stdout
    assert "--file-format" in result.stdout


# Integration tests (require network)


@pytest.mark.ci
def test_gp_by_catnr():
    """Test querying GP by NORAD catalog number."""
    result = runner.invoke(app, ["gp", "--catnr", "25544", "--output-format", "json"])
    assert result.exit_code == 0
    assert "ISS" in result.stdout or "ZARYA" in result.stdout


@pytest.mark.ci
def test_gp_by_group_with_limit():
    """Test querying GP by group with limit."""
    result = runner.invoke(
        app, ["gp", "--group", "stations", "--limit", "3", "--output-format", "json"]
    )
    assert result.exit_code == 0
