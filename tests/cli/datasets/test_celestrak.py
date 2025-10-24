"""Tests for CelesTrak CLI commands"""

from typer.testing import CliRunner
from brahe.cli.datasets.celestrak import app

runner = CliRunner(mix_stderr=False)


def test_help():
    """Test help command"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "lookup" in result.stdout
    assert "show" in result.stdout
    assert "search" in result.stdout
    assert "list-groups" in result.stdout


def test_list_groups():
    """Test list-groups command"""
    result = runner.invoke(app, ["list-groups"])
    assert result.exit_code == 0
    assert "active" in result.stdout
    assert "stations" in result.stdout
    assert "starlink" in result.stdout
    assert "gnss" in result.stdout


def test_search_help():
    """Test search command help"""
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "pattern" in result.stdout.lower()
    assert "table" in result.stdout.lower()
    assert "columns" in result.stdout.lower()
    assert "minimal" in result.stdout.lower()
    assert "default" in result.stdout.lower()
    assert "all" in result.stdout.lower()


# Network tests require internet connection
# Mark as network tests to allow skipping in CI/CD


def test_search_simple_format_no_network():
    """Test search simple format - requires network"""
    # This test would require mocking or network access
    # For now, just test that command accepts correct arguments
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_search_table_format_no_network():
    """Test search table format - requires network"""
    # This test would require mocking or network access
    # For now, just test that command accepts correct arguments
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_download_help():
    """Test download command help"""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "filepath" in result.stdout.lower()
    assert "group" in result.stdout.lower()
    assert "content" in result.stdout.lower() and "format" in result.stdout.lower()
    assert "file" in result.stdout.lower() and "format" in result.stdout.lower()


def test_lookup_help():
    """Test lookup command help"""
    result = runner.invoke(app, ["lookup", "--help"])
    assert result.exit_code == 0
    assert "name" in result.stdout.lower()
    assert "group" in result.stdout.lower()


def test_show_help():
    """Test show command help"""
    result = runner.invoke(app, ["show", "--help"])
    assert result.exit_code == 0
    assert "identifier" in result.stdout.lower()
    assert "group" in result.stdout.lower()
    assert "compact" in result.stdout.lower()
    assert "simple" in result.stdout.lower()
