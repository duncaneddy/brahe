from typer.testing import CliRunner
from brahe.cli.__main__ import app

# Monkey patch the app to disable rich colors for testing
app.rich_markup_mode = None

# Disable color output to avoid ANSI escape codes in output
runner = CliRunner(mix_stderr=False)


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: brahe [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_eop_help():
    result = runner.invoke(app, ["eop", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe eop [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_time_help():
    result = runner.invoke(app, ["time", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe time [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_orbits_help():
    result = runner.invoke(app, ["orbits", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe orbits [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_convert_help():
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe convert [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_space_track_help():
    result = runner.invoke(app, ["space-track", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe space-track [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_access_help():
    result = runner.invoke(app, ["access", "--help"])
    assert result.exit_code == 0
    assert "Usage: brahe access [OPTIONS] COMMAND [ARGS]..." in result.stdout
