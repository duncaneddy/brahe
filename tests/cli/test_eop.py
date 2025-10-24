import tempfile
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from brahe.cli.__main__ import app
import brahe

# Monkey patch the app to disable rich colors for testing
app.rich_markup_mode = None

# Disable color output to avoid ANSI escape codes in output
runner = CliRunner(mix_stderr=False)


def test_cli_eop_download_standard():
    tmpfile = tempfile.NamedTemporaryFile().name
    with patch("brahe.download_standard_eop_file") as mock:
        result = runner.invoke(
            app, ["eop", "download", tmpfile, "--product", "standard"]
        )
        assert result.exit_code == 0
        assert f"Downloaded standard EOP data to {tmpfile}" in result.stdout
        # Normalize path to POSIX format to match CLI behavior (uses .as_posix() internally)
        mock.assert_called_once_with(Path(tmpfile).absolute().as_posix())


def test_cli_eop_download_c04():
    tmpfile = tempfile.NamedTemporaryFile().name
    with patch("brahe.download_c04_eop_file") as mock:
        result = runner.invoke(app, ["eop", "download", tmpfile, "--product", "c04"])
        assert result.exit_code == 0
        assert f"Downloaded c04 EOP data to {tmpfile}" in result.stdout
        # Normalize path to POSIX format to match CLI behavior (uses .as_posix() internally)
        mock.assert_called_once_with(Path(tmpfile).absolute().as_posix())


def test_cli_eop_get_utc_ut1(iau2000_standard_filepath, iau2000_c04_20_filepath):
    # Patch the eop initialization to use the test files
    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(app, ["eop", "get-utc-ut1", "2022-01-01T00:00:00Z"])
        assert result.exit_code == 0
        assert "-0.1104988" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-utc-ut1", "2022-01-01T00:00:00Z", "--product", "standard"]
        )
        assert result.exit_code == 0
        assert "-0.1104988" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-utc-ut1",
            "2022-01-01T00:00:00Z",
            "--product",
            "standard",
            "--source",
            "file",
            "--filepath",
            iau2000_standard_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "-0.1104988" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-utc-ut1", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "-0.1105073" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-utc-ut1", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "-0.1105073" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-utc-ut1",
            "2022-01-01T00:00:00Z",
            "--product",
            "c04",
            "--source",
            "file",
            "--filepath",
            iau2000_c04_20_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "-0.1105073" in result.stdout

    result = runner.invoke(app, ["eop", "get-utc-ut1", "3000-01-01T00:00:00Z"])
    assert result.exit_code == 1
    assert "is out of range for EOP data." in result.stdout


def test_cli_eop_get_polar_motion(iau2000_standard_filepath, iau2000_c04_20_filepath):
    # Patch the eop initialization to use the test files
    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(app, ["eop", "get-polar-motion", "2022-01-01T00:00:00Z"])
        assert result.exit_code == 0
        assert "2.6492158790549485e-07, 1.3428660227580594e-06" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app,
            [
                "eop",
                "get-polar-motion",
                "2022-01-01T00:00:00Z",
                "--product",
                "standard",
            ],
        )
        assert result.exit_code == 0
        assert "2.6492158790549485e-07, 1.3428660227580594e-06" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-polar-motion",
            "2022-01-01T00:00:00Z",
            "--product",
            "standard",
            "--source",
            "file",
            "--filepath",
            iau2000_standard_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "2.6492158790549485e-07, 1.3428660227580594e-06" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-polar-motion", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "2.649555248631725e-07, 1.342846630210815e-06" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-polar-motion", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "2.649555248631725e-07, 1.342846630210815e-06" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-polar-motion",
            "2022-01-01T00:00:00Z",
            "--product",
            "c04",
            "--source",
            "file",
            "--filepath",
            iau2000_c04_20_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "2.649555248631725e-07, 1.342846630210815e-06" in result.stdout

    result = runner.invoke(app, ["eop", "get-polar-motion", "3000-01-01T00:00:00Z"])
    assert result.exit_code == 1
    assert "is out of range for EOP data." in result.stdout


def test_cli_eop_get_cip_offset(iau2000_standard_filepath, iau2000_c04_20_filepath):
    # Patch the eop initialization to use the test files
    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(app, ["eop", "get-cip-offset", "2022-01-01T00:00:00Z"])
        assert result.exit_code == 0
        assert "4.6057299705405924e-10, -1.21203420277384e-09" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app,
            ["eop", "get-cip-offset", "2022-01-01T00:00:00Z", "--product", "standard"],
        )
        assert result.exit_code == 0
        assert "4.6057299705405924e-10, -1.21203420277384e-09" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-cip-offset",
            "2022-01-01T00:00:00Z",
            "--product",
            "standard",
            "--source",
            "file",
            "--filepath",
            iau2000_standard_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "4.6057299705405924e-10, -1.21203420277384e-09" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-cip-offset", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "3.1997702953229375e-10, -1.0714382352520745e-09" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-cip-offset", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "3.1997702953229375e-10, -1.0714382352520745e-09" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-cip-offset",
            "2022-01-01T00:00:00Z",
            "--product",
            "c04",
            "--source",
            "file",
            "--filepath",
            iau2000_c04_20_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "3.1997702953229375e-10, -1.0714382352520745e-09" in result.stdout

    result = runner.invoke(app, ["eop", "get-cip-offset", "3000-01-01T00:00:00Z"])
    assert result.exit_code == 1
    assert "is out of range for EOP data." in result.stdout


def test_cli_eop_get_lod(iau2000_standard_filepath, iau2000_c04_20_filepath):
    # Patch the eop initialization to use the test files
    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(app, ["eop", "get-lod", "2022-01-01T00:00:00Z"])
        assert result.exit_code == 0
        assert "-2.67e-05" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_standard_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-lod", "2022-01-01T00:00:00Z", "--product", "standard"]
        )
        assert result.exit_code == 0
        assert "-2.67e-05" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-lod",
            "2022-01-01T00:00:00Z",
            "--product",
            "standard",
            "--source",
            "file",
            "--filepath",
            iau2000_standard_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "-2.67e-05" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-lod", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "-3.14" in result.stdout and "e-05" in result.stdout

    with patch(
        "brahe.cli.eop.get_global_eop_source",
        return_value=brahe.FileEOPProvider.from_file(
            iau2000_c04_20_filepath, True, "Error"
        ),
    ):
        result = runner.invoke(
            app, ["eop", "get-lod", "2022-01-01T00:00:00Z", "--product", "c04"]
        )
        assert result.exit_code == 0
        assert "-3.14" in result.stdout and "e-05" in result.stdout

    result = runner.invoke(
        app,
        [
            "eop",
            "get-lod",
            "2022-01-01T00:00:00Z",
            "--product",
            "c04",
            "--source",
            "file",
            "--filepath",
            iau2000_c04_20_filepath,
        ],
    )
    assert result.exit_code == 0
    assert "-3.14" in result.stdout and "e-05" in result.stdout

    result = runner.invoke(app, ["eop", "get-lod", "3000-01-01T00:00:00Z"])
    assert result.exit_code == 1
    assert "is out of range for EOP data." in result.stdout
