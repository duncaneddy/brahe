"""Tests for the shared CLI output module."""

import json

import pytest
from click.exceptions import Exit
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from brahe.cli._output import (
    CLIOutputFormat,
    GP_COLUMNS,
    GP_COLUMN_PRESETS,
    SATCAT_COLUMNS,
    SATCAT_COLUMN_PRESETS,
    _build_markdown_table,
    _build_rich_table,
    format_gp_records,
    format_satcat_records,
    parse_columns,
    parse_filters,
    compute_gp_row,
)


class TestParseColumns:
    """Tests for parse_columns()."""

    def test_none_returns_default(self):
        result = parse_columns(None, GP_COLUMNS, GP_COLUMN_PRESETS)
        assert result == GP_COLUMN_PRESETS["default"]

    def test_preset_minimal(self):
        result = parse_columns("minimal", GP_COLUMNS, GP_COLUMN_PRESETS)
        assert result == ["name", "norad_id"]

    def test_preset_all(self):
        result = parse_columns("all", GP_COLUMNS, GP_COLUMN_PRESETS)
        assert result == list(GP_COLUMNS.keys())

    def test_preset_case_insensitive(self):
        result = parse_columns("MINIMAL", GP_COLUMNS, GP_COLUMN_PRESETS)
        assert result == ["name", "norad_id"]

    def test_custom_comma_separated(self):
        result = parse_columns("name,norad_id,inc", GP_COLUMNS, GP_COLUMN_PRESETS)
        assert result == ["name", "norad_id", "inc"]

    def test_invalid_column_exits(self):
        with pytest.raises(Exit):
            parse_columns("name,invalid_col", GP_COLUMNS, GP_COLUMN_PRESETS)

    def test_satcat_presets(self):
        result = parse_columns("minimal", SATCAT_COLUMNS, SATCAT_COLUMN_PRESETS)
        assert result == ["satname", "norad_cat_id"]


class TestParseFilters:
    """Tests for parse_filters()."""

    def test_none_returns_empty(self):
        assert parse_filters(None) == []

    def test_empty_list_returns_empty(self):
        assert parse_filters([]) == []

    def test_single_filter(self):
        result = parse_filters(["INCLINATION >50"])
        assert result == [("INCLINATION", ">50")]

    def test_multiple_filters(self):
        result = parse_filters(["INCLINATION >50", "ECCENTRICITY <0.01"])
        assert result == [("INCLINATION", ">50"), ("ECCENTRICITY", "<0.01")]

    def test_range_filter(self):
        result = parse_filters(["INCLINATION 40--60"])
        assert result == [("INCLINATION", "40--60")]

    def test_like_filter(self):
        result = parse_filters(["OBJECT_NAME ~~ISS"])
        assert result == [("OBJECT_NAME", "~~ISS")]

    def test_invalid_format_exits(self):
        with pytest.raises(Exit):
            parse_filters(["NO_SPACE"])


class TestComputeGpRow:
    """Tests for compute_gp_row()."""

    def test_basic_computation(self):
        """Test row computation with a mock GPRecord."""
        record = MagicMock()
        record.object_name = "ISS (ZARYA)"
        record.norad_cat_id = 25544
        record.mean_motion = 15.5  # rev/day
        record.eccentricity = 0.0001
        record.inclination = 51.64
        record.ra_of_asc_node = 200.0
        record.arg_of_pericenter = 100.0
        record.mean_anomaly = 260.0
        record.epoch = "2024-01-15T12:00:00.000"
        record.country_code = "US"
        record.object_type = "PAYLOAD"

        epoch_mock = MagicMock()
        epoch_mock.__sub__ = MagicMock(return_value=3600.0)

        row = compute_gp_row(record, epoch_mock)

        assert row["name"] == "ISS (ZARYA)"
        assert row["norad_id"] == 25544
        assert row["inc"] == 51.64
        assert row["ecc"] == 0.0001
        assert row["sma"] > 0
        assert row["period"] > 0
        assert row["country"] == "US"

    def test_missing_fields(self):
        """Test row computation with missing optional fields."""
        record = MagicMock()
        record.object_name = None
        record.norad_cat_id = None
        record.mean_motion = None
        record.eccentricity = None
        record.inclination = None
        record.ra_of_asc_node = None
        record.arg_of_pericenter = None
        record.mean_anomaly = None
        record.epoch = None
        record.country_code = None
        record.object_type = None

        epoch_mock = MagicMock()

        row = compute_gp_row(record, epoch_mock)

        assert row["name"] == "UNKNOWN"
        assert row["norad_id"] == "?"
        assert row["sma"] == 0.0
        assert row["period"] == 0.0


class TestCLIOutputFormat:
    """Tests for CLIOutputFormat enum."""

    def test_values(self):
        assert CLIOutputFormat.rich == "rich"
        assert CLIOutputFormat.markdown == "markdown"
        assert CLIOutputFormat.json == "json"
        assert CLIOutputFormat.omm == "omm"


# ---------------------------------------------------------------------------
# Helper factory for mock GP records
# ---------------------------------------------------------------------------


def _make_gp_record(**overrides):
    """Create a MagicMock GP record with realistic defaults.

    Any keyword argument overrides the corresponding attribute on the mock.
    The mock's ``to_dict()`` returns a dict of the same attribute values.
    """
    defaults = {
        "object_name": "ISS (ZARYA)",
        "norad_cat_id": 25544,
        "mean_motion": 15.5,
        "eccentricity": 0.0001,
        "inclination": 51.64,
        "ra_of_asc_node": 200.0,
        "arg_of_pericenter": 100.0,
        "mean_anomaly": 260.0,
        "epoch": "2024-01-15T12:00:00.000",
        "country_code": "US",
        "object_type": "PAYLOAD",
    }
    defaults.update(overrides)

    rec = MagicMock()
    for attr, value in defaults.items():
        setattr(rec, attr, value)

    rec.to_dict.return_value = defaults
    return rec


# ---------------------------------------------------------------------------
# _build_markdown_table tests
# ---------------------------------------------------------------------------


class TestBuildMarkdownTable:
    """Tests for _build_markdown_table()."""

    def test_two_rows_structure(self):
        """Markdown table should have header, separator, and data lines."""
        rows = [
            {"name": "ISS", "norad_id": 25544},
            {"name": "NOAA 15", "norad_id": 25338},
        ]
        column_list = ["name", "norad_id"]
        result = _build_markdown_table(rows, column_list, GP_COLUMNS)

        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows

        # Header line contains column titles
        assert "Name" in lines[0]
        assert "ID" in lines[0]

        # Separator line is dashes
        assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}

        # Data lines contain values
        assert "ISS" in lines[2]
        assert "25544" in lines[2]
        assert "NOAA 15" in lines[3]
        assert "25338" in lines[3]

    def test_pipe_delimiters(self):
        """Each line should start and end with pipe characters."""
        rows = [{"name": "SAT", "norad_id": 1}]
        result = _build_markdown_table(rows, ["name", "norad_id"], GP_COLUMNS)
        for line in result.split("\n"):
            assert line.startswith("|")
            assert line.endswith("|")


# ---------------------------------------------------------------------------
# _build_rich_table tests
# ---------------------------------------------------------------------------


class TestBuildRichTable:
    """Tests for _build_rich_table()."""

    def test_row_and_column_count(self):
        """Rich table should contain the right number of rows and columns."""
        rows = [
            {"name": "ISS", "norad_id": 25544},
            {"name": "NOAA 15", "norad_id": 25338},
        ]
        column_list = ["name", "norad_id"]
        table = _build_rich_table(rows, column_list, GP_COLUMNS)

        assert table.row_count == 2
        assert len(table.columns) == 2

    def test_single_column(self):
        """Rich table with a single column."""
        rows = [{"name": "ISS"}]
        table = _build_rich_table(rows, ["name"], GP_COLUMNS)
        assert table.row_count == 1
        assert len(table.columns) == 1


# ---------------------------------------------------------------------------
# format_gp_records tests
# ---------------------------------------------------------------------------


class TestFormatGpRecords:
    """Tests for format_gp_records()."""

    def _capture_console(self):
        """Return a Console that writes to a StringIO buffer."""
        buf = StringIO()
        return Console(file=buf, width=200, no_color=True, highlight=False), buf

    # -- empty records --

    def test_empty_records(self):
        console, buf = self._capture_console()
        format_gp_records([], CLIOutputFormat.rich, None, None, console)
        assert "No records found" in buf.getvalue()

    # -- json stdout --

    def test_json_stdout(self, capsys):
        rec = _make_gp_record()
        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.json, None, None, console)
        rec.to_dict.assert_called_once()
        # typer.echo writes to real stdout
        captured = capsys.readouterr()
        assert "ISS (ZARYA)" in captured.out

    # -- json file --

    def test_json_file_output(self, tmp_path):
        rec = _make_gp_record()
        out = tmp_path / "out.json"
        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.json, None, out, console)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["object_name"] == "ISS (ZARYA)"
        assert "Wrote 1 record(s)" in buf.getvalue()

    # -- omm stdout --

    @patch("brahe.CCSDSJsonKeyCase", create=True)
    def test_omm_stdout(self, mock_key_case, capsys):
        rec = _make_gp_record()
        omm_mock = MagicMock()
        omm_mock.to_json_string.return_value = '{"OBJECT_NAME": "ISS"}'
        rec.to_omm.return_value = omm_mock

        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.omm, None, None, console)

        rec.to_omm.assert_called_once()
        captured = capsys.readouterr()
        assert "ISS" in captured.out

    # -- omm conversion error --

    def test_omm_conversion_error(self, capsys):
        rec = _make_gp_record()
        rec.to_omm.side_effect = RuntimeError("conversion failed")

        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.omm, None, None, console)

        assert "Warning" in buf.getvalue()
        # Should still produce valid JSON (empty list)
        captured = capsys.readouterr()
        assert json.loads(captured.out) == []

    # -- omm file --

    @patch("brahe.CCSDSJsonKeyCase", create=True)
    def test_omm_file_output(self, mock_key_case, tmp_path):
        rec = _make_gp_record()
        omm_mock = MagicMock()
        omm_mock.to_json_string.return_value = '{"OBJECT_NAME": "ISS"}'
        rec.to_omm.return_value = omm_mock

        out = tmp_path / "out.omm.json"
        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.omm, None, out, console)

        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["OBJECT_NAME"] == "ISS"
        assert "Wrote 1 OMM record(s)" in buf.getvalue()

    # -- markdown stdout --

    @patch("brahe.cli._output._current_epoch")
    def test_markdown_stdout(self, mock_epoch, capsys):
        mock_epoch.return_value = MagicMock(__sub__=MagicMock(return_value=3600.0))
        rec = _make_gp_record()
        console, buf = self._capture_console()
        format_gp_records(
            [rec], CLIOutputFormat.markdown, "name,norad_id", None, console
        )
        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "ISS (ZARYA)" in captured.out
        assert "1 record(s)" in buf.getvalue()

    # -- markdown file --

    @patch("brahe.cli._output._current_epoch")
    def test_markdown_file_output(self, mock_epoch, tmp_path):
        mock_epoch.return_value = MagicMock(__sub__=MagicMock(return_value=3600.0))
        rec = _make_gp_record()
        out = tmp_path / "out.md"
        console, buf = self._capture_console()
        format_gp_records(
            [rec], CLIOutputFormat.markdown, "name,norad_id", out, console
        )
        content = out.read_text()
        assert "Name" in content
        assert "ISS (ZARYA)" in content
        assert "Wrote 1 record(s)" in buf.getvalue()

    # -- rich stdout --

    @patch("brahe.cli._output._current_epoch")
    def test_rich_stdout(self, mock_epoch):
        mock_epoch.return_value = MagicMock(__sub__=MagicMock(return_value=3600.0))
        rec = _make_gp_record()
        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.rich, "name,norad_id", None, console)
        output = buf.getvalue()
        assert "1 record(s)" in output

    # -- rich file --

    @patch("brahe.cli._output._current_epoch")
    def test_rich_file_output(self, mock_epoch, tmp_path):
        mock_epoch.return_value = MagicMock(__sub__=MagicMock(return_value=3600.0))
        rec = _make_gp_record()
        out = tmp_path / "out.txt"
        console, buf = self._capture_console()
        format_gp_records([rec], CLIOutputFormat.rich, "name,norad_id", out, console)
        assert out.exists()
        content = out.read_text()
        assert "ISS (ZARYA)" in content
        assert "Wrote 1 record(s)" in buf.getvalue()


# ---------------------------------------------------------------------------
# format_satcat_records tests
# ---------------------------------------------------------------------------


def _make_satcat_record(**overrides):
    """Create a MagicMock SATCAT record with realistic defaults."""
    defaults = {
        "satname": "ISS (ZARYA)",
        "norad_cat_id": 25544,
        "intldes": "1998-067A",
        "object_type": "PAYLOAD",
        "country": "US",
        "launch": "1998-11-20",
        "decay": "",
        "period": "92.9",
        "inclination": "51.6",
        "apogee": "419",
        "perigee": "417",
        "rcs_size": "LARGE",
    }
    defaults.update(overrides)

    rec = MagicMock()
    for attr, value in defaults.items():
        setattr(rec, attr, value)
    return rec


class TestFormatSatcatRecords:
    """Tests for format_satcat_records()."""

    def _capture_console(self):
        buf = StringIO()
        return Console(file=buf, width=200, no_color=True, highlight=False), buf

    # -- empty records --

    def test_empty_records(self):
        console, buf = self._capture_console()
        format_satcat_records([], CLIOutputFormat.rich, None, None, console)
        assert "No records found" in buf.getvalue()

    # -- json stdout --

    def test_json_stdout(self, capsys):
        rec = _make_satcat_record()
        console, buf = self._capture_console()
        format_satcat_records([rec], CLIOutputFormat.json, None, None, console)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert data[0]["satname"] == "ISS (ZARYA)"

    # -- json file --

    def test_json_file_output(self, tmp_path):
        rec = _make_satcat_record()
        out = tmp_path / "satcat.json"
        console, buf = self._capture_console()
        format_satcat_records([rec], CLIOutputFormat.json, None, out, console)
        data = json.loads(out.read_text())
        assert data[0]["satname"] == "ISS (ZARYA)"
        assert "Wrote 1 record(s)" in buf.getvalue()

    # -- omm treated as json --

    def test_omm_treated_as_json(self, capsys):
        rec = _make_satcat_record()
        console, buf = self._capture_console()
        format_satcat_records([rec], CLIOutputFormat.omm, None, None, console)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data[0]["satname"] == "ISS (ZARYA)"

    # -- markdown format --

    def test_markdown_format(self, capsys):
        rec = _make_satcat_record()
        console, buf = self._capture_console()
        format_satcat_records(
            [rec], CLIOutputFormat.markdown, "satname,norad_cat_id", None, console
        )
        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "ISS (ZARYA)" in captured.out
        assert "1 record(s)" in buf.getvalue()

    # -- markdown file --

    def test_markdown_file_output(self, tmp_path):
        rec = _make_satcat_record()
        out = tmp_path / "satcat.md"
        console, buf = self._capture_console()
        format_satcat_records(
            [rec], CLIOutputFormat.markdown, "satname,norad_cat_id", out, console
        )
        content = out.read_text()
        assert "Name" in content
        assert "ISS (ZARYA)" in content
        assert "Wrote 1 record(s)" in buf.getvalue()

    # -- rich format --

    def test_rich_format(self):
        rec = _make_satcat_record()
        console, buf = self._capture_console()
        format_satcat_records(
            [rec], CLIOutputFormat.rich, "satname,norad_cat_id", None, console
        )
        output = buf.getvalue()
        assert "1 record(s)" in output

    # -- rich file --

    def test_rich_file_output(self, tmp_path):
        rec = _make_satcat_record()
        out = tmp_path / "satcat.txt"
        console, buf = self._capture_console()
        format_satcat_records(
            [rec], CLIOutputFormat.rich, "satname,norad_cat_id", out, console
        )
        assert out.exists()
        content = out.read_text()
        assert "ISS (ZARYA)" in content
        assert "Wrote 1 record(s)" in buf.getvalue()

    # -- custom columns_def --

    def test_custom_columns_def(self, capsys):
        """Passing a custom columns_def uses those definitions."""
        custom_cols = {
            "satname": ("Satellite", 20, str),
            "norad_cat_id": ("NORAD", 8, lambda x: str(x) if x else "?"),
        }
        custom_presets = {
            "minimal": ["satname"],
            "default": ["satname", "norad_cat_id"],
            "all": ["satname", "norad_cat_id"],
        }
        rec = _make_satcat_record()
        console, buf = self._capture_console()
        format_satcat_records(
            [rec],
            CLIOutputFormat.markdown,
            None,
            None,
            console,
            columns_def=custom_cols,
            presets=custom_presets,
        )
        captured = capsys.readouterr()
        # Custom header "Satellite" instead of default "Name"
        assert "Satellite" in captured.out
        assert "ISS (ZARYA)" in captured.out
