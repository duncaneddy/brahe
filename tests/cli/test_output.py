"""Tests for the shared CLI output module."""

import pytest
from click.exceptions import Exit
from unittest.mock import MagicMock

from brahe.cli._output import (
    CLIOutputFormat,
    GP_COLUMNS,
    GP_COLUMN_PRESETS,
    SATCAT_COLUMNS,
    SATCAT_COLUMN_PRESETS,
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
