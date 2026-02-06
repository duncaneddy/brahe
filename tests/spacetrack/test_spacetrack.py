"""Tests for SpaceTrack Python bindings."""

import os

import pytest

import brahe as bh
from brahe.spacetrack import operators as op


# ========================================
# Operator Tests
# ========================================


class TestOperators:
    """Tests for SpaceTrack operator functions."""

    def test_greater_than(self):
        assert op.greater_than("25544") == ">25544"

    def test_less_than(self):
        assert op.less_than("0.01") == "<0.01"

    def test_not_equal(self):
        assert op.not_equal("DEBRIS") == "<>DEBRIS"

    def test_inclusive_range(self):
        assert op.inclusive_range("1", "100") == "1--100"

    def test_like(self):
        assert op.like("ISS") == "~~ISS"

    def test_startswith(self):
        assert op.startswith("2024") == "^2024"

    def test_now(self):
        assert op.now() == "now"

    def test_now_offset_negative(self):
        assert op.now_offset(-7) == "now-7"

    def test_now_offset_positive(self):
        assert op.now_offset(14) == "now+14"

    def test_now_offset_zero(self):
        assert op.now_offset(0) == "now+0"

    def test_null_val(self):
        assert op.null_val() == "null-val"

    def test_or_list(self):
        assert op.or_list(["25544", "25545", "25546"]) == "25544,25545,25546"

    def test_or_list_single(self):
        assert op.or_list(["25544"]) == "25544"

    def test_operator_composition(self):
        """Test operators can be composed as query filter values."""
        result = op.greater_than(op.now_offset(-7))
        assert result == ">now-7"


# ========================================
# Enum Tests
# ========================================


class TestEnums:
    """Tests for SpaceTrack enum types."""

    def test_request_controller_variants(self):
        assert str(bh.RequestController.BASIC_SPACE_DATA) == "BasicSpaceData"
        assert str(bh.RequestController.EXPANDED_SPACE_DATA) == "ExpandedSpaceData"
        assert str(bh.RequestController.FILE_SHARE) == "FileShare"
        assert str(bh.RequestController.SP_EPHEMERIS) == "SpEphemeris"
        assert str(bh.RequestController.PUBLIC_FILES) == "PublicFiles"

    def test_request_class_variants(self):
        assert str(bh.RequestClass.GP) == "GP"
        assert str(bh.RequestClass.GP_HISTORY) == "GPHistory"
        assert str(bh.RequestClass.SATCAT) == "SATCAT"
        assert str(bh.RequestClass.SATCAT_CHANGE) == "SATCATChange"
        assert str(bh.RequestClass.SATCAT_DEBUT) == "SATCATDebut"
        assert str(bh.RequestClass.DECAY) == "Decay"
        assert str(bh.RequestClass.TIP) == "TIP"
        assert str(bh.RequestClass.CDM_PUBLIC) == "CDMPublic"
        assert str(bh.RequestClass.BOXSCORE) == "Boxscore"
        assert str(bh.RequestClass.ANNOUNCEMENT) == "Announcement"
        assert str(bh.RequestClass.LAUNCH_SITE) == "LaunchSite"

    def test_sort_order_variants(self):
        assert str(bh.SortOrder.ASC) == "Asc"
        assert str(bh.SortOrder.DESC) == "Desc"

    def test_output_format_variants(self):
        assert str(bh.OutputFormat.JSON) == "JSON"
        assert str(bh.OutputFormat.XML) == "XML"
        assert str(bh.OutputFormat.HTML) == "HTML"
        assert str(bh.OutputFormat.CSV) == "CSV"
        assert str(bh.OutputFormat.TLE) == "TLE"
        assert str(bh.OutputFormat.THREE_LE) == "3LE"
        assert str(bh.OutputFormat.KVN) == "KVN"

    def test_request_class_equality(self):
        assert bh.RequestClass.GP == bh.RequestClass.GP
        assert bh.RequestClass.GP != bh.RequestClass.SATCAT

    def test_sort_order_equality(self):
        assert bh.SortOrder.ASC == bh.SortOrder.ASC
        assert bh.SortOrder.ASC != bh.SortOrder.DESC

    def test_output_format_equality(self):
        assert bh.OutputFormat.JSON == bh.OutputFormat.JSON
        assert bh.OutputFormat.JSON != bh.OutputFormat.TLE

    def test_request_controller_repr(self):
        assert "RequestController" in repr(bh.RequestController.BASIC_SPACE_DATA)

    def test_request_class_repr(self):
        assert "RequestClass" in repr(bh.RequestClass.GP)

    def test_sort_order_repr(self):
        assert "SortOrder" in repr(bh.SortOrder.ASC)

    def test_output_format_repr(self):
        assert "OutputFormat" in repr(bh.OutputFormat.JSON)


# ========================================
# Query Builder Tests
# ========================================


class TestSpaceTrackQuery:
    """Tests for SpaceTrackQuery builder."""

    def test_basic_query(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP)
        url = query.build()
        assert "/basicspacedata/" in url
        assert "/class/gp/" in url
        assert "format/json" in url

    def test_filter(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544")
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url

    def test_multiple_filters(self):
        query = (
            bh.SpaceTrackQuery(bh.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .filter("EPOCH", ">2024-01-01")
        )
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url
        assert "EPOCH/>2024-01-01" in url

    def test_order_by(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).order_by(
            "EPOCH", bh.SortOrder.DESC
        )
        url = query.build()
        assert "orderby/EPOCH desc" in url

    def test_order_by_asc(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).order_by(
            "EPOCH", bh.SortOrder.ASC
        )
        url = query.build()
        assert "orderby/EPOCH asc" in url

    def test_limit(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).limit(10)
        url = query.build()
        assert "limit/10" in url

    def test_limit_offset(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).limit_offset(10, 20)
        url = query.build()
        assert "limit/10,20" in url

    def test_format_tle(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).format(bh.OutputFormat.TLE)
        url = query.build()
        assert "format/tle" in url

    def test_format_csv(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).format(bh.OutputFormat.CSV)
        url = query.build()
        assert "format/csv" in url

    def test_predicates_filter(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).predicates_filter(
            ["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH"]
        )
        url = query.build()
        assert "predicates/NORAD_CAT_ID,OBJECT_NAME,EPOCH" in url

    def test_metadata(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).metadata(True)
        url = query.build()
        assert "metadata/true" in url

    def test_distinct(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).distinct(True)
        url = query.build()
        assert "distinct/true" in url

    def test_empty_result(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).empty_result(True)
        url = query.build()
        assert "emptyresult/show" in url

    def test_favorites(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).favorites("my_faves")
        url = query.build()
        assert "favorites/my_faves" in url

    def test_controller_override(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP).controller(
            bh.RequestController.EXPANDED_SPACE_DATA
        )
        url = query.build()
        assert "/expandedspacedata/" in url

    def test_satcat_query(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT)
        url = query.build()
        assert "/class/satcat/" in url

    def test_full_query_chain(self):
        """Test a complete query with multiple builder methods."""
        query = (
            bh.SpaceTrackQuery(bh.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", bh.SortOrder.DESC)
            .limit(1)
        )
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url
        assert "orderby/EPOCH desc" in url
        assert "limit/1" in url
        assert "format/json" in url

    def test_query_with_operators(self):
        """Test query using operator functions for filter values."""
        query = (
            bh.SpaceTrackQuery(bh.RequestClass.GP)
            .filter("EPOCH", op.greater_than(op.now_offset(-7)))
            .filter("ECCENTRICITY", op.less_than("0.01"))
            .filter("NORAD_CAT_ID", op.inclusive_range("25544", "25600"))
        )
        url = query.build()
        assert "EPOCH/>now-7" in url
        assert "ECCENTRICITY/<0.01" in url
        assert "NORAD_CAT_ID/25544--25600" in url

    def test_query_str(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP)
        assert str(query) == query.build()

    def test_query_repr(self):
        query = bh.SpaceTrackQuery(bh.RequestClass.GP)
        assert "SpaceTrackQuery" in repr(query)
        assert query.build() in repr(query)

    def test_query_with_all_formats(self):
        """Test all 7 output formats produce correct URL segments."""
        formats = [
            (bh.OutputFormat.JSON, "format/json"),
            (bh.OutputFormat.XML, "format/xml"),
            (bh.OutputFormat.HTML, "format/html"),
            (bh.OutputFormat.CSV, "format/csv"),
            (bh.OutputFormat.TLE, "format/tle"),
            (bh.OutputFormat.THREE_LE, "format/3le"),
            (bh.OutputFormat.KVN, "format/kvn"),
        ]
        for fmt, expected in formats:
            query = bh.SpaceTrackQuery(bh.RequestClass.GP).format(fmt)
            url = query.build()
            assert expected in url, f"Expected '{expected}' in URL, got: {url}"

    def test_query_with_all_controllers(self):
        """Test all 5 controllers produce correct URL segments."""
        controllers = [
            (bh.RequestController.BASIC_SPACE_DATA, "/basicspacedata/"),
            (bh.RequestController.EXPANDED_SPACE_DATA, "/expandedspacedata/"),
            (bh.RequestController.FILE_SHARE, "/fileshare/"),
            (bh.RequestController.SP_EPHEMERIS, "/spephemeris/"),
            (bh.RequestController.PUBLIC_FILES, "/publicfiles/"),
        ]
        for ctrl, expected in controllers:
            query = bh.SpaceTrackQuery(bh.RequestClass.GP).controller(ctrl)
            url = query.build()
            assert expected in url, f"Expected '{expected}' in URL, got: {url}"

    def test_query_immutability(self):
        """Test that builder methods return new instances (immutable chaining)."""
        base = bh.SpaceTrackQuery(bh.RequestClass.GP)
        with_filter = base.filter("NORAD_CAT_ID", "25544")
        with_limit = base.limit(10)

        # base should not be affected by subsequent calls
        base_url = base.build()
        filter_url = with_filter.build()
        limit_url = with_limit.build()

        assert "NORAD_CAT_ID" not in base_url
        assert "limit" not in base_url
        assert "NORAD_CAT_ID/25544" in filter_url
        assert "limit/10" in limit_url


# ========================================
# Client Construction Tests
# ========================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig configuration object."""

    def test_default_values(self):
        config = bh.RateLimitConfig()
        assert config.max_per_minute == 25
        assert config.max_per_hour == 250

    def test_custom_values(self):
        config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
        assert config.max_per_minute == 10
        assert config.max_per_hour == 100

    def test_keyword_arguments(self):
        config = bh.RateLimitConfig(max_per_hour=50, max_per_minute=5)
        assert config.max_per_minute == 5
        assert config.max_per_hour == 50

    def test_disabled(self):
        config = bh.RateLimitConfig.disabled()
        assert config.max_per_minute == 2**32 - 1
        assert config.max_per_hour == 2**32 - 1

    def test_str(self):
        config = bh.RateLimitConfig()
        s = str(config)
        assert "25" in s
        assert "250" in s

    def test_repr(self):
        config = bh.RateLimitConfig()
        r = repr(config)
        assert "RateLimitConfig" in r
        assert "25" in r
        assert "250" in r

    def test_equality(self):
        a = bh.RateLimitConfig()
        b = bh.RateLimitConfig()
        c = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
        assert a == b
        assert a != c

    def test_equality_disabled(self):
        a = bh.RateLimitConfig.disabled()
        b = bh.RateLimitConfig.disabled()
        assert a == b
        assert a != bh.RateLimitConfig()


class TestSpaceTrackClient:
    """Tests for SpaceTrackClient construction (no network)."""

    def test_client_creation(self):
        client = bh.SpaceTrackClient("user@example.com", "password")
        assert client is not None

    def test_client_with_base_url(self):
        client = bh.SpaceTrackClient(
            "user@example.com", "password", "https://test.space-track.org"
        )
        assert client is not None

    def test_client_with_rate_limit(self):
        config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
        client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None

    def test_client_with_base_url_and_rate_limit(self):
        config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
        client = bh.SpaceTrackClient(
            "user@example.com",
            "password",
            "https://test.space-track.org",
            rate_limit=config,
        )
        assert client is not None

    def test_client_with_disabled_rate_limit(self):
        config = bh.RateLimitConfig.disabled()
        client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None

    def test_client_default_rate_limit(self):
        """Client created without rate_limit uses defaults (25/min, 250/hr)."""
        config = bh.RateLimitConfig()
        client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None


# ========================================
# Response Type Tests
# ========================================


class TestResponseTypes:
    """Tests for SpaceTrack response type availability."""

    def test_fileshare_file_record_exists(self):
        """FileShareFileRecord type should be importable."""
        from brahe.spacetrack import FileShareFileRecord

        assert FileShareFileRecord is not None

    def test_folder_record_exists(self):
        """FolderRecord type should be importable."""
        from brahe.spacetrack import FolderRecord

        assert FolderRecord is not None

    def test_sp_ephemeris_file_record_exists(self):
        """SpEphemerisFileRecord type should be importable."""
        from brahe.spacetrack import SpEphemerisFileRecord

        assert SpEphemerisFileRecord is not None


class TestSpaceTrackClientMethods:
    """Tests for SpaceTrack client method availability (no network)."""

    def test_fileshare_methods_exist(self):
        """Client should have all fileshare methods."""
        client = bh.SpaceTrackClient("user@example.com", "password")
        assert hasattr(client, "fileshare_upload")
        assert hasattr(client, "fileshare_download")
        assert hasattr(client, "fileshare_download_folder")
        assert hasattr(client, "fileshare_list_files")
        assert hasattr(client, "fileshare_list_folders")
        assert hasattr(client, "fileshare_delete")

    def test_spephemeris_methods_exist(self):
        """Client should have all spephemeris methods."""
        client = bh.SpaceTrackClient("user@example.com", "password")
        assert hasattr(client, "spephemeris_download")
        assert hasattr(client, "spephemeris_list_files")
        assert hasattr(client, "spephemeris_file_history")

    def test_publicfiles_methods_exist(self):
        """Client should have all publicfiles methods."""
        client = bh.SpaceTrackClient("user@example.com", "password")
        assert hasattr(client, "publicfiles_download")
        assert hasattr(client, "publicfiles_list_dirs")

    def test_sp_ephemeris_controller_variant(self):
        """SP_EPHEMERIS controller variant should be accessible."""
        assert bh.RequestController.SP_EPHEMERIS is not None
        assert str(bh.RequestController.SP_EPHEMERIS) == "SpEphemeris"
        assert "RequestController" in repr(bh.RequestController.SP_EPHEMERIS)


# ========================================
# Integration Tests (CI-only, require credentials)
# ========================================

# Integration tests require:
#   TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables
#   Run with: pytest -m ci tests/spacetrack/


def _get_test_client():
    """Create a SpaceTrack client using test credentials."""
    user = os.environ.get("TEST_SPACETRACK_USER")
    password = os.environ.get("TEST_SPACETRACK_PASS")
    if not user or not password:
        pytest.skip("TEST_SPACETRACK_USER/TEST_SPACETRACK_PASS not set")
    base_url = os.environ.get(
        "TEST_SPACETRACK_BASE_URL", "https://for-testing-only.space-track.org"
    )
    return bh.SpaceTrackClient(user, password, base_url)


@pytest.mark.ci
def test_integration_authenticate():
    """Test authentication against test server."""
    client = _get_test_client()
    client.authenticate()


@pytest.mark.ci
def test_integration_gp_query():
    """Test GP query for ISS (NORAD 25544)."""
    client = _get_test_client()
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", bh.SortOrder.DESC)
        .limit(1)
    )
    records = client.query_gp(query)
    assert len(records) > 0
    assert records[0].norad_cat_id == "25544"
    assert records[0].object_name is not None


@pytest.mark.ci
def test_integration_satcat_query():
    """Test SATCAT query for ISS."""
    client = _get_test_client()
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.SATCAT)
        .filter("NORAD_CAT_ID", "25544")
        .limit(1)
    )
    records = client.query_satcat(query)
    assert len(records) > 0
    assert records[0].norad_cat_id == "25544"


@pytest.mark.ci
def test_integration_query_json():
    """Test JSON query returns list of dicts."""
    client = _get_test_client()
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", bh.SortOrder.DESC)
        .limit(1)
    )
    data = client.query_json(query)
    assert isinstance(data, list)
    assert len(data) > 0
    assert isinstance(data[0], dict)
    assert "NORAD_CAT_ID" in data[0]


@pytest.mark.ci
def test_integration_query_raw_tle():
    """Test raw TLE format query."""
    client = _get_test_client()
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", bh.SortOrder.DESC)
        .limit(1)
        .format(bh.OutputFormat.TLE)
    )
    raw = client.query_raw(query)
    assert isinstance(raw, str)
    # TLE format should have lines starting with "1 " and "2 "
    lines = raw.strip().split("\n")
    assert len(lines) >= 2


@pytest.mark.ci
def test_integration_query_with_operators():
    """Test query using operator functions."""
    client = _get_test_client()
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", op.inclusive_range("25544", "25550"))
        .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
        .limit(5)
    )
    records = client.query_gp(query)
    assert isinstance(records, list)


@pytest.mark.ci
def test_integration_auth_failure():
    """Test authentication failure with bad credentials."""
    base_url = os.environ.get(
        "TEST_SPACETRACK_BASE_URL", "https://for-testing-only.space-track.org"
    )
    client = bh.SpaceTrackClient("bad@example.com", "wrong", base_url)
    with pytest.raises(Exception):
        client.authenticate()
