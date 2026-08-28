"""Tests for CelestrakClient Python bindings."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

import brahe as bh


class TestCelestrakClientConstruction:
    """Tests for CelestrakClient construction."""

    def test_default_construction(self):
        client = bh.celestrak.CelestrakClient()
        assert client is not None

    def test_with_cache_age(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=3600.0)
        assert client is not None

    def test_with_base_url(self):
        client = bh.celestrak.CelestrakClient(base_url="https://test.celestrak.org")
        assert client is not None

    def test_with_base_url_and_cache_age(self):
        client = bh.celestrak.CelestrakClient(
            base_url="https://test.celestrak.org", cache_max_age=1800.0
        )
        assert client is not None

    def test_with_max_retries(self):
        client = bh.celestrak.CelestrakClient(max_retries=5)
        assert client is not None

    def test_with_max_retries_zero(self):
        client = bh.celestrak.CelestrakClient(max_retries=0)
        assert client is not None

    def test_with_all_options(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=3600.0, max_retries=2)
        assert client is not None


class TestCelestrakQueryClassattrs:
    """Tests for CelestrakQuery class attribute constructors."""

    def test_gp_classattr(self):
        query = bh.celestrak.CelestrakQuery.gp
        assert query is not None
        assert "CelestrakQuery" in repr(query)

    def test_sup_gp_classattr(self):
        query = bh.celestrak.CelestrakQuery.sup_gp
        assert query is not None

    def test_satcat_classattr(self):
        query = bh.celestrak.CelestrakQuery.satcat
        assert query is not None

    def test_gp_chaining(self):
        query = bh.celestrak.CelestrakQuery.gp.group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_satcat_chaining(self):
        query = bh.celestrak.CelestrakQuery.satcat.active(True)
        assert "ACTIVE=Y" in query.build_url()


class TestGetGpValidation:
    """Tests for get_gp() argument validation."""

    def test_get_gp_no_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp()

    def test_get_gp_multiple_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp(catnr=25544, name="ISS")


class TestGetSatcatValidation:
    """Tests for get_satcat() argument validation."""

    def test_get_satcat_no_args_raises(self):
        client = bh.celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="at least one"):
            client.get_satcat()


class TestCelestrakSATCATRecord:
    """Tests for CelestrakSATCATRecord attributes."""

    def test_import(self):
        assert hasattr(bh.celestrak, "CelestrakSATCATRecord")


class TestCelestrakClientNetworkMode:
    def test_offline_strict_miss_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
        monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
        client = bh.celestrak.CelestrakClient(
            base_url="https://brahe-network-mode-test.invalid"
        )
        with pytest.raises(bh.BraheError, match="offline-strict"):
            client.get_gp(name="ISS")


# -- CI-gated integration tests --


@pytest.mark.integration
class TestCelestrakClientIntegration:
    """Integration tests against live Celestrak API."""

    def test_get_gp_by_group(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(group="stations")
        assert len(records) > 0
        for r in records:
            assert r.object_name is not None

    def test_get_gp_by_catnr(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(catnr=25544)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544

    def test_get_gp_by_name(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(name="ISS")
        assert len(records) > 0

    def test_get_gp_returns_gprecord(self):
        """Verify GP queries return GPRecord (same as SpaceTrack)."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_gp(catnr=25544)
        assert len(records) > 0
        assert isinstance(records[0], bh.GPRecord)
        assert records[0].norad_cat_id is not None
        assert records[0].inclination is not None

    def test_get_satcat(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        records = client.get_satcat(catnr=25544)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544
        assert records[0].object_name is not None

    def test_query_raw(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.catnr(25544).format(
            bh.celestrak.CelestrakOutputFormat.THREE_LE
        )
        result = client.query_raw(query)
        assert "25544" in result

    def test_query_gp_with_filter(self):
        """Test query() with client-side filtering on live data."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.group("stations").filter(
            "INCLINATION", ">50"
        )
        records = client.query(query)
        for r in records:
            if r.inclination is not None:
                assert float(r.inclination) > 50.0

    def test_query_gp_with_limit(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.gp.group("stations").limit(2)
        records = client.query(query)
        assert len(records) <= 2

    def test_query_satcat(self):
        """Test query() dispatch for SATCAT queries."""
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        query = bh.celestrak.CelestrakQuery.satcat.catnr(25544)
        records = client.query(query)
        assert len(records) > 0
        assert records[0].norad_cat_id == 25544

    def test_get_sgp_propagator(self):
        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
        propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)
        assert propagator is not None
        assert isinstance(propagator, bh.SGPPropagator)


ACTIVE_RECORDS = [
    {
        "OBJECT_NAME": "ISS (ZARYA)",
        "OBJECT_ID": "1998-067A",
        "EPOCH": "2026-08-27T12:00:00.000000",
        "MEAN_MOTION": 15.49,
        "ECCENTRICITY": 0.0006,
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 120.5,
        "ARG_OF_PERICENTER": 30.2,
        "MEAN_ANOMALY": 329.9,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 25544,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 54000,
        "BSTAR": 0.0001,
        "MEAN_MOTION_DOT": 0.0001,
        "MEAN_MOTION_DDOT": 0,
    },
    {
        "OBJECT_NAME": "ISS (NAUKA)",
        "OBJECT_ID": "2021-066A",
        "EPOCH": "2026-08-27T12:00:00.000000",
        "MEAN_MOTION": 15.49,
        "ECCENTRICITY": 0.0006,
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 120.5,
        "ARG_OF_PERICENTER": 30.2,
        "MEAN_ANOMALY": 329.9,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 49044,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 28000,
        "BSTAR": 0.0001,
        "MEAN_MOTION_DOT": 0.0001,
        "MEAN_MOTION_DDOT": 0,
    },
]
SINGLE_RECORD = [
    {
        "OBJECT_NAME": "COSMOS 2251 DEB",
        "OBJECT_ID": "1993-036AAB",
        "EPOCH": "2026-08-27T12:00:00.000000",
        "MEAN_MOTION": 14.1,
        "ECCENTRICITY": 0.01,
        "INCLINATION": 74.0,
        "RA_OF_ASC_NODE": 10.0,
        "ARG_OF_PERICENTER": 20.0,
        "MEAN_ANOMALY": 30.0,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 34427,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 1,
        "BSTAR": 0.0001,
        "MEAN_MOTION_DOT": 0.0001,
        "MEAN_MOTION_DDOT": 0,
    },
]


@pytest.fixture
def celestrak_server(tmp_path, monkeypatch):
    """Serve a fake gp.php; yields (base_url, request_log) and isolates BRAHE_CACHE."""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.delenv("BRAHE_NETWORK_MODE", raising=False)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            requests.append(params)
            body = ACTIVE_RECORDS if params.get("GROUP") == "active" else SINGLE_RECORD
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()


class TestCelestrakClientActiveResolution:
    def test_catnr_resolves_from_active(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url)
        records = client.get_gp(catnr=25544)
        assert [r.norad_cat_id for r in records] == [25544]
        assert [r.get("GROUP") for r in requests] == ["active"]

    def test_name_matches_substring_case_insensitive(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url)
        records = client.get_gp(name="iss")
        assert sorted(r.object_name for r in records) == ["ISS (NAUKA)", "ISS (ZARYA)"]
        assert len(requests) == 1

    def test_object_absent_from_active_is_requested_directly(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url)
        records = client.get_gp(catnr=34427)
        assert records[0].object_name == "COSMOS 2251 DEB"
        assert [r.get("GROUP") or r.get("CATNR") for r in requests] == [
            "active",
            "34427",
        ]

    def test_group_query_does_not_fetch_active(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url)
        client.get_gp(group="stations")
        assert [r.get("GROUP") for r in requests] == ["stations"]

    def test_sgp_propagator_from_active(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url)
        propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)
        assert isinstance(propagator, bh.SGPPropagator)
        assert [r.get("GROUP") for r in requests] == ["active"]

    def test_zero_cache_age_sends_query_directly(self, celestrak_server):
        base_url, requests = celestrak_server
        client = bh.celestrak.CelestrakClient(base_url=base_url, cache_max_age=0.0)
        client.get_gp(catnr=25544)
        assert [r.get("GROUP") for r in requests] == [None]
        assert requests[0].get("CATNR") == "25544"
