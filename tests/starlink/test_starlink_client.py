"""Tests mirroring src/clients/starlink/client.rs."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import brahe as bh

ASSETS = Path(__file__).resolve().parents[2] / "test_assets" / "starlink"
FULL_FILE = "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
SHORT_FILE = (
    "MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"
)
TWO_LINE_MANIFEST = f"{FULL_FILE}\n{SHORT_FILE}\n"


def age_file(path, seconds):
    past = time.time() - seconds
    os.utime(path, (past, past))


@pytest.fixture
def starlink_server(tmp_path, monkeypatch):
    """Serve MANIFEST.txt and the two committed assets; yields (base_url, hits, files)."""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "online")
    hits = []
    files = {
        "/MANIFEST.txt": (
            TWO_LINE_MANIFEST,
            {"ETag": '"m1"', "Last-Modified": "Fri, 11 Sep 2026 05:15:30 GMT"},
        ),
        f"/{FULL_FILE}": ((ASSETS / FULL_FILE).read_text(), {}),
        f"/{SHORT_FILE}": ((ASSETS / SHORT_FILE).read_text(), {}),
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            hits.append((self.path, self.headers.get("If-None-Match")))
            entry = files.get(self.path)
            if entry is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                return
            body, headers = entry
            if self.path == "/MANIFEST.txt" and self.headers.get(
                "If-None-Match"
            ) == headers.get("ETag"):
                self.send_response(304)
                self.send_header("ETag", headers["ETag"])
                self.send_header("Connection", "close")
                self.end_headers()
                return
            payload = body.encode()
            self.send_response(200)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", hits, files
    finally:
        server.shutdown()
        server.server_close()


def cache_dir(tmp_path):
    return tmp_path / "starlink"


def test_starlink_client_constructors(tmp_path, monkeypatch):
    """Rust: test_starlink_client_constructors"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    client = bh.StarlinkClient()
    assert client.base_url == "https://api.starlink.com/public-files/ephemerides"
    assert client.cache_max_age == 3600.0
    client = bh.StarlinkClient(
        base_url="http://127.0.0.1:1/", cache_max_age=10.0, max_retries=0
    )
    assert client.base_url == "http://127.0.0.1:1"
    assert client.cache_max_age == 10.0
    limited = bh.StarlinkClient(
        rate_limit=bh.RateLimitConfig(max_per_minute=1, max_per_hour=1)
    )
    assert limited.cache_max_age == 3600.0
    assert Path(client.cache_dir()).name == "starlink"


def test_get_manifest_downloads_and_caches(starlink_server, tmp_path):
    """Rust: test_get_manifest_downloads_and_caches"""
    base_url, hits, _ = starlink_server
    client = bh.StarlinkClient(base_url=base_url)
    manifest = client.get_manifest()
    assert len(manifest) == 2
    assert manifest.last_modified is not None
    meta = json.loads((cache_dir(tmp_path) / "MANIFEST.meta.json").read_text())
    assert meta["etag"] == '"m1"'
    assert (cache_dir(tmp_path) / "MANIFEST.txt").read_text() == TWO_LINE_MANIFEST
    client.get_manifest()
    assert [h[0] for h in hits] == ["/MANIFEST.txt"]
    assert len(client.cached_manifest()) == 2
    assert client.previous_manifest() is None


def test_get_manifest_stale_uses_conditional_get_and_304_keeps_cache(
    starlink_server, tmp_path
):
    """Rust: test_get_manifest_stale_uses_conditional_get_and_304_keeps_cache"""
    base_url, hits, _ = starlink_server
    client = bh.StarlinkClient(base_url=base_url, cache_max_age=3600.0)
    client.get_manifest()
    age_file(cache_dir(tmp_path) / "MANIFEST.txt", 7200)
    manifest = client.get_manifest()
    assert len(manifest) == 2
    assert hits[-1] == ("/MANIFEST.txt", '"m1"')
    assert time.time() - (cache_dir(tmp_path) / "MANIFEST.txt").stat().st_mtime < 60
    assert client.previous_manifest() is None


def test_get_manifest_changed_rotates_previous(starlink_server, tmp_path):
    """Rust: test_get_manifest_changed_rotates_previous"""
    base_url, _hits, files = starlink_server
    client = bh.StarlinkClient(base_url=base_url)
    client.get_manifest()
    updated = f"MEME_100001_STARLINK-38128_2540942_Operational_1473414180_UNCLASSIFIED.txt\n{SHORT_FILE}\n"
    files["/MANIFEST.txt"] = (updated, {"ETag": '"m2"'})
    age_file(cache_dir(tmp_path) / "MANIFEST.txt", 7200)
    manifest = client.refresh_manifest()
    assert manifest.find_by_norad_id(100001).file_name.metadata == "1473414180"
    assert (
        cache_dir(tmp_path) / "MANIFEST.previous.txt"
    ).read_text() == TWO_LINE_MANIFEST
    previous = client.previous_manifest()
    assert [e.norad_cat_id for e in manifest.changed_since(previous)] == [100001]
    assert (
        json.loads((cache_dir(tmp_path) / "MANIFEST.meta.json").read_text())["etag"]
        == '"m2"'
    )


def test_get_manifest_offline_strict_serves_fresh_cache_only(tmp_path, monkeypatch):
    """Rust: test_get_manifest_offline_strict_serves_fresh_cache_only"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
    client = bh.StarlinkClient(base_url="https://brahe-network-mode-test.invalid")
    with pytest.raises(bh.BraheError, match="BRAHE_NETWORK_MODE is offline-strict"):
        client.get_manifest()
    cache_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    (cache_dir(tmp_path) / "MANIFEST.txt").write_text(TWO_LINE_MANIFEST)
    assert len(client.get_manifest()) == 2
    age_file(cache_dir(tmp_path) / "MANIFEST.txt", 7200)
    with pytest.raises(bh.BraheError, match="offline-strict"):
        client.get_manifest()
    with pytest.raises(bh.BraheError, match="BRAHE_NETWORK_MODE is offline-strict"):
        client.refresh_manifest()


def test_get_manifest_offline_serves_stale_cache(tmp_path, monkeypatch):
    """Rust: test_get_manifest_offline_serves_stale_cache"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    cache_dir(tmp_path).mkdir(parents=True)
    (cache_dir(tmp_path) / "MANIFEST.txt").write_text(TWO_LINE_MANIFEST)
    age_file(cache_dir(tmp_path) / "MANIFEST.txt", 7200)
    client = bh.StarlinkClient(base_url="https://brahe-network-mode-test.invalid")
    assert len(client.get_manifest()) == 2
    with pytest.raises(bh.BraheError, match="BRAHE_NETWORK_MODE is offline"):
        client.refresh_manifest()


def test_get_manifest_online_refresh_failure_is_an_error(tmp_path, monkeypatch):
    """Rust: test_get_manifest_online_refresh_failure_is_an_error"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "online")
    cache_dir(tmp_path).mkdir(parents=True)
    (cache_dir(tmp_path) / "MANIFEST.txt").write_text(TWO_LINE_MANIFEST)
    client = bh.StarlinkClient(base_url="http://127.0.0.1:1", max_retries=0)
    assert len(client.get_manifest()) == 2
    age_file(cache_dir(tmp_path) / "MANIFEST.txt", 7200)
    with pytest.raises(bh.BraheError):
        client.get_manifest()
    assert (cache_dir(tmp_path) / "MANIFEST.txt").read_text() == TWO_LINE_MANIFEST
    assert len(client.cached_manifest()) == 2


def test_fetch_text_retries_then_fails_on_server_error(tmp_path, monkeypatch):
    """Rust: test_fetch_text_retries_then_fails_on_server_error"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "online")
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        client = bh.StarlinkClient(base_url=f"http://{host}:{port}", max_retries=2)
        with pytest.raises(bh.BraheError, match="503"):
            client.get_manifest()
        assert len(hits) == 3
    finally:
        server.shutdown()
        server.server_close()


def test_download_ephemeris_caches_and_evicts_superseded(starlink_server, tmp_path):
    """Rust: test_download_ephemeris_caches_and_evicts_superseded"""
    base_url, _, _ = starlink_server
    d = cache_dir(tmp_path)
    d.mkdir(parents=True)
    old_name = (
        "MEME_100002_STARLINK-37711_2530149_Operational_1472521800_UNCLASSIFIED.txt"
    )
    unrelated = (
        "MEME_100003_STARLINK-38123_2540140_Operational_1473385260_UNCLASSIFIED.txt"
    )
    (d / old_name).write_text("x")
    (d / unrelated).write_text("x")
    client = bh.StarlinkClient(base_url=base_url)
    path = Path(client.download_ephemeris(100002))
    assert path == d / SHORT_FILE
    assert path.read_text() == (ASSETS / SHORT_FILE).read_text()
    assert not (d / old_name).exists()
    assert (d / unrelated).exists()
    assert [Path(p).name for p in client.cached_files()] == [SHORT_FILE, unrelated]


def test_download_ephemeris_serves_cached_file_without_request(
    starlink_server, tmp_path
):
    """Rust: test_download_ephemeris_serves_cached_file_without_request"""
    base_url, hits, _ = starlink_server
    d = cache_dir(tmp_path)
    d.mkdir(parents=True)
    (d / SHORT_FILE).write_text((ASSETS / SHORT_FILE).read_text())
    client = bh.StarlinkClient(base_url=base_url)
    client.download_ephemeris(100002)
    assert [h[0] for h in hits] == ["/MANIFEST.txt"]
    itc = client.get_ephemeris(100002)
    assert len(itc) == 50
    traj = client.get_trajectory(100002)
    assert len(traj) == 50
    assert traj.covariance(traj.start_epoch()) is not None
    rotating = client.get_trajectory(
        100002, covariance_variant=bh.OrbitRelativeFrameVariant.ROTATING
    )
    assert len(rotating) == 50
    explicit = client.get_trajectory_with_covariance_variant(
        100002, bh.OrbitRelativeFrameVariant.INERTIAL
    )
    assert len(explicit) == 50


def test_download_ephemeris_unknown_id_malformed_body_and_offline(
    starlink_server, tmp_path, monkeypatch
):
    """Rust: test_download_ephemeris_unknown_id_malformed_body_and_offline"""
    base_url, _, files = starlink_server
    files[f"/{FULL_FILE}"] = ("not an ephemeris\n", {})
    d = cache_dir(tmp_path)
    client = bh.StarlinkClient(base_url=base_url)
    with pytest.raises(bh.BraheError, match="424242"):
        client.download_ephemeris(424242)
    with pytest.raises(bh.BraheError):
        client.download_ephemeris(100001)
    assert not (d / FULL_FILE).exists()
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
    strict = bh.StarlinkClient(base_url="https://brahe-network-mode-test.invalid")
    with pytest.raises(bh.BraheError, match="BRAHE_NETWORK_MODE is offline-strict"):
        strict.download_ephemeris(100002)
    (d / SHORT_FILE).write_text((ASSETS / SHORT_FILE).read_text())
    assert Path(strict.download_ephemeris(100002)) == d / SHORT_FILE


def test_save_ephemeris_directory_and_file_destinations(starlink_server, tmp_path):
    """Rust: test_save_ephemeris_directory_and_file_destinations"""
    base_url, _, _ = starlink_server
    out = tmp_path / "out"
    client = bh.StarlinkClient(base_url=base_url)
    existing = out / "existing"
    existing.mkdir(parents=True)
    assert Path(client.save_ephemeris(100002, str(existing))) == existing / SHORT_FILE
    new_dir = out / "nested" / "new_dir"
    assert Path(client.save_ephemeris(100002, str(new_dir))) == new_dir / SHORT_FILE
    assert new_dir.is_dir()
    renamed = out / "renamed" / "sat.txt"
    assert Path(client.save_ephemeris(100002, str(renamed))) == renamed
    assert renamed.read_text() == (ASSETS / SHORT_FILE).read_text()
    assert Path(client.save_ephemeris(100002, str(renamed))) == renamed
    assert client.prune_cache() == 0
    assert renamed.exists()


def test_download_all_downloads_missing_and_keeps_cached(starlink_server, tmp_path):
    """Rust: test_download_all_downloads_missing_and_keeps_cached"""
    base_url, hits, _ = starlink_server
    d = cache_dir(tmp_path)
    d.mkdir(parents=True)
    (d / SHORT_FILE).write_text((ASSETS / SHORT_FILE).read_text())
    stale = "MEME_100009_STARLINK-9_2530149_Operational_1472521800_UNCLASSIFIED.txt"
    (d / stale).write_text("x")
    client = bh.StarlinkClient(base_url=base_url)
    paths = [Path(p) for p in client.download_all(concurrency=4)]
    assert paths == [d / FULL_FILE, d / SHORT_FILE]
    assert [h[0] for h in hits] == ["/MANIFEST.txt", f"/{FULL_FILE}"]
    assert (d / stale).exists()
    with pytest.raises(bh.BraheError):
        client.download_all(concurrency=0)
    assert client.prune_cache() == 1
    assert not (d / stale).exists()
    assert [Path(p) for p in client.download_all()] == [d / FULL_FILE, d / SHORT_FILE]


def test_download_all_stops_on_first_error(starlink_server, tmp_path):
    """Rust: test_download_all_stops_on_first_error"""
    base_url, _, files = starlink_server
    del files[f"/{FULL_FILE}"]
    d = cache_dir(tmp_path)
    client = bh.StarlinkClient(base_url=base_url, max_retries=0)
    with pytest.raises(bh.BraheError, match="404"):
        client.download_all(concurrency=1)
    assert not (d / SHORT_FILE).exists()
    assert not (d / FULL_FILE).exists()


def test_save_all_copies_into_directory(starlink_server, tmp_path):
    """Rust: test_save_all_copies_into_directory"""
    base_url, _, _ = starlink_server
    dest = tmp_path / "out" / "ephemerides"
    client = bh.StarlinkClient(base_url=base_url)
    saved = [Path(p) for p in client.save_all(str(dest), concurrency=2)]
    assert saved == [dest / FULL_FILE, dest / SHORT_FILE]
    assert (dest / FULL_FILE).read_text() == (ASSETS / FULL_FILE).read_text()
    file_dest = tmp_path / "a_file.txt"
    file_dest.write_text("x")
    with pytest.raises(bh.BraheError):
        client.save_all(str(file_dest))
    assert (cache_dir(tmp_path) / FULL_FILE).exists()


def test_starlink_client_releases_gil(starlink_server):
    """A download running in one thread must not block Python code in another."""
    base_url, _, _ = starlink_server
    client = bh.StarlinkClient(base_url=base_url)
    progressed = threading.Event()

    def ticker():
        progressed.set()

    worker = threading.Thread(target=lambda: client.download_all(concurrency=2))
    worker.start()
    t = threading.Thread(target=ticker)
    t.start()
    t.join(timeout=5.0)
    worker.join(timeout=30.0)
    assert progressed.is_set()
    assert not worker.is_alive()


# -- CI-gated integration tests --


@pytest.mark.integration
class TestStarlinkClientIntegration:
    """Integration tests against the live Starlink mirror."""

    def test_manifest_and_one_download(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
        monkeypatch.delenv("BRAHE_NETWORK_MODE", raising=False)
        client = bh.StarlinkClient(cache_max_age=0.0)
        manifest = client.get_manifest()
        assert len(manifest) > 1000
        entry = manifest.entries()[0]
        assert entry.ephemeris_stop is not None
        traj = client.get_trajectory(entry.norad_cat_id)
        assert len(traj) > 100
        df = manifest.to_dataframe()
        assert df.height == len(manifest)
