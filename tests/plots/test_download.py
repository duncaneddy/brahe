"""Unit tests for the shared atomic zip download helper.

These tests serve a real zip over a local HTTP server (stdlib only), so they
exercise the genuine download -> validate -> extract -> atomic-publish path
without any external network access or mocking. They are intentionally NOT
marked ``integration`` so they run on every PR and guard the de-flake logic.
"""

import io
import threading
import zipfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from brahe.plots._download import download_and_extract_zip, download_file


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    """Write a zip file at ``path`` containing ``members`` (arcname -> bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for arcname, data in members.items():
            zf.writestr(arcname, data)
    path.write_bytes(buf.getvalue())


@pytest.fixture
def zip_server(tmp_path):
    """Serve a temp directory over localhost; yields (base_url, served_dir)."""
    served_dir = tmp_path / "served"
    served_dir.mkdir()

    handler = partial(SimpleHTTPRequestHandler, directory=str(served_dir))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", served_dir
    finally:
        server.shutdown()
        server.server_close()


def test_downloads_and_extracts_zip(zip_server, tmp_path):
    base_url, served_dir = zip_server
    _make_zip(served_dir / "thing.zip", {"data.txt": b"hello"})

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "data.txt"

    result = download_and_extract_zip(
        f"{base_url}/thing.zip",
        extract_dir,
        sentinel,
        description="thing",
        retry_delay=0,
    )

    assert result == sentinel
    assert sentinel.read_bytes() == b"hello"


def test_fast_path_when_sentinel_exists(tmp_path):
    extract_dir = tmp_path / "cache" / "thing"
    extract_dir.mkdir(parents=True)
    sentinel = extract_dir / "data.txt"
    sentinel.write_bytes(b"cached")

    # URL points at a closed port; if the fast path works, it is never hit.
    result = download_and_extract_zip(
        "http://127.0.0.1:1/thing.zip",
        extract_dir,
        sentinel,
        description="thing",
        retry_delay=0,
    )

    assert result == sentinel
    assert sentinel.read_bytes() == b"cached"


def test_invalid_zip_raises_after_retries(zip_server, tmp_path):
    base_url, served_dir = zip_server
    # Not a real zip: no PK magic bytes.
    (served_dir / "thing.zip").write_bytes(b"<html>404 from a flaky CDN</html>")

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "data.txt"

    with pytest.raises(RuntimeError):
        download_and_extract_zip(
            f"{base_url}/thing.zip",
            extract_dir,
            sentinel,
            description="thing",
            max_retries=2,
            retry_delay=0,
        )

    assert not sentinel.exists()


def test_http_error_raises_after_retries(zip_server, tmp_path):
    base_url, _served_dir = zip_server
    # No file is created on the server, so the request 404s and
    # raise_for_status() raises httpx.HTTPError.

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "data.txt"

    with pytest.raises(RuntimeError):
        download_and_extract_zip(
            f"{base_url}/missing.zip",
            extract_dir,
            sentinel,
            description="thing",
            max_retries=2,
            retry_delay=0,
        )

    assert not sentinel.exists()


def test_relocate_error_propagates(zip_server, tmp_path):
    base_url, served_dir = zip_server
    _make_zip(served_dir / "thing.zip", {"data.txt": b"hello"})

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "data.txt"

    def relocate(staging: Path) -> None:
        raise RuntimeError("layout changed")

    with pytest.raises(RuntimeError, match="layout changed"):
        download_and_extract_zip(
            f"{base_url}/thing.zip",
            extract_dir,
            sentinel,
            description="thing",
            relocate=relocate,
            retry_delay=0,
        )


def test_relocate_repositions_nested_file(zip_server, tmp_path):
    base_url, served_dir = zip_server
    _make_zip(served_dir / "thing.zip", {"nested/sub/raster.tif": b"PIXELS"})

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "raster.tif"

    def relocate(staging: Path) -> None:
        found = next(staging.rglob("raster.tif"))
        found.rename(staging / "raster.tif")

    result = download_and_extract_zip(
        f"{base_url}/thing.zip",
        extract_dir,
        sentinel,
        description="thing",
        relocate=relocate,
        retry_delay=0,
    )

    assert result == sentinel
    assert sentinel.read_bytes() == b"PIXELS"


def test_concurrent_callers_do_not_corrupt(zip_server, tmp_path):
    """Reproduces the original race: many workers fetch the same resource."""
    base_url, served_dir = zip_server
    _make_zip(served_dir / "thing.zip", {"data.txt": b"shared-payload"})

    extract_dir = tmp_path / "cache" / "thing"
    sentinel = extract_dir / "data.txt"

    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(
                download_and_extract_zip(
                    f"{base_url}/thing.zip",
                    extract_dir,
                    sentinel,
                    description="thing",
                    retry_delay=0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 6
    assert sentinel.read_bytes() == b"shared-payload"


def test_download_file_validation_rejects_bad_content(zip_server, tmp_path):
    """A validator that rejects the staged content must discard the temp
    file and raise, rather than publishing (and permanently caching) a bad
    response to ``dest``."""
    base_url, served_dir = zip_server
    (served_dir / "bad.jpg").write_bytes(b"<html>404 from a flaky CDN</html>")

    dest = tmp_path / "cache" / "bad.jpg"

    with pytest.raises(RuntimeError):
        download_file(
            f"{base_url}/bad.jpg",
            dest,
            description="thing",
            max_retries=2,
            retry_delay=0,
            validate=lambda p: False,
        )

    assert not dest.exists()
