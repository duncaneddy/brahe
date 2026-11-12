"""Tests mirroring src/clients/igs/client.rs and src/clients/igs/source.rs."""

import gzip
import itertools
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import brahe as bh

ROOT = Path(__file__).resolve().parents[2]
IGS_ASSETS = ROOT / "test_assets" / "igs"
SP3_FIXTURE = (
    ROOT / "test_assets" / "sp3" / "IGS0OPSFIN_20243150000_01D_15M_ORB_subset.SP3.gz"
)
GARNER_PATH = "/pub/products/"
OFFLINE_BASE_URL = "https://brahe-network-mode-test.invalid/pub/products/"
SP3_NAME = "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"
SP3_TEXT = "#dP2024 11 10  0  0  0.00000000     289 ORBIT IGS20 FIT  COD\n"
SHORT_Z_NAME = "igr23400.sp3.Z"
SHORT_Z_BYTES = bytes([0x1F, 0x9D, 0x90, 0x23, 0x64, 0x50])
UNIX_SP3_FIXTURE = ROOT / "test_assets" / "sp3" / "gbm18432.sp3.Z"


def gps(year, month, day, hour):
    return bh.Epoch(year, month, day, hour, 0, 0.0, 0.0, time_system=bh.GPS)


def week_2340():
    """An instant in GPS week 2340 (2024-11-10 through 2024-11-16)."""
    return gps(2024, 11, 12, 12)


def asset(name):
    return (IGS_ASSETS / name).read_text()


def apache_listing(names):
    """A minimal Apache-style listing of names, with a sort link and a parent link."""
    rows = "".join(
        f'<tr><td><a href="{name}">{name}</a></td><td align="right">1.0M</td></tr>\n'
        for name in names
    )
    return (
        '<html><body><table>\n<tr><th><a href="?C=N;O=D">Name</a></th></tr>\n'
        '<tr><td><a href="/pub/products/">Parent Directory</a></td></tr>\n'
        f"{rows}</table></body></html>\n"
    )


def names(products):
    return [product.name for product in products]


def age_file(path, seconds):
    past = time.time() - seconds
    os.utime(path, (past, past))


def same_sp3(sp3, fixture):
    return isinstance(sp3, bh.SP3) and sp3.to_string() == fixture.to_string()


def cod_daily(doy, sampling):
    """Name of the COD final daily orbit for 2024 day of year doy."""
    return f"COD0OPSFIN_2024{doy:03d}0000_01D_{sampling}_ORB.SP3.gz"


class Archive:
    """Routes, request log, and cache control for the mock archive server."""

    def __init__(self, tmp_path, monkeypatch):
        self.routes = {}
        self.hits = []
        self.user_agents = []
        self.base = ""
        self._tmp_path = tmp_path
        self._monkeypatch = monkeypatch
        self._caches = 0
        self.fresh_cache()

    def fresh_cache(self):
        """Point BRAHE_CACHE at a new, empty directory."""
        self._caches += 1
        cache = self._tmp_path / f"cache{self._caches}"
        self._monkeypatch.setenv("BRAHE_CACHE", str(cache))
        return cache

    def url(self, path):
        return f"{self.base}{path}"

    def serve(self, path, body, status=200, headers=None):
        """Answer GET path (query ignored) with body, status, and extra headers."""
        if isinstance(body, str):
            body = body.encode()
        self.routes[path] = (status, body, headers or {})

    def redirect(self, path, location):
        """Answer GET path with a 302 to location."""
        self.serve(path, b"", status=302, headers={"Location": location})

    def missing(self, path):
        self.serve(path, b"", status=404)

    def listing(self, path, body):
        self.serve(path, body)

    def calls(self, path):
        return sum(1 for hit in self.hits if urlsplit(hit).path == path)

    def reset(self):
        self.hits.clear()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """Serve the registered routes over HTTP on 127.0.0.1; yields an Archive."""
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "online")
    state = Archive(tmp_path, monkeypatch)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            state.hits.append(self.path)
            state.user_agents.append(self.headers.get("User-Agent"))
            status, body, headers = state.routes.get(
                urlsplit(self.path).path, (404, b"", {})
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    state.base = f"http://{host}:{port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def garner_client(archive, **kwargs):
    return bh.IGSProductsClient(
        source=bh.IGSProductSource.GARNER, base_url=archive.url(GARNER_PATH), **kwargs
    )


def offline_garner_client():
    """A Garner client whose base URL is not loopback, so offline modes apply."""
    return bh.IGSProductsClient(
        source="GARNER", base_url=OFFLINE_BASE_URL, max_retries=0
    )


def listed_product(archive, client, name, week, epoch):
    """The product named name from a mocked listing of week holding only it."""
    archive.listing(f"{GARNER_PATH}{week:04d}/", apache_listing([name]))
    products = client.list_products(epoch)
    archive.reset()
    return next(product for product in products if product.name == name)


# Sources


def test_igs_product_source_default_base_url():
    """Rust: test_igs_product_source_default_base_url"""
    assert (
        bh.IGSProductSource.CDDIS.default_base_url
        == "https://cddis.nasa.gov/archive/gnss/products/"
    )
    assert (
        bh.IGSProductSource.GARNER.default_base_url
        == "http://garner.ucsd.edu/pub/products/"
    )
    assert (
        bh.IGSProductSource.BKG.default_base_url
        == "https://igs.bkg.bund.de/root_ftp/IGS/products/"
    )


def test_igs_product_source_display():
    """Rust: test_igs_product_source_display"""
    assert str(bh.IGSProductSource.CDDIS) == "CDDIS"
    assert str(bh.IGSProductSource.GARNER) == "Garner"
    assert str(bh.IGSProductSource.BKG) == "BKG"
    assert repr(bh.IGSProductSource.GARNER) == "IGSProductSource.GARNER"
    assert bh.IGSProductSource.BKG == bh.IGSProductSource.BKG
    assert bh.IGSProductSource.BKG != bh.IGSProductSource.CDDIS
    assert len({bh.IGSProductSource.BKG, bh.IGSProductSource.BKG}) == 1


# Construction


def test_igs_products_client_new_uses_default_base_url():
    """Rust: test_igs_products_client_new_uses_default_base_url"""
    for source, code in [
        (bh.IGSProductSource.CDDIS, "cddis"),
        (bh.IGSProductSource.GARNER, "Garner"),
        (bh.IGSProductSource.BKG, "BKG"),
    ]:
        for given in (source, code):
            for client in (
                bh.IGSProductsClient(given),
                bh.IGSProductsClient(source=given),
            ):
                assert client.source == source
                assert client.base_url == source.default_base_url


def test_igs_products_client_with_cddis_credentials():
    """Rust: test_igs_products_client_with_cddis_credentials"""
    for source in ("CDDIS", bh.IGSProductSource.CDDIS):
        client = bh.IGSProductsClient(
            source, cddis_username="user", cddis_password="secret"
        )
        assert client.source == bh.IGSProductSource.CDDIS
        assert client.base_url == bh.IGSProductSource.CDDIS.default_base_url


def test_igs_products_client_debug_omits_credentials():
    """Rust: test_igs_products_client_debug_omits_credentials"""
    client = bh.IGSProductsClient("CDDIS", cddis_username="u", cddis_password="secret")
    text = repr(client)
    assert "secret" not in text
    assert "CDDIS" in text


def test_igs_products_client_with_base_url_normalises_trailing_slash():
    """Rust: test_igs_products_client_with_base_url_normalises_trailing_slash"""
    client = bh.IGSProductsClient(source="BKG", base_url="http://localhost:1/x/")
    assert client.source == bh.IGSProductSource.BKG
    assert client.base_url == "http://localhost:1/x/"
    client = bh.IGSProductsClient(source="BKG", base_url="http://localhost:1/x")
    assert client.base_url == "http://localhost:1/x/"


def test_igs_products_client_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        bh.IGSProductsClient()
    with pytest.raises(TypeError):
        bh.IGSProductsClient(base_url="http://localhost:1/")
    with pytest.raises(ValueError, match="together"):
        bh.IGSProductsClient("CDDIS", cddis_username="user")
    with pytest.raises(ValueError, match="together"):
        bh.IGSProductsClient("CDDIS", cddis_password="secret")
    with pytest.raises(ValueError, match="only to the CDDIS source"):
        bh.IGSProductsClient(source="GARNER", cddis_username="u", cddis_password="p")
    with pytest.raises(ValueError, match="cannot be combined with base_url"):
        bh.IGSProductsClient(
            source="CDDIS",
            base_url="http://localhost:1/",
            cddis_username="u",
            cddis_password="p",
        )
    with pytest.raises(ValueError, match="unknown IGS product source"):
        bh.IGSProductsClient(source="SIO")
    with pytest.raises(TypeError):
        bh.IGSProductsClient(source=3)
    with pytest.raises(TypeError):
        bh.IGSProductsClient("GARNER", "http://localhost:1/")


def test_igs_products_client_cache_dir(tmp_path, monkeypatch):
    """Rust: test_igs_products_client_cache_dir"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    directory = Path(bh.IGSProductsClient(source="GARNER").cache_dir())
    assert directory.name == "igs"
    assert directory.is_dir()
    assert directory.parent == tmp_path
    assert Path(bh.igs.get_igs_cache_dir()) == directory
    assert bh.igs.get_igs_cache_dir is bh.get_igs_cache_dir


# Listings


def test_igs_products_client_list_weeks(archive):
    """Rust: test_igs_products_client_list_weeks"""
    archive.listing(GARNER_PATH, asset("garner_root.html"))
    client = garner_client(archive)
    assert client.list_weeks() == [
        576,
        577,
        578,
        579,
        580,
        2339,
        2340,
        2341,
        2442,
        2443,
    ]
    assert archive.calls(GARNER_PATH) == 1
    cached = Path(client.cache_dir()) / "listings" / "garner" / "root.txt"
    assert cached.read_text() == asset("garner_root.html")


def test_igs_products_client_list_files_skips_listing_noise(archive):
    """Rust: test_igs_products_client_list_files_skips_listing_noise"""
    for source, fixture in [("Garner", "garner_2340.html"), ("BKG", "bkg_2340.html")]:
        archive.listing(f"/{source}/2340/", asset(fixture))
        client = bh.IGSProductsClient(
            source=source, base_url=archive.url(f"/{source}/")
        )
        files = client.list_files(week_2340())
        assert archive.calls(f"/{source}/2340/") == 1

        assert "readme.txt" in files, source
        assert "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz" in files, source
        for name in files:
            assert not name.startswith("?"), name
            assert "/" not in name, name
            assert not name.startswith("repro3"), name
        hrefs = re.findall(r'href="([^"]*)"', asset(fixture))
        expected = [h for h in hrefs if not h.startswith("?") and "/" not in h]
        assert len(files) == len(expected), source


def test_igs_products_client_list_products_center_is_case_insensitive(archive):
    """Rust: test_igs_products_client_list_products_center_is_case_insensitive"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    client = garner_client(archive)

    lower = client.list_products(week_2340(), center="cod")
    upper = client.list_products(week_2340(), center="COD")
    typed = client.list_products(week_2340(), center=bh.IGSAnalysisCenter.COD)
    assert lower == upper == typed
    assert len(lower) >= 20
    assert all(
        product.file_name.center == bh.IGSAnalysisCenter.COD for product in lower
    )
    assert SP3_NAME in names(lower)
    assert not any(name.startswith("ESA") for name in names(lower))
    assert archive.calls("/pub/products/2340/") == 1


def test_igs_products_client_list_products_filters_solution_and_content(archive):
    """Rust: test_igs_products_client_list_products_filters_solution_and_content"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    client = garner_client(archive)

    products = client.list_products(
        week_2340(),
        solution=bh.IGSSolutionType.FINAL,
        content=bh.IGSContentType.ORBIT,
    )
    assert products
    for product in products:
        assert product.file_name.solution == bh.IGSSolutionType.FINAL
        assert product.file_name.content == bh.IGSContentType.ORBIT
    listed = names(products)
    assert "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz" in listed
    assert "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz" in listed
    assert "JPL0OPSFIN_20243150000_01D_05M_ORB.SP3.gz" in listed
    assert "COD0OPSRAP_20243150000_01D_05M_ORB.SP3.gz" not in listed
    assert "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz" not in listed
    assert client.list_products(week_2340(), solution="fin", content="ORB") == products

    expected_clocks = [
        "ESA0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
        "ESA0OPSRAP_20243150000_01D_05M_CLK.CLK.gz",
    ]
    clocks = client.list_products(
        week_2340(), center="ESA", format=bh.IGSFileFormat.CLK
    )
    assert names(clocks) == expected_clocks
    assert names(client.list_products(week_2340(), center="esa", format="clk")) == (
        expected_clocks
    )

    for kwarg in ("solution", "content", "format"):
        with pytest.raises(TypeError):
            client.list_products(week_2340(), **{kwarg: 1})


def test_igs_products_client_list_products_skips_unparseable_names(archive):
    """Rust: test_igs_products_client_list_products_skips_unparseable_names"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    client = garner_client(archive)

    files = client.list_files(week_2340())
    products = client.list_products(week_2340())
    listed = names(products)
    for raw in ["readme.txt", "index315.html", "gpgga4.315.rap.Z"]:
        assert raw in files, raw
        assert raw not in listed, raw
    assert "igr23400.sp3.Z" in listed
    assert len(products) < len(files)
    for product in products:
        assert product.name in files
        assert str(product.file_name) == product.name
        assert product.url == f"{client.base_url}2340/{product.name}"
        assert product.size is None
        assert product.gps_week == 2340
        assert isinstance(product.file_name, bh.IGSProductFileName)


def test_igs_products_client_cddis_listing_reports_sizes(archive):
    """Rust: test_igs_products_client_cddis_listing_reports_sizes"""
    archive.listing("/archive/gnss/products/2340/*", asset("cddis_2340.txt"))
    client = bh.IGSProductsClient(
        source="CDDIS", base_url=archive.url("/archive/gnss/products/")
    )

    files = client.list_files(week_2340())
    assert len(files) == len(asset("cddis_2340.txt").splitlines())
    assert "readme.txt" in files

    products = client.list_products(
        week_2340(), center="cod", solution="FIN", content="ORB"
    )
    orbit = next(product for product in products if product.name == SP3_NAME)
    assert orbit.size == 617472
    assert orbit.url == archive.url(f"/archive/gnss/products/2340/{SP3_NAME}")
    assert all(product.size is not None for product in products)
    assert archive.calls("/archive/gnss/products/2340/*") == 1
    assert all(urlsplit(hit).query == "list" for hit in archive.hits)


def test_igs_products_client_list_products_between_spans_weeks(archive):
    """Rust: test_igs_products_client_list_products_between_spans_weeks"""
    archive.listing("/pub/products/2339/", apache_listing([]))
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    archive.listing(
        "/pub/products/2341/",
        apache_listing(
            [
                "COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243240000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243250000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243260000_01D_05M_ORB.SP3.gz",
                "igs23417.erp.Z",
                "igu23413_00.sp3.Z",
                "igu23414_00.sp3.Z",
                "igu23414_18.sp3.Z",
                "readme.txt",
            ]
        ),
    )
    archive.listing("/pub/products/2342/", apache_listing([]))
    client = garner_client(archive)
    start, end = gps(2024, 11, 13, 0), gps(2024, 11, 20, 12)

    cod = client.list_products_between(start, end, center="cod")
    assert names(cod) == [
        "COD0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        "COD0OPSFIN_20243150000_07D_01D_SUM.SUM.gz",
        "COD0OPSFIN_20243210000_01D_05M_ORB.SP3.gz",
        "COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz",
        "COD0OPSFIN_20243240000_01D_05M_ORB.SP3.gz",
        "COD0OPSFIN_20243250000_01D_05M_ORB.SP3.gz",
    ]
    assert cod[3].gps_week == 2341
    assert cod[3].url.endswith("/2341/COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz")

    igs = client.list_products_between(start, end, center="IGS")
    listed = names(igs)
    for kept in [
        "igs23417.erp.Z",
        "igu23406_18.sp3.Z",
        "igu23413_00.sp3.Z",
        "igu23414_00.sp3.Z",
        "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        "IGS0OPSULT_20243211800_02D_15M_ORB.SP3.gz",
    ]:
        assert kept in listed, kept
    for dropped in [
        "igu23414_18.sp3.Z",
        "igu23400_00.sp3.Z",
        "igr23400.sp3.Z",
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
        "IGS0OPSULT_20243150000_02D_15M_ORB.SP3.gz",
    ]:
        assert dropped not in listed, dropped
    starts = [product.file_name.start_epoch for product in igs]
    assert all(a <= b for a, b in itertools.pairwise(starts))

    assert archive.calls("/pub/products/2340/") == 1
    assert archive.calls("/pub/products/2341/") == 1
    assert archive.calls("/pub/products/2339/") == 0
    assert archive.calls("/pub/products/2342/") == 0


def test_igs_products_client_list_products_between_pads_and_skips_missing_padding_weeks(
    archive,
):
    """Rust: test_igs_products_client_list_products_between_pads_and_skips_missing_padding_weeks"""
    archive.missing("/pub/products/2339/")
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    archive.listing(
        "/pub/products/2341/",
        apache_listing(["igu23410_00.sp3.Z", "igu23410_18.sp3.Z"]),
    )
    client = garner_client(archive)

    # Saturday of week 2340: the ultra-rapid file named Sunday 00:00 of week
    # 2341 starts at Saturday 00:00 and covers the range.
    products = client.list_products_between(
        gps(2024, 11, 16, 6), gps(2024, 11, 16, 12), center="igs"
    )
    assert "igu23410_00.sp3.Z" in names(products)
    assert "igu23410_18.sp3.Z" not in names(products)
    assert archive.calls("/pub/products/2341/") == 1
    assert archive.calls("/pub/products/2339/") == 0

    # Sunday of week 2340: the earlier padding week is requested, and its 404
    # is skipped.
    products = client.list_products_between(
        gps(2024, 11, 10, 0), gps(2024, 11, 10, 6), center="COD"
    )
    assert SP3_NAME in names(products)
    assert archive.calls("/pub/products/2339/") == 1
    assert archive.calls("/pub/products/2340/") == 1


def test_igs_products_client_list_products_between_rejects_reversed_range():
    """Rust: test_igs_products_client_list_products_between_rejects_reversed_range"""
    client = bh.IGSProductsClient(source="GARNER", base_url="http://127.0.0.1:1/")
    with pytest.raises(bh.BraheError, match="before its start"):
        client.list_products_between(gps(2024, 11, 13, 0), gps(2024, 11, 12, 0))


def test_igs_products_client_missing_week_names_week_and_source(archive):
    """Rust: test_igs_products_client_missing_week_names_week_and_source"""
    archive.missing("/pub/products/2340/")
    client = garner_client(archive)
    with pytest.raises(bh.BraheError) as error:
        client.list_products(week_2340())
    message = str(error.value)
    assert "Garner" in message
    assert "2340" in message
    assert "404" in message

    with pytest.raises(bh.BraheError, match="GPS week 2340"):
        client.list_products_between(week_2340(), week_2340())
    assert archive.calls("/pub/products/2340/") == 2
    assert not (Path(client.cache_dir()) / "listings" / "garner" / "2340.txt").exists()


def test_igs_products_client_listing_is_cached(archive):
    """Rust: test_igs_products_client_listing_is_cached"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    client = garner_client(archive)

    first = client.list_files(week_2340())
    second = client.list_files(week_2340())
    products = client.list_products(week_2340())
    assert first == second
    assert products
    assert archive.calls("/pub/products/2340/") == 1

    path = Path(client.cache_dir()) / "listings" / "garner" / "2340.txt"
    assert path.read_text() == asset("garner_2340.html")

    age_file(path, 7200)
    client.list_files(week_2340())
    assert archive.calls("/pub/products/2340/") == 2

    client = garner_client(archive, cache_max_age=10000.0)
    age_file(path, 7200)
    client.list_files(week_2340())
    assert archive.calls("/pub/products/2340/") == 2


def test_igs_products_client_listing_offline_serves_cache(archive, monkeypatch):
    """Rust: test_igs_products_client_listing_offline_serves_cache"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    expected = garner_client(archive).list_files(week_2340())

    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    client = offline_garner_client()
    path = Path(client.cache_dir()) / "listings" / "garner" / "2340.txt"
    age_file(path, 7200)
    assert client.list_files(week_2340()) == expected

    with pytest.raises(bh.BraheError) as error:
        client.list_files(gps(2024, 11, 18, 0))
    assert "BRAHE_NETWORK_MODE is offline" in str(error.value)
    assert "GPS week 2341" in str(error.value)


def test_igs_products_client_listing_offline_strict_rejects_stale_cache(
    tmp_path, monkeypatch
):
    """Rust: test_igs_products_client_listing_offline_strict_rejects_stale_cache"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
    client = bh.IGSProductsClient(
        source="BKG", base_url="https://brahe-network-mode-test.invalid/products/"
    )
    path = Path(client.cache_dir()) / "listings" / "bkg" / "2340.txt"
    path.parent.mkdir(parents=True)
    path.write_text(asset("bkg_2340.html"))
    assert "readme.txt" in client.list_files(week_2340())

    age_file(path, 7200)
    with pytest.raises(bh.BraheError, match="offline-strict"):
        client.list_files(week_2340())


def test_igs_products_client_listing_rejects_non_200_status(archive):
    """Rust: test_igs_products_client_listing_rejects_non_200_status"""
    archive.serve("/pub/products/2340/", b"", status=204)
    client = garner_client(archive)

    with pytest.raises(bh.BraheError, match="unexpected HTTP status 204"):
        client.list_files(week_2340())
    assert archive.calls("/pub/products/2340/") == 1
    cached = Path(client.cache_dir()) / "listings" / "garner" / "2340.txt"
    assert not cached.exists()


def test_igs_products_client_listing_retries_server_errors(archive):
    """Rust: test_igs_products_client_listing_retries_server_errors"""
    archive.serve("/pub/products/2340/", b"", status=503)
    client = garner_client(archive, max_retries=1)
    with pytest.raises(bh.BraheError) as error:
        client.list_files(week_2340())
    assert "503" in str(error.value)
    assert "Garner listing for GPS week 2340" in str(error.value)
    assert archive.calls("/pub/products/2340/") == 2


def test_igs_listed_file_rejects_scheme_links(archive):
    """Rust: test_igs_listed_file_rejects_scheme_links"""
    archive.listing(
        "/pub/products/2340/",
        '<a href="mailto:x@example.com">m</a><a href="javascript:void(0)">j</a>'
        '<a href="readme.txt">readme.txt</a>',
    )
    assert garner_client(archive).list_files(week_2340()) == ["readme.txt"]


def test_igs_listings_drop_traversal_names(archive):
    """Rust: test_igs_listings_drop_traversal_names"""
    archive.listing(
        "/pub/products/2340/",
        apache_listing(
            [
                f"../{SP3_NAME}",
                "..%2F..%2Fetc",
                f"C:{SP3_NAME}",
                SP3_NAME,
            ]
        ),
    )
    client = garner_client(archive)
    assert client.list_files(week_2340()) == ["..%2F..%2Fetc", SP3_NAME]
    assert names(client.list_products(week_2340())) == [SP3_NAME]

    archive.listing(
        "/archive/2340/*",
        f"../../{SP3_NAME} 10\n..\\{SP3_NAME} 10\nC:{SP3_NAME} 10\n.. 1\n{SP3_NAME} 12\n",
    )
    cddis = bh.IGSProductsClient(source="CDDIS", base_url=archive.url("/archive/"))
    assert cddis.list_files(week_2340()) == [SP3_NAME]
    assert [p.size for p in cddis.list_products(week_2340())] == [12]


def test_igs_products_client_list_products_between_offline_repeats_online_query(
    archive, monkeypatch
):
    """Rust: test_igs_products_client_list_products_between_offline_repeats_online_query"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    archive.missing("/pub/products/2341/")
    start, end = gps(2024, 11, 16, 6), gps(2024, 11, 16, 12)
    expected = garner_client(archive).list_products_between(start, end, center="COD")
    assert "COD0OPSFIN_20243210000_01D_05M_ORB.SP3.gz" in names(expected)
    assert archive.calls("/pub/products/2341/") == 1

    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    offline = offline_garner_client().list_products_between(start, end, center="COD")
    assert names(offline) == names(expected)


def test_igs_products_client_list_products_between_offline_skips_uncached_padding_week(
    archive, monkeypatch
):
    """Rust: test_igs_products_client_list_products_between_offline_skips_uncached_padding_week"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    garner_client(archive).list_products(week_2340())

    start, end = gps(2024, 11, 11, 0), gps(2024, 11, 12, 0)
    client = offline_garner_client()
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    products = client.list_products_between(start, end, center="COD")
    assert "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz" in names(products)

    # A stale cached padding week is skipped under offline-strict, while the
    # fresh in-range week is served.
    padding = Path(client.cache_dir()) / "listings" / "garner" / "2339.txt"
    padding.write_text(apache_listing(["COD0OPSFIN_20243140000_01D_05M_ORB.SP3.gz"]))
    age_file(padding, 7200)
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
    strict = client.list_products_between(start, end, center="COD")
    assert names(strict) == names(products)


def test_igs_products_client_list_products_between_offline_uncached_range_week_errors(
    archive, monkeypatch
):
    """Rust: test_igs_products_client_list_products_between_offline_uncached_range_week_errors"""
    archive.listing("/pub/products/2340/", asset("garner_2340.html"))
    garner_client(archive).list_files(week_2340())

    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    with pytest.raises(bh.BraheError) as error:
        offline_garner_client().list_products_between(
            gps(2024, 11, 13, 0), gps(2024, 11, 20, 0)
        )
    assert "BRAHE_NETWORK_MODE is offline" in str(error.value)
    assert "GPS week 2341" in str(error.value)


# Downloads


def test_igs_products_client_download_caches_file(archive):
    """Rust: test_igs_products_client_download_caches_file"""
    body = gzip.compress(SP3_TEXT.encode())
    archive.serve(f"/pub/products/2340/{SP3_NAME}", body)
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())

    path = client.download(product)
    assert isinstance(path, str)
    assert Path(path) == Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME
    assert Path(path).read_bytes() == body
    assert client.download(product) == path
    assert archive.calls(f"/pub/products/2340/{SP3_NAME}") == 1


def test_igs_products_client_download_offline_serves_cache_and_rejects_uncached(
    archive, monkeypatch
):
    """Rust: test_igs_products_client_download_offline_serves_cache_and_rejects_uncached"""
    uncached = "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz"
    archive.serve(f"/pub/products/2340/{SP3_NAME}", gzip.compress(SP3_TEXT.encode()))
    archive.listing("/pub/products/2340/", apache_listing([SP3_NAME, uncached]))
    path = garner_client(archive).download(
        garner_client(archive).list_products(week_2340())[0]
    )

    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    client = offline_garner_client()
    listed = {product.name: product for product in client.list_products(week_2340())}
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline-strict")
    assert client.download(listed[SP3_NAME]) == path
    with pytest.raises(bh.BraheError) as error:
        client.download(listed[uncached])
    assert "BRAHE_NETWORK_MODE is offline-strict" in str(error.value)
    assert uncached in str(error.value)


def test_igs_products_client_download_missing_product_names_file_and_source(archive):
    """Rust: test_igs_products_client_download_missing_product_names_file_and_source"""
    archive.missing(f"/pub/products/2340/{SP3_NAME}")
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    with pytest.raises(bh.BraheError) as error:
        client.download(product)
    message = str(error.value)
    assert "Garner" in message
    assert SP3_NAME in message
    assert "404" in message
    assert archive.calls(f"/pub/products/2340/{SP3_NAME}") == 1
    assert not (Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME).exists()


def test_igs_products_client_download_rejects_body_not_matching_compression(archive):
    """Rust: test_igs_products_client_download_rejects_body_not_matching_compression"""
    archive.serve(f"/pub/products/2340/{SP3_NAME}", b"<html>error</html>")
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    with pytest.raises(bh.BraheError, match="is not gzip data"):
        client.download(product)
    assert not (Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME).exists()


def test_igs_products_client_download_rejects_truncated_gzip_without_caching(archive):
    """Rust: test_igs_products_client_download_rejects_truncated_gzip_without_caching"""
    body = gzip.compress(SP3_TEXT.encode())
    path = f"/pub/products/2340/{SP3_NAME}"
    archive.serve(path, body[:-8])
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    cached = Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME

    with pytest.raises(bh.BraheError) as error:
        client.download(product)
    assert "is not valid gzip data" in str(error.value)
    assert SP3_NAME in str(error.value)
    assert not cached.exists()

    archive.serve(path, body)
    assert Path(client.download(product)) == cached
    assert cached.read_bytes() == body
    assert archive.calls(path) == 2


def test_igs_products_client_redirects_are_checked_against_network_mode(
    archive, monkeypatch
):
    """Rust: test_igs_products_client_redirects_are_checked_against_network_mode"""
    body = gzip.compress(SP3_TEXT.encode())
    name = "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz"
    archive.listing("/pub/products/2340/", apache_listing([SP3_NAME, name]))
    archive.redirect(f"/pub/products/2340/{SP3_NAME}", f"/mirror/{SP3_NAME}")
    archive.serve(f"/mirror/{SP3_NAME}", body)
    archive.redirect(
        f"/pub/products/2340/{name}", f"http://brahe-redirect-test.invalid/{name}"
    )
    client = garner_client(archive)
    listed = {product.name: product for product in client.list_products(week_2340())}
    assert Path(client.download(listed[SP3_NAME])).read_bytes() == body
    assert archive.calls(f"/mirror/{SP3_NAME}") == 1

    external = listed[name]
    for mode in ("offline", "offline-strict"):
        monkeypatch.setenv("BRAHE_NETWORK_MODE", mode)
        with pytest.raises(bh.BraheError) as error:
            client.download(external)
        assert f"BRAHE_NETWORK_MODE is {mode}" in str(error.value)
        assert "brahe-redirect-test.invalid" in str(error.value)
    assert archive.calls(f"/pub/products/2340/{name}") == 2


def test_igs_products_client_save_gzip_decompresses(archive, tmp_path):
    """Rust: test_igs_products_client_save_gzip_decompresses"""
    out = tmp_path / "out"
    out.mkdir()
    archive.serve(f"/pub/products/2340/{SP3_NAME}", gzip.compress(SP3_TEXT.encode()))
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    cached = Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME

    saved = client.save(product, str(out), True, True)
    assert Path(saved) == out / "COD0OPSFIN_20243150000_01D_05M_ORB.SP3"
    assert Path(saved).read_text() == SP3_TEXT
    assert cached.is_file()

    custom = out / "nested" / "custom.sp3"
    saved = client.save(product, custom, keep_cached=False, decompress=True)
    assert Path(saved) == custom
    assert custom.read_text() == SP3_TEXT
    assert not cached.exists()
    assert archive.calls(f"/pub/products/2340/{SP3_NAME}") == 1


def test_igs_products_client_save_keeps_published_file(archive, tmp_path):
    """Rust: test_igs_products_client_save_keeps_published_file"""
    out = tmp_path / "out"
    out.mkdir()
    published = gzip.compress(SP3_TEXT.encode())
    archive.serve(f"/pub/products/2340/{SP3_NAME}", published)
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    cached = Path(client.cache_dir()) / "garner" / "2340" / SP3_NAME

    saved = client.save(product, out, keep_cached=True)
    assert Path(saved) == out / SP3_NAME
    assert Path(saved).read_bytes() == published
    assert cached.is_file()

    custom = out / "nested" / "custom.sp3.gz"
    saved = client.save(product, custom)
    assert Path(saved) == custom
    assert custom.read_bytes() == published
    assert not cached.exists()
    assert archive.calls(f"/pub/products/2340/{SP3_NAME}") == 1


def test_igs_products_client_save_unix_compress_decompresses(archive, tmp_path):
    """Rust: test_igs_products_client_save_unix_compress_decompresses"""
    archive.serve(f"/pub/products/2340/{SHORT_Z_NAME}", SHORT_Z_BYTES)
    client = garner_client(archive)
    product = listed_product(archive, client, SHORT_Z_NAME, 2340, week_2340())

    saved = client.save(product, tmp_path, decompress=True)
    assert Path(saved) == tmp_path / "igr23400.sp3"
    assert Path(saved).read_bytes() == b"#2"
    assert not (Path(client.cache_dir()) / "garner" / "2340" / SHORT_Z_NAME).exists()


def test_igs_products_client_save_into_cache_errors_before_download(archive):
    """Rust: test_igs_products_client_save_into_cache_errors_before_download"""
    archive.serve(f"/pub/products/2340/{SP3_NAME}", gzip.compress(SP3_TEXT.encode()))
    client = garner_client(archive)
    product = listed_product(archive, client, SP3_NAME, 2340, week_2340())
    cache = Path(client.cache_dir())
    for destination in [cache, cache / "saved" / "out.sp3"]:
        with pytest.raises(bh.BraheError, match="inside the IGS cache directory"):
            client.save(product, destination)
    assert archive.hits == []


def test_igs_products_client_save_unix_compress_keep_cached(archive, tmp_path):
    """Rust: test_igs_products_client_save_unix_compress_keep_cached"""
    archive.serve(f"/pub/products/2340/{SHORT_Z_NAME}", SHORT_Z_BYTES)
    client = garner_client(archive)
    product = listed_product(archive, client, SHORT_Z_NAME, 2340, week_2340())

    saved = client.save(product, tmp_path, keep_cached=True, decompress=True)
    assert Path(saved) == tmp_path / "igr23400.sp3"
    assert Path(saved).read_bytes() == b"#2"
    cached = Path(client.cache_dir()) / "garner" / "2340" / SHORT_Z_NAME
    assert cached.read_bytes() == SHORT_Z_BYTES


def test_igs_products_client_listing_rejects_non_utf8_body(archive):
    """Rust: test_igs_products_client_listing_rejects_non_utf8_body"""
    archive.serve("/pub/products/2340/", b"\xff\xfe\x00")
    client = garner_client(archive)

    with pytest.raises(bh.BraheError, match="is not UTF-8 text"):
        client.list_files(week_2340())
    assert not (Path(client.cache_dir()) / "listings" / "garner" / "2340.txt").exists()


def test_igs_products_client_cddis_html_product_is_rejected(archive):
    """Rust: test_igs_products_client_cddis_html_product_is_rejected"""
    archive.listing("/archive/gnss/products/2340/*", "igr23400.sp3 1024\n")
    archive.serve(
        "/archive/gnss/products/2340/igr23400.sp3", "<html>Earthdata Login</html>"
    )
    client = bh.IGSProductsClient(
        source="CDDIS", base_url=archive.url("/archive/gnss/products/")
    )
    product = client.list_products(week_2340())[0]

    with pytest.raises(bh.BraheError, match="is an HTML page") as error:
        client.download(product)
    assert "check the client's credentials" in str(error.value)
    assert not (Path(client.cache_dir()) / "cddis" / "2340" / "igr23400.sp3").exists()


# SP3 downloads


def serve_cod_week_2340(archive, days):
    """Serve a week 2340 listing of COD final daily orbits at 05M and 15M for
    each 2024 day of year in days, plus products of other solutions,
    contents, spans, and centers, each with the SP3 fixture as its body.
    Returns the listed file names."""
    listed = [cod_daily(doy, sampling) for doy in days for sampling in ("05M", "15M")]
    listed += [
        "COD0OPSRAP_20243160000_01D_15M_ORB.SP3.gz",
        "COD0OPSULT_20243160000_02D_05M_ORB.SP3.gz",
        "COD0OPSFIN_20243160000_01D_30S_CLK.CLK.gz",
        "COD0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        "ESA0OPSFIN_20243160000_01D_15M_ORB.SP3.gz",
    ]
    archive.listing("/pub/products/2340/", apache_listing(listed))
    for name in listed:
        archive.serve(f"/pub/products/2340/{name}", SP3_FIXTURE.read_bytes())
    return listed


def assert_downloaded(archive, listed, expected):
    """Each listed file was requested once when in expected and never otherwise."""
    for name in listed:
        calls = archive.calls(f"/pub/products/2340/{name}")
        assert calls == (1 if name in expected else 0), name


def downloaded_files(archive):
    """Names of the files requested, sorted, excluding listings."""
    return sorted(
        urlsplit(hit).path.rsplit("/", 1)[1]
        for hit in archive.hits
        if not urlsplit(hit).path.endswith("/")
    )


def test_igs_products_client_download_sp3_parses_file(archive):
    """Rust: test_igs_products_client_download_sp3_parses_file"""
    name = "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz"
    archive.serve(f"/pub/products/2340/{name}", SP3_FIXTURE.read_bytes())
    client = garner_client(archive)
    product = listed_product(archive, client, name, 2340, week_2340())

    sp3 = client.download_sp3(product)
    assert same_sp3(sp3, bh.SP3.from_file(SP3_FIXTURE))
    assert archive.calls(f"/pub/products/2340/{name}") == 1


def test_igs_products_client_download_sp3_unix_compress(archive):
    """Rust: test_igs_products_client_download_sp3_unix_compress"""
    name = "gbm18432.sp3.Z"
    client = garner_client(archive)
    product = listed_product(archive, client, name, 1843, gps(2015, 5, 5, 0))
    archive.serve(f"/pub/products/1843/{name}", UNIX_SP3_FIXTURE.read_bytes())

    sp3 = client.download_sp3(product)
    assert same_sp3(sp3, bh.SP3.from_file(UNIX_SP3_FIXTURE))
    assert sp3.num_epochs == 288
    assert archive.calls(f"/pub/products/1843/{name}") == 1


def test_igs_products_client_download_sp3_rejects_before_download(archive):
    """Rust: test_igs_products_client_download_sp3_rejects_before_download"""
    client = garner_client(archive)
    clock = listed_product(
        archive, client, "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz", 2340, week_2340()
    )

    with pytest.raises(bh.BraheError, match="not an SP3"):
        client.download_sp3(clock)
    assert archive.hits == []


def test_igs_products_client_download_sp3_range_pads_with_coarsest_sampling(archive):
    """Rust: test_igs_products_client_download_sp3_range_pads_with_coarsest_sampling"""
    listed = serve_cod_week_2340(archive, [315, 316, 317, 318])
    client = garner_client(archive)
    start, end = gps(2024, 11, 11, 0), gps(2024, 11, 12, 0)
    fixture = bh.SP3.from_file(SP3_FIXTURE)

    sp3s = client.download_sp3_range(start, end, "cod", bh.IGSSolutionType.FINAL)
    expected = [cod_daily(doy, "15M") for doy in range(315, 319)]
    assert len(sp3s) == 4
    assert all(same_sp3(sp3, fixture) for sp3 in sp3s)
    assert downloaded_files(archive) == expected
    assert_downloaded(archive, listed, expected)

    sp3s = client.download_sp3_range(
        start, end, bh.IGSAnalysisCenter.COD, "FIN", sampling=300.0
    )
    assert len(sp3s) == 4
    expected_fine = [cod_daily(doy, "05M") for doy in range(315, 319)]
    assert downloaded_files(archive) == sorted(expected + expected_fine)
    assert_downloaded(archive, listed, expected + expected_fine)


def test_igs_products_client_download_sp3_range_without_padding(archive):
    """Rust: test_igs_products_client_download_sp3_range_without_padding"""
    listed = serve_cod_week_2340(archive, [315, 316, 317, 318])
    client = garner_client(archive)

    sp3s = client.download_sp3_range(
        gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "COD", "FIN", pad=False
    )
    assert len(sp3s) == 2
    expected = [cod_daily(316, "15M"), cod_daily(317, "15M")]
    assert downloaded_files(archive) == expected
    assert_downloaded(archive, listed, expected)


def test_igs_products_client_download_sp3_range_skips_missing_padding(archive):
    """Rust: test_igs_products_client_download_sp3_range_skips_missing_padding"""
    listed = serve_cod_week_2340(archive, [316, 317, 318])
    archive.missing("/pub/products/2339/")
    client = garner_client(archive)

    # Day 315, the padding day before Monday 2024-11-11, is missing.
    sp3s = client.download_sp3_range(
        gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "COD", bh.IGSSolutionType.FINAL
    )
    assert len(sp3s) == 3
    assert_downloaded(
        archive, listed, [cod_daily(doy, "15M") for doy in range(316, 319)]
    )
    assert archive.calls("/pub/products/2339/") == 0

    archive.fresh_cache()
    archive.reset()
    sp3s = client.download_sp3_range(
        gps(2024, 11, 11, 6), gps(2024, 11, 11, 12), "COD", "FIN"
    )
    assert len(sp3s) == 2
    assert downloaded_files(archive) == [cod_daily(316, "15M"), cod_daily(317, "15M")]

    # Inside the span, the missing day 315 is an error. The padding day
    # before it is in week 2339, whose 404 is skipped.
    archive.reset()
    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(
            gps(2024, 11, 10, 6), gps(2024, 11, 11, 12), "COD", "FIN"
        )
    assert "for 2024-11-10" in str(error.value)
    assert archive.calls("/pub/products/2339/") == 1
    assert downloaded_files(archive) == []


def test_igs_products_client_download_sp3_range_skips_padding_file_not_found(archive):
    """Rust: test_igs_products_client_download_sp3_range_skips_padding_file_not_found"""
    listed = serve_cod_week_2340(archive, [315, 316, 317, 318, 319])
    archive.missing(f"/pub/products/2340/{cod_daily(318, '15M')}")
    client = garner_client(archive)

    # Day 318, the padding day after the span, is listed but answers 404.
    sp3s = client.download_sp3_range(
        gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "COD", "FIN"
    )
    assert len(sp3s) == 3
    expected = [cod_daily(doy, "15M") for doy in range(315, 319)]
    assert downloaded_files(archive) == expected
    assert_downloaded(archive, listed, expected)
    cached = Path(client.cache_dir()) / "garner" / "2340" / cod_daily(318, "15M")
    assert not cached.exists()

    # Inside the span the same 404 is an error naming the file.
    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(
            gps(2024, 11, 12, 0), gps(2024, 11, 13, 0), "COD", "FIN"
        )
    message = str(error.value)
    assert "not found" in message
    assert "Garner" in message
    assert cod_daily(318, "15M") in message


def test_igs_products_client_download_sp3_range_padding_server_error_fails(archive):
    """Rust: test_igs_products_client_download_sp3_range_padding_server_error_fails"""
    serve_cod_week_2340(archive, [315, 316, 317, 318])
    padding = f"/pub/products/2340/{cod_daily(318, '15M')}"
    archive.serve(padding, b"", status=503)
    client = garner_client(archive, max_retries=1)

    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(
            gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "COD", "FIN"
        )
    assert "503" in str(error.value)
    assert cod_daily(318, "15M") in str(error.value)
    assert archive.calls(padding) == 2


def test_igs_products_client_download_sp3_range_skips_missing_padding_week(archive):
    """Rust: test_igs_products_client_download_sp3_range_skips_missing_padding_week"""
    serve_cod_week_2340(archive, [315, 316])
    archive.missing("/pub/products/2339/")
    client = garner_client(archive)

    sp3s = client.download_sp3_range(
        gps(2024, 11, 10, 6), gps(2024, 11, 10, 12), "COD", "FIN"
    )
    assert len(sp3s) == 2
    assert downloaded_files(archive) == [cod_daily(315, "15M"), cod_daily(316, "15M")]
    assert archive.calls("/pub/products/2339/") == 1


def test_igs_products_client_download_sp3_range_missing_day_names_day_and_source(
    archive,
):
    """Rust: test_igs_products_client_download_sp3_range_missing_day_names_day_and_source"""
    listed = serve_cod_week_2340(archive, [315, 316, 318])
    client = garner_client(archive)
    start, end = gps(2024, 11, 11, 0), gps(2024, 11, 12, 0)

    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(start, end, "COD", "FIN")
    message = str(error.value)
    assert "IGS Garner has no daily COD FIN SP3 orbit file for 2024-11-12" in message
    assert "GPS week 2340, day 2" in message
    assert "/pub/products/2340/ lists none" in message

    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(start, end, "COD", "FIN", sampling=30.0)
    assert "SP3 orbit file with 30 s sampling for 2024-11-11" in str(error.value)
    assert_downloaded(archive, listed, [])


def test_igs_products_client_download_sp3_range_rejects_ultra_rapid_before_request(
    archive,
):
    """Rust: test_igs_products_client_download_sp3_range_rejects_ultra_rapid_before_request"""
    client = garner_client(archive)

    for solution in (bh.IGSSolutionType.ULTRA_RAPID, "ult"):
        with pytest.raises(bh.BraheError, match="ultra-rapid"):
            client.download_sp3_range(
                gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "IGS", solution
            )
    with pytest.raises(bh.BraheError, match="before its start"):
        client.download_sp3_range(
            gps(2024, 11, 12, 0), gps(2024, 11, 11, 0), "IGS", "FIN"
        )
    with pytest.raises(TypeError):
        client.download_sp3_range(gps(2024, 11, 11, 0), gps(2024, 11, 12, 0), "IGS", 1)
    assert archive.hits == []


def test_igs_products_client_download_sp3_range_unix_compress(archive):
    """Rust: test_igs_products_client_download_sp3_range_unix_compress"""
    listed = ["cod21000.sp3.Z", "cod21001.sp3.gz", "cod21002.sp3.Z", "cod21003.clk.Z"]
    archive.listing("/pub/products/2100/", apache_listing(listed))
    for name in listed:
        body = SP3_FIXTURE if name.endswith(".gz") else UNIX_SP3_FIXTURE
        archive.serve(f"/pub/products/2100/{name}", body.read_bytes())
    client = garner_client(archive)

    # Sunday 2020-04-05 is day 0 of week 2100. Unix-compressed files are
    # chosen like any other, inside the span and as padding.
    sp3s = client.download_sp3_range(
        gps(2020, 4, 6, 6), gps(2020, 4, 6, 12), "COD", "FIN"
    )
    assert len(sp3s) == 3
    assert downloaded_files(archive) == listed[:3]

    archive.reset()
    sp3s = client.download_sp3_range(
        gps(2020, 4, 5, 6), gps(2020, 4, 6, 6), "cod", "FIN", pad=False
    )
    assert len(sp3s) == 2
    assert same_sp3(sp3s[0], bh.SP3.from_file(UNIX_SP3_FIXTURE))
    assert same_sp3(sp3s[1], bh.SP3.from_file(SP3_FIXTURE))

    # A missing short-name day names the day.
    with pytest.raises(
        bh.BraheError, match="no daily COD RAP SP3 orbit file for 2020-04-09"
    ):
        client.download_sp3_range(
            gps(2020, 4, 9, 6), gps(2020, 4, 9, 12), "COD", bh.IGSSolutionType.RAPID
        )


def test_igs_products_client_retry_waits_for_retry_after(archive):
    """Rust: test_igs_products_client_retry_waits_for_retry_after"""
    archive.serve("/pub/products/2340/", b"", status=503, headers={"Retry-After": "1"})
    client = garner_client(archive, max_retries=1)

    started = time.monotonic()
    with pytest.raises(bh.BraheError, match="503"):
        client.list_files(week_2340())
    # The backoff before a first retry is at most 0.5 s.
    assert time.monotonic() - started >= 1.0
    assert archive.calls("/pub/products/2340/") == 2


def test_igs_products_client_rate_limit_counts_requests(archive):
    """Rust: test_igs_products_client_rate_limit_counts_requests"""
    archive.listing("/pub/products/2340/", apache_listing([SP3_NAME]))
    client = garner_client(
        archive, rate_limit=bh.RateLimitConfig(max_per_minute=1, max_per_hour=100)
    )
    assert client.list_files(week_2340()) == [SP3_NAME]


def test_igs_products_client_sends_user_agent(archive):
    """Rust: test_igs_products_client_sends_user_agent"""
    archive.listing("/pub/products/2340/", apache_listing([SP3_NAME]))
    garner_client(archive).list_files(week_2340())
    assert archive.user_agents == [f"brahe/{bh.__version__}"]


@pytest.mark.parametrize("concurrency", [1, 3])
def test_igs_products_client_download_sp3_range_concurrency_keeps_time_order(
    archive, concurrency
):
    """Rust: test_igs_products_client_download_sp3_range_concurrency_keeps_time_order"""
    listed = ["cod21000.sp3.Z", "cod21001.sp3.gz", "cod21002.sp3.Z"]
    archive.listing("/pub/products/2100/", apache_listing(listed))
    for name in listed:
        body = SP3_FIXTURE if name.endswith(".gz") else UNIX_SP3_FIXTURE
        archive.serve(f"/pub/products/2100/{name}", body.read_bytes())

    sp3s = garner_client(archive).download_sp3_range(
        gps(2020, 4, 6, 6), gps(2020, 4, 6, 12), "COD", "FIN", concurrency=concurrency
    )
    unix, gzipped = bh.SP3.from_file(UNIX_SP3_FIXTURE), bh.SP3.from_file(SP3_FIXTURE)
    assert len(sp3s) == 3
    assert same_sp3(sp3s[0], unix)
    assert same_sp3(sp3s[1], gzipped)
    assert same_sp3(sp3s[2], unix)
    assert downloaded_files(archive) == listed


def test_igs_products_client_download_sp3_range_rejects_zero_concurrency(archive):
    """Rust: test_igs_products_client_download_sp3_range_rejects_zero_concurrency"""
    with pytest.raises(bh.BraheError, match="concurrency >= 1"):
        garner_client(archive).download_sp3_range(
            gps(2020, 4, 6, 6), gps(2020, 4, 6, 12), "COD", "FIN", concurrency=0
        )
    assert archive.hits == []


def test_igs_products_client_download_sp3_range_cddis(archive):
    """Rust: test_igs_products_client_download_sp3_range_cddis"""
    archive.listing("/archive/2340/*", asset("cddis_2340.txt"))
    expected = [cod_daily(doy, "05M") for doy in (315, 316)]
    for name in expected:
        archive.serve(f"/archive/2340/{name}", SP3_FIXTURE.read_bytes())
    client = bh.IGSProductsClient("CDDIS", base_url=archive.url("/archive/"))

    sp3s = client.download_sp3_range(
        gps(2024, 11, 10, 0), gps(2024, 11, 11, 0), "COD", "FIN", pad=False
    )
    assert len(sp3s) == 2
    assert sorted(urlsplit(hit).path.rsplit("/", 1)[1] for hit in archive.hits) == [
        "*",
        *expected,
    ]


def test_igs_products_client_download_sp3_range_missing_week_names_day_and_listing(
    archive,
):
    """Rust: test_igs_products_client_download_sp3_range_missing_week_names_day_and_listing"""
    archive.missing("/pub/products/2341/")
    client = garner_client(archive)

    # Monday 2024-11-18 is day of year 323, in GPS week 2341.
    with pytest.raises(bh.BraheError) as error:
        client.download_sp3_range(
            gps(2024, 11, 18, 6), gps(2024, 11, 18, 12), "COD", "FIN", pad=False
        )
    message = str(error.value)
    assert "IGS Garner has no daily COD FIN SP3 orbit file for 2024-11-18" in message
    assert "GPS week 2341, day 1" in message
    assert "answered 404" in message
    assert archive.calls("/pub/products/2341/") == 1


def test_igs_products_client_download_sp3_parse_errors_name_product(archive):
    """Rust: test_igs_products_client_download_sp3_parse_errors_name_product"""
    name = cod_daily(316, "15M")
    archive.listing("/pub/products/2340/", apache_listing([name]))
    archive.serve(f"/pub/products/2340/{name}", gzip.compress(b"not an SP3 file\n"))
    client = garner_client(archive)
    product = next(p for p in client.list_products(week_2340()) if p.name == name)

    with pytest.raises(bh.BraheError, match=f"IGS Garner product {name}"):
        client.download_sp3(product)
    with pytest.raises(bh.BraheError, match=f"IGS Garner product {name}"):
        client.download_sp3_range(
            gps(2024, 11, 11, 6), gps(2024, 11, 11, 12), "COD", "FIN", pad=False
        )


def test_igs_products_client_download_sp3_range_crosses_long_name_week(archive):
    """Rust: test_igs_products_client_download_sp3_range_crosses_long_name_week"""
    plain = gzip.decompress(SP3_FIXTURE.read_bytes())
    short = ["cod22375.sp3", "cod22376.sp3"]
    long = [
        "COD0OPSFIN_20223310000_01D_05M_ORB.SP3.gz",
        "COD0OPSFIN_20223320000_01D_05M_ORB.SP3.gz",
    ]
    # Each week also lists a file in the other convention, which is not used.
    archive.listing(
        "/pub/products/2237/",
        apache_listing([*short, "COD0OPSFIN_20223300000_01D_05M_ORB.SP3.gz"]),
    )
    archive.listing("/pub/products/2238/", apache_listing([*long, "cod22380.sp3.Z"]))
    for name in short:
        archive.serve(f"/pub/products/2237/{name}", plain)
    for name in long:
        archive.serve(f"/pub/products/2238/{name}", SP3_FIXTURE.read_bytes())
    client = garner_client(archive)

    # Saturday 2022-11-26 is the last day of week 2237.
    sp3s = client.download_sp3_range(
        gps(2022, 11, 26, 12), gps(2022, 11, 27, 12), "COD", bh.IGSSolutionType.FINAL
    )
    fixture = bh.SP3.from_file(SP3_FIXTURE)
    assert len(sp3s) == 4
    assert all(same_sp3(sp3, fixture) for sp3 in sp3s)
    assert downloaded_files(archive) == sorted([*short, *long])


INTEGRATION_FINAL_ORBIT = "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz"


@pytest.fixture
def live_cache(tmp_path, monkeypatch):
    """A new, empty cache with network access enabled."""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.delenv("BRAHE_NETWORK_MODE", raising=False)


def assert_integration_final_orbit(client):
    products = client.list_products(
        week_2340(),
        center="IGS",
        solution=bh.IGSSolutionType.FINAL,
        content=bh.IGSContentType.ORBIT,
        format=bh.IGSFileFormat.SP3,
    )
    product = next((p for p in products if p.name == INTEGRATION_FINAL_ORBIT), None)
    assert product is not None, f"{INTEGRATION_FINAL_ORBIT} not listed"
    sp3 = client.download_sp3(product)
    assert sp3.num_epochs == 96
    assert len(sp3.satellite_ids) > 0


@pytest.mark.integration
def test_integration_garner_listing_and_download(live_cache):
    """Rust: test_integration_garner_listing_and_download"""
    client = bh.IGSProductsClient(source="GARNER")
    assert INTEGRATION_FINAL_ORBIT in client.list_files(week_2340())
    assert_integration_final_orbit(client)


@pytest.mark.integration
def test_integration_garner_unix_compress_download(live_cache):
    """Rust: test_integration_garner_unix_compress_download"""
    client = bh.IGSProductsClient(source="GARNER")
    products = client.list_products(gps(2020, 4, 8, 12), center="IGS")
    product = next((p for p in products if p.name == "igs21000.sp3.Z"), None)
    assert product is not None, "igs21000.sp3.Z not listed"
    assert Path(client.download(product)).read_bytes()[:2] == b"\x1f\x9d"
    sp3 = client.download_sp3(product)
    assert sp3.num_epochs == 96
    assert sp3.num_satellites >= 30


@pytest.mark.integration
def test_integration_bkg_listing_and_download(live_cache):
    """Rust: test_integration_bkg_listing_and_download"""
    client = bh.IGSProductsClient(source="BKG")
    assert INTEGRATION_FINAL_ORBIT in client.list_files(week_2340())
    assert_integration_final_orbit(client)


@pytest.mark.integration
def test_integration_cddis_listing_and_download(live_cache):
    """Rust: test_integration_cddis_listing_and_download"""
    username = os.environ.get("CDDIS_EARTHDATA_USERNAME", "")
    password = os.environ.get("CDDIS_EARTHDATA_PASSWORD", "")
    if not username or not password:
        pytest.skip("CDDIS credentials not set")
    client = bh.IGSProductsClient(
        source="CDDIS", cddis_username=username, cddis_password=password
    )
    assert INTEGRATION_FINAL_ORBIT in client.list_files(week_2340())
    assert_integration_final_orbit(client)
