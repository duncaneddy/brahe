import os
import pathlib
import shutil
import pytest
import brahe


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_ASSETS = PACKAGE_ROOT / "test_assets"
TEST_DATA = PACKAGE_ROOT / "test_data"


@pytest.fixture(scope="module", autouse=True)
def _sw_provider():
    """Initialize space weather from local test asset for propagator tests.

    ForceModelConfig.default() includes NRLMSISE-00 atmospheric drag which
    requires space weather data to be initialized.
    """
    sw = brahe.FileSpaceWeatherProvider.from_file(
        str(TEST_ASSETS / "sw19571001.txt"), "Hold"
    )
    brahe.set_global_space_weather_provider(sw)


# A real full IERS download is 3,686,988 bytes; the packaged n<=30 test
# fixture is ~492 KB. Anything at/above this threshold is treated as a
# genuine full download and never touched or removed.
_FES2004_FULL_DOWNLOAD_MIN_BYTES = 3_000_000


@pytest.fixture(scope="session")
def _fes2004_cache_setup():
    """Seed the FES2004 ocean tide coefficients into the tides cache.

    ForceModelConfig.high_fidelity() enables ocean tides, whose model is loaded
    from ``$BRAHE_CACHE/tides/fes2004_Cnm-Snm.dat`` at propagator construction.
    Pre-populating the cache with the packaged degree/order <= 30 test fixture
    avoids hitting the IERS server during tests (mirrors ``naif_cache_setup``).

    Only referenced by tests that actually construct a high-fidelity
    propagator (not merely a ``ForceModelConfig``), since it mutates the
    user's real tides cache. If the canonical file already exists and looks
    like a full download (see ``_FES2004_FULL_DOWNLOAD_MIN_BYTES``), it is
    left alone. Otherwise (absent, or a truncated leftover from an
    interrupted run) the test fixture is copied in and removed again during
    teardown so a truncated file never lingers under the production name.
    """
    brahe_cache = os.environ.get("BRAHE_CACHE")
    if brahe_cache:
        cache_dir = pathlib.Path(brahe_cache) / "tides"
    else:
        cache_dir = pathlib.Path.home() / ".cache" / "brahe" / "tides"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "fes2004_Cnm-Snm.dat"

    created_or_replaced = (
        not cached.exists() or cached.stat().st_size < _FES2004_FULL_DOWNLOAD_MIN_BYTES
    )
    if created_or_replaced:
        shutil.copy2(TEST_DATA / "fes2004_Cnm-Snm_n30.dat", cached)

    yield str(cached)

    if created_or_replaced:
        cached.unlink(missing_ok=True)
