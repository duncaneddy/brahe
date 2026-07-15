import os
import pathlib
import shutil
import pytest
import numpy as np
import brahe

# Configure matplotlib to use non-GUI backend for testing
import matplotlib

matplotlib.use("Agg")

# Testing Paths
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ASSETS = PACKAGE_ROOT / "test_assets"
TEST_DATA = PACKAGE_ROOT / "test_data"


@pytest.fixture(scope="session")
def iau2000_c04_20_filepath():
    filepath = TEST_ASSETS / "EOP_C04_one_file_1962-now.txt"
    yield str(filepath)


@pytest.fixture(scope="session")
def iau2000_standard_filepath():
    filepath = TEST_ASSETS / "finals.all.iau2000.txt"
    yield str(filepath)


@pytest.fixture(scope="session")
def brahe_original_eop_filepath():
    filepath = TEST_ASSETS / "brahe_original_eop_file.txt"
    yield str(filepath)


@pytest.fixture(scope="session")
def de440s_kernel_filepath():
    """Path to de440s.bsp NAIF kernel test asset."""
    filepath = TEST_ASSETS / "de440s.bsp"
    yield str(filepath)


@pytest.fixture(scope="session")
def sw_test_filepath():
    """Path to space weather test data file."""
    filepath = TEST_ASSETS / "sw19571001.txt"
    yield str(filepath)


@pytest.fixture(scope="session")
def naif_cache_setup(de440s_kernel_filepath):
    """Copy de440s.bsp from test_assets to NAIF cache directory for testing.

    This avoids hitting NAIF servers during tests by pre-populating the cache
    with the test asset.
    """
    import shutil
    import os

    # Construct NAIF cache directory path
    # (mirrors brahe::utils::cache::get_naif_cache_dir())
    brahe_cache = os.environ.get("BRAHE_CACHE")
    if brahe_cache:
        cache_dir = pathlib.Path(brahe_cache) / "naif"
    else:
        cache_dir = pathlib.Path.home() / ".cache" / "brahe" / "naif"

    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_kernel_path = cache_dir / "de440s.bsp"

    # Copy test asset to cache if not already there
    if not cache_kernel_path.exists():
        shutil.copy2(de440s_kernel_filepath, cache_kernel_path)

    yield str(cache_kernel_path)


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


@pytest.fixture(scope="module", autouse=True)
def eop(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    brahe.set_global_eop_provider(eop)


@pytest.fixture(scope="function")
def eop_original_brahe(brahe_original_eop_filepath):
    """EOP provider using the original brahe EOP file for reference TLE tests."""
    eop = brahe.FileEOPProvider.from_file(brahe_original_eop_filepath, True, "Hold")
    brahe.set_global_eop_provider(eop)
    yield eop


@pytest.fixture(scope="module")
def point_earth():
    """Two-body point mass Earth dynamics for 6D state [r, v].

    Returns a dynamics function suitable for orbital integration tests.
    """

    def dynamics(t, state):
        r = state[:3]
        v = state[3:]

        r_norm = np.linalg.norm(r)
        a_mag = -brahe.GM_EARTH / (r_norm**3)
        a = a_mag * r

        return np.concatenate([v, a])

    return dynamics
