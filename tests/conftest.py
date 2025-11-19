import pathlib
import pytest
import numpy as np
import brahe

# Configure matplotlib to use non-GUI backend for testing
import matplotlib

matplotlib.use("Agg")

# Testing Paths
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ASSETS = PACKAGE_ROOT / "test_assets"


@pytest.fixture(scope="session")
def iau2000_c04_20_filepath():
    filepath = TEST_ASSETS / "EOP_20_C04_one_file_1962-now.txt"
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
