import pathlib
import pytest
import brahe

# Testing Paths
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ASSETS  = PACKAGE_ROOT / 'test_assets'

@pytest.fixture(scope='session')
def iau2000_c04_20_filepath():
    filepath = TEST_ASSETS / 'EOP_20_C04_one_file_1962-now.txt'
    yield str(filepath)


@pytest.fixture(scope='session')
def iau2000_standard_filepath():
    filepath = TEST_ASSETS / 'finals.all.iau2000.txt'
    yield str(filepath)

@pytest.fixture
def eop(iau2000_c04_20_filepath):
    eop = brahe.FileEOPProvider.from_file(iau2000_c04_20_filepath, "Hold", True)
    brahe.set_global_eop_from_c04_file(eop)