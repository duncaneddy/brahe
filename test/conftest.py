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

@pytest.fixture(scope='module')
def eop(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(iau2000_standard_filepath, True, "Hold")
    brahe.set_global_eop_provider_from_file_provider(eop)