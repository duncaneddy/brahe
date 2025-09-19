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

@pytest.fixture(scope='session')
def brahe_original_eop_filepath():
    filepath = TEST_ASSETS / 'brahe_original_eop_file.txt'
    yield str(filepath)

@pytest.fixture(scope='module', autouse=True)
def eop(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(iau2000_standard_filepath, True, "Hold")
    brahe.set_global_eop_provider_from_file_provider(eop)

@pytest.fixture(scope='function')
def eop_original_brahe(brahe_original_eop_filepath):
    """EOP provider using the original brahe EOP file for reference TLE tests."""
    eop = brahe.FileEOPProvider.from_file(brahe_original_eop_filepath, True, "Hold")
    brahe.set_global_eop_provider_from_file_provider(eop)
    yield eop