import pathlib
import pytest
import brahe

# Testing Paths
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ASSETS  = PACKAGE_ROOT / 'test' / 'assets'

@pytest.fixture(scope='session')
def iau2000_c04_14_filepath():
    filepath = TEST_ASSETS / 'iau2000A_c04_14.txt'
    yield str(filepath)


@pytest.fixture(scope='session')
def iau2000_finals_ab_filepath():
    filepath = TEST_ASSETS / 'iau2000A_finals_ab.txt'
    yield str(filepath)

# @pytest.fixture
# def eop(iau2000_c04_14_filepath):
#     brahe.set_global_eop_from_c04_file(iau2000_c04_14_filepath, "Hold", True)