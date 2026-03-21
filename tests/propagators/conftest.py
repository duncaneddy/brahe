import pathlib
import pytest
import brahe


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_ASSETS = PACKAGE_ROOT / "test_assets"


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
