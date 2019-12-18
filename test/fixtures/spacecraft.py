import json
import pytest

import brahe.data_models as bdm
from .paths import TEST_DATA

@pytest.fixture
def spacecraft_polar():
    filepath = TEST_DATA / 'spacecraft_polar.json'
    spacecraft_json = json.load(open(filepath, 'r'))
    yield bdm.Spacecraft(**spacecraft_json)