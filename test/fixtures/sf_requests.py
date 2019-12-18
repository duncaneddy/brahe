import json
import pytest

import brahe.data_models as bdm
from .paths import TEST_DATA

@pytest.fixture
def request_sf_point():
    filepath = TEST_DATA / 'request_sf_point.json'
    request_json = json.load(open(filepath, 'r'))
    yield bdm.Request(**request_json)

@pytest.fixture
def request_sf_polygon():
    filepath = TEST_DATA / 'request_sf_polygon.json'
    request_json = json.load(open(filepath, 'r'))
    print(request_json)
    yield bdm.Request(**request_json)