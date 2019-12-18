import json
import pytest

import brahe
from .paths import TEST_DATA

@pytest.fixture
def tle_polar():
    filepath = TEST_DATA / 'tle_polar.json'
    tle_json = json.load(open(filepath, 'r'))
    yield brahe.TLE(tle_json['line1'], tle_json['line2'])

@pytest.fixture
def tle_inclined():
    filepath = TEST_DATA / 'tle_inclined.json'
    tle_json = json.load(open(filepath, 'r'))
    yield brahe.TLE(tle_json['line1'], tle_json['line2'])
