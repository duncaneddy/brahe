import json
import pytest

import brahe.data_models as bdm
from .paths import TEST_DATA

@pytest.fixture
def stations():
    filepath = TEST_DATA / 'stations.json'
    stations = json.load(open(filepath, 'r'))
    yield [bdm.Station(**s) for s in stations]

@pytest.fixture
def station_svalbard(stations):
    yield stations[0]

@pytest.fixture
def station_troll(stations):
    yield stations[1]

@pytest.fixture
def station_tokyo(stations):
    yield stations[2]