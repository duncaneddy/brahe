import pytest
from pytest import approx
import uuid
import json

import brahe.data_models as bdm
from brahe.access.tessellation import *

def test_tessellate_point_point(spacecraft_polar, request_sf_point):
    tiles = tessellate(spacecraft_polar, request_sf_point)

    assert len(tiles) == 2


def test_tessellate_point_polygon(spacecraft_polar, request_sf_polygon):
    tiles = tessellate(spacecraft_polar, request_sf_polygon)

    assert len(tiles) == 4