# Test Imports
import pytest
from pytest import approx
import uuid

# Modules Under Test
from brahe.epoch import Epoch
from brahe.data_models.spacecraft import *

def test_spacecraft():
    spacecraft_json = {
        "id": 1,
        "name": "Spacecraft 1",
        "line1": "1 39418U 13066C   19350.83278205 +.00000478 +00000-0 +44250-4 0  9992",
        "line2": "2 39418 097.6393 069.2166 0023256 099.0938 261.2918 14.98892401331776",
        "model": {
            "slew_rate": 1.0,
            "settling_time": 15.0
        }
    }

    sc = Spacecraft(**spacecraft_json)

    assert sc.id == 1
    assert sc.name == "Spacecraft 1"