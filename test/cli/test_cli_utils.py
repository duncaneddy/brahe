import pytest
import brahe
from brahe import Epoch
from brahe.cli.utils import epoch_from_epochlike, parse_float

def test_parse_float():
    assert parse_float("1") == 1.0
    assert parse_float("1.5") == 1.5
    assert parse_float("1.5e-3") == 0.0015
    assert parse_float("potato") == None
    assert parse_float("2022-01-01T00:00:00") == None

def test_epoch_from_epochlike_mjd():
    assert epoch_from_epochlike("55420.0") == Epoch.from_mjd(55420.0, brahe.TimeSystem.UTC)
    assert epoch_from_epochlike("55420") == Epoch.from_mjd(55420.0, brahe.TimeSystem.UTC)

def test_epoch_from_epochlike_jd():
    assert epoch_from_epochlike("2455420.0") == Epoch.from_jd(2455420.0, brahe.TimeSystem.UTC)
    assert epoch_from_epochlike("2455420") == Epoch.from_jd(2455420.0, brahe.TimeSystem.UTC)

def test_epoch_from_epochlike_string():
    assert epoch_from_epochlike("2022-01-01T00:00:00Z") == Epoch.from_string("2022-01-01T00:00:00Z")
    assert epoch_from_epochlike("2022-01-01T00:00:00.000Z") == Epoch.from_string("2022-01-01T00:00:00Z")
    assert epoch_from_epochlike("2022-01-01 00:00:00 GPS") == Epoch.from_string("2022-01-01 00:00:00 GPS")
    assert epoch_from_epochlike("2022-01-01") == Epoch.from_string("2022-01-01")