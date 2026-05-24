"""Smoke test: load brahe's EOP/space-weather data into astrojax's providers.

Skipped if astrojax is not importable in the parent process."""
import pytest

from benchmarks.gpu_comparison.config import BRAHE_EOP_FILE, BRAHE_SPACE_WEATHER_FILE

astrojax = pytest.importorskip("astrojax")


def test_load_brahe_eop_into_astrojax():
    from benchmarks.gpu_comparison.data_alignment import (
        load_eop_for_astrojax,
        load_space_weather_for_astrojax,
    )
    eop = load_eop_for_astrojax(BRAHE_EOP_FILE)
    assert eop is not None
    sw = load_space_weather_for_astrojax(BRAHE_SPACE_WEATHER_FILE)
    assert sw is not None
