# Test Imports
import pytest
from pytest import approx
import math
import numpy as np

# Modules Under Test
from brahe.epoch import Epoch
import brahe.astro as _astro
import brahe.orbit_dynamics.relativity as _rel

def test_accel_relativity(state_keplerian_deg):
    a_rel = _rel.accel_relativity(_astro.sOSCtoCART(state_keplerian_deg, use_degrees=True))

    assert 0.0 < math.fabs(a_rel[0]) < 1.0e-7
    assert 0.0 < math.fabs(a_rel[1]) < 1.0e-7
    assert 0.0 < math.fabs(a_rel[2]) < 1.0e-7