import pytest
import brahe
import numpy as np
from pytest import approx

def test_state_osculating_to_cartesian(eop):
    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    cart = brahe.state_osculating_to_cartesian(osc, False)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)
    assert cart[5] == 0.0

    osc = np.array([brahe.R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0])
    cart = brahe.state_osculating_to_cartesian(osc, True)

    assert isinstance(cart, np.ndarray)
    assert cart[0] == brahe.R_EARTH + 500e3
    assert cart[1] == 0.0
    assert cart[2] == 0.0
    assert cart[3] == 0.0
    assert cart[4] == pytest.approx(0.0, abs=1.0e-12)
    assert cart[5] == brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0)

def test_state_cartesian_to_osculating(eop):
    cart = np.array([
        brahe.R_EARTH + 500e3,
        0.0,
        0.0,
        0.0,
        brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0),
        0.0,
        ])
    osc = brahe.state_cartesian_to_osculating(cart, True)

    assert osc[0] == approx(brahe.R_EARTH + 500e3, abs = 1e-9)
    assert osc[1] == 0.0
    assert osc[2] == 0.0
    assert osc[3] == 180.0
    assert osc[4] == 0.0
    assert osc[5] == 0.0

    cart = np.array([
        brahe.R_EARTH + 500e3,
        0.0,
        0.0,
        0.0,
        0.0,
        brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.0),
        ])
    osc = brahe.state_cartesian_to_osculating(cart, True)

    assert osc[0] == approx(brahe.R_EARTH + 500e3, abs = 1.0e-9)
    assert osc[1] == 0.0
    assert osc[2] == 90.0
    assert osc[3] == 0.0
    assert osc[4] == 0.0
    assert osc[5] == 0.0