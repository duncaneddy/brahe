# Test Imports
from pytest import approx

# Modules Under Test
import brahe.utils as butil

def test_tle_format_exp():
    # i x j == k
    c = butil.fcross([1, 0, 0], [0, 1, 0])
    assert c[0] == 0
    assert c[1] == 0
    assert c[2] == 1


    c = butil.fcross([0, 0, 1], [1, 0, 0])
    assert c[0] == 0
    assert c[1] == 1
    assert c[2] == 0

    # j x k == i
    c = butil.fcross([0, 1, 0], [0, 0, 1])
    assert c[0] == 1
    assert c[1] == 0
    assert c[2] == 0