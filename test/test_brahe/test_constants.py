# Test Imports
from pytest import approx
import math

# Modules Under Test
from brahe.constants import *

def test_constants():
    assert AS2RAD == 2.0*math.pi/360.0/3600.0

if __name__ == '__main__':
    test_constants()