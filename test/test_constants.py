#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
from   pytest import approx
from   os import path

# Import module under test
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from brahe.constants import *

# Other imports
import math

def test_constants():
    assert AS2RAD == 2.0*math.pi/360.0/3600.0

if __name__ == '__main__':
    test_constants()