#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
import logging
from   pytest import approx
from   os     import path
import math
import numpy as np

# Import module undera test
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# Set Log level
LOG_FORMAT = '%(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

# Import modules for testing
from brahe.constants import *
from brahe.epoch     import *
from brahe.orbits    import *

if __name__ == '__main__':
    pass