# -*- coding: utf-8 -*-
"""The tle module provides class definitions to provide an convenient interface
for interacting with NORAD Two-Line Element (TLE) sets, the associated SGP4
propagated, as well as how to convert the output from the base frame to other
common reference frames.

Note:
    The implementation of SGP4 propagator comes from Brandon Rhoade's Python
    implemntation `pysofa2 <https://github.com/duncaneddy/pysofa2/>`_. His
    python implementation is based on the original code provided by David
    Vallado in _Revisiting Spacetrack Report #3_
"""

# Imports
import pysofa2 as _sofa
import brahe.constants as _constants

# Brahe Imports
from   brahe.utils import logger

#############
# TLE Class #
#############