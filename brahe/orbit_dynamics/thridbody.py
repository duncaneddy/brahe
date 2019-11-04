# -*- coding: utf-8 -*-
"""This orbit dynamics submoduble provides functions for computing thrid-body
perturbations on orbits.
"""

# Imports
import logging
import copy    as _copy
import math    as _math
import numpy   as _np

import brahe.constants as _constants
from   brahe.epoch       import Epoch
from   brahe.orbit_dyanmics.grav import accel_point_mass
