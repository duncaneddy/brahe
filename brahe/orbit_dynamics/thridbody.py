# -*- coding: utf-8 -*-
"""This orbit dynamics submoduble provides functions for computing thrid-body
perturbations on orbits.
"""

# Imports
import logging
import copy    as copy
import math    as math
import numpy   as np

import brahe.constants as _constants
from   brahe.epoch       import Epoch
from   brahe.orbit_dyanmics.grav import accel_point_mass
