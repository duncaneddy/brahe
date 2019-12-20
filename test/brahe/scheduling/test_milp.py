import pytest
from pytest import approx
import uuid

from brahe.epoch import Epoch

import brahe.data_models as bdm
from brahe.scheduling.tessellation import tessellate
from brahe.scheduling.access import *

def milp(tle_polar):
    pass