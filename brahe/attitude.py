# -*- coding: utf-8 -*-
"""This module provides functions to convert between different attitude
representations, as well as computing 
"""

# Imports
import logging as _logging
import copy    as _copy
import math    as _math
import numpy   as _np

# Brahe Imports
from   brahe.utils import logger
import brahe.constants as _constants
from   brahe.epoch     import Epoch

#####################
# Rotation Matrices #
#####################

def Rx(angle:float, use_degrees:bool=False) -> _np.ndarray:
    """Rotation matrix, for a rotation about the x-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        r (np.ndarray): Rotation matrix

    References:
    
        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    
    if use_degrees:
        angle *= _math.pi/180.0


    c = _math.cos(angle)
    s = _math.sin(angle)

    return _np.array([[1.0,  0.0,  0.0],
                      [0.0,   +c,   +s],
                      [0.0,   -s,   +c]])

def Ry(angle:float, use_degrees:bool=False) -> _np.ndarray:
    """Rotation matrix, for a rotation about the y-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        r (np.ndarray): Rotation matrix

    References:
    
        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    
    if use_degrees:
        angle *= _math.pi/180.0


    c = _math.cos(angle)
    s = _math.sin(angle)

    return _np.array([[ +c,  0.0,   -s],
                      [0.0, +1.0,  0.0],
                      [ +s,  0.0,   +c]])

def Rz(angle:float, use_degrees:bool=False) -> _np.ndarray:
    """Rotation matrix, for a rotation about the z-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        r (np.ndarray): Rotation matrix

    References:
    
        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    
    if use_degrees:
        angle *= _math.pi/180.0


    c = _math.cos(angle)
    s = _math.sin(angle)

    return _np.array([[ +c,   +s,  0.0],
                      [ -s,   +c,  0.0],
                      [0.0,  0.0,  1.0]])