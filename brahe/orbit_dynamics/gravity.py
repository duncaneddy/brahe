# -*- coding: utf-8 -*-
"""This orbit dynamics submoduble provides functions for computing gravitational
forces or satellites.
"""

# Imports
import logging
import copy    as _copy
import pathlib as _pathlib
import math    as _math
import numpy   as _np
import numba   as _numba

# Internal Imports
import brahe.constants   as _constants
from   brahe.epoch       import Epoch
import brahe.ephemerides as _ephem

# Get Logger
logger = logging.getLogger(__name__)

# Gravity Models
GRAV_MODELGRAV_MODEL_EGM2008_90 = _constants.DATA_PATH / 'EGM2008_90.gfc'
GRAV_MODEL_GGM05C     = _constants.DATA_PATH / 'GGM05C.gfc'

######################
# Point Mass Gravity #
######################

def accel_point_mass(r_sat:_np.ndarray, r_body:_np.ndarray, gm:float=_constants.GM_EARTH):
    """Computes the acceleration of a satellite caused by a point-mass approximation 
    of the central body. Returns the acceleration vector of the satellite.

    Assumes the satellite is much, much less massive than the central body.

    Arguments:
        r_sat (np.ndarray): satellite position in a commonn inertial frame. Units: *m*
        r_body (np.ndarray): position of body in a commonn inertial frame. Units: *m*
        gm (float): gravitational coeffient of attracting body. Units: [*m*^3/*s*^2]
            Default: brahe.constants.GM_EARTH)
    Return:
        a (np.ndarray): Acceleration in X, Y, and Z inertial directions [m/s^2]
    """

    # Ensure input is array-like
    r_sat  = _np.asarray(r_sat)
    r_body = _np.asarray(r_body)

    # Restrict inputs to position only
    r_sat  = r_sat[0:3]
    r_body = r_body[0:3]

    # Relative position vector of satellite w.r.t. the attraching body
    d = r_sat - r_body

    # Acceleration
    a = - gm * (d/_np.linalg.norm(d)**3 + r_body/_np.linalg.norm(r_body)**3)

    return a

#######################
# Gravity Model Class #
#######################

class GravityModel():
    '''GravityModel stores a spherical harmonic representation of the gravity
    field.

    Attributes:
        _CS (:obj:`np.ndarray`): gravity field coefficients stored as square matrix.
            C coeffients are store in the lower triangular corner and diagonal
            S coeffients in the upper triangular corner excluding the diagonal
        _n_max (:obj:`int`): maximum model degree
        _m_max (:obj:`int`): maximum model order
        _gm    (:obj:`float`): gravity constant
        _r     (:obj:`float`): equatorial raidus of body
        _normalized (:obj:`str`): coefficient normalization status
        _tides (:obj:`str`): model tide system
    '''

    # Initialize class variables
    _initialized = False
    _CS          = _np.array((1, 1))
    _n_max       = 0
    _m_max       = 0
    _gm          = 0
    _radius      = 0
    _tides       = ''
    _modelname   = ''
    _errors      = ''
    _norm        = ''

    @classmethod
    def initialize(cls, file_path=GRAV_MODEL_EGM2008_90):
        '''Initialize GravityModel if it is not already initialized.'''

        if not cls.initialized:
            cls.load(file_path)

    @classmethod
    def load(cls, file_path=GRAV_MODEL_EGM2008_90):
        '''Loads spherical harmonic gravity model file into memory.

        Arguments:
            file_path - String path to gravity field model to load.
                        (Default: EGM2008_90)

        Notes:
            1) Will convert normalized coefficients into unnormalized values for
               storage.

            2) Will convert correct C2,0 term so the gravity model is a 
               'zero-tide' gravity model.

            3) Claification on the terminology of tides:
                'tide-free' - the gravity field of the Earth assuming that the
                              moon and sun do not exist.

                'mean-tide' - The gravity field of the Earth, plus the gravity 
                              fields of the Moon and the Sun averaged over a 
                              long time (which is called the permanent tidal effect), 
                              plus the effect of Earth's deformation, caused by 
                              Moon and Sun, on the gravity field (also averaged 
                              over time).

                'zero-tide' - The gravity field of the Earth without the gravity
                              fields of the Moon and Sun, but with their
                              effect on Earth's deformation, and therefore indirect
                              effect on Earth's gravity field.
        '''

        logger.debug(f'GravityModel loading file: {file_path:s}')
        
        # Open file
        with open(file_path) as fp:
            # Read first line
            line = fp.readline()
            
            # Read header
            while line[0:11] != 'end_of_head':
                line = fp.readline()

                if line[0:9] == 'modelname':
                    cls.modelname = line.split()[1]
                elif line[0:22] == 'earth_gravity_constant':
                    cls.gm = float(line.split()[1])
                elif line[0:6] == 'radius':
                    cls.radius = float(line.split()[1])
                elif line[0:10] == 'max_degree':
                    cls.n_max = int(line.split()[1])
                    cls.m_max = cls.n_max
                elif line[0:6] == 'errors':
                    cls.errors = line.split()[1]
                elif line[0:4] == 'norm':
                    cls.norm = line.split()[1]
                elif line[0:11] == 'tide_system':
                    cls.tides = line.split()[1]

            # Initialize CS
            cls._CS = _np.zeros((cls.n_max + 1, cls.m_max + 1))

            # Read in gravity model data:
            for line in fp:
                # Reformat possible scietific notation characters to e
                line = line.replace('d', 'e').replace('D', 'e')

                # Expand line into values
                _, n, m, C, S, sig_c, sig_s = line.replace('d','e').split()

                # Convert values from string to numeric types
                n     = int(n)
                m     = int(m)
                C     = float(C)
                S     = float(S)
                sig_c = float(sig_c)
                sig_s = float(sig_s)

                # Store coefficients in matrix
                cls._CS[n, m]     = C
                cls._CS[m - 1, n] = S  

        cls.initialized = True

##############################
# Spherical Harmonic Gravity #
##############################

"""
Kronecker Delta Function.

Returns 1 if inputs are equal returns 0 otherwise.

Arguments:
- `a::Real`: First input argument
- `b::Real`: Second input argument 

Returns:
- `delta:Integer`: Kronecker delta result.
"""
function kron(a::Real, b::Real)
    return a == b ? 1 : 0
end

"""
Internal helper function to aid in denormalization of gravity field coefficients.

Provides method co compute the factorial ratio: (n-m)!/(n+m)!

Arguments:
- `n::Real`: Gravity model degree, n.
- `m::Real`: Gravity model order, m.

Returns:
- `p::Real`: Factorial product
"""
def _facprod(n:int, m:int) -> float:
    '''Helper function to ain in denormalization of gravity field coefficients.
    This method computes the factorial ratio (n-m)!/(n+m)! in an efficient
    manner, without computing the full factorial products.

    A
    '''
    p = 1.0

    for i in range(n-m+1, n+m+1): # TODO: Confirm range termination of n+m+1 vs n+m
        p = p/i

    return p


######################
# Third-Body Gravity #
######################
