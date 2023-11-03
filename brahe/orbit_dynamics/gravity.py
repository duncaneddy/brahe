"""This orbit dynamics submoduble provides functions for computing gravitational
forces or satellites.
"""

# Imports
import logging
import copy    as copy
import pathlib as pathlib
import math    as math
import numpy   as np
import numba   as numba

# Internal Imports
from   brahe.utils import kron_delta, AbstractArray
import brahe.constants   as _constants
from   brahe.epoch       import Epoch
import brahe.ephemerides as _ephem

# Get Logger
logger = logging.getLogger(__name__)

# Gravity Models
GRAV_MODEL_EGM2008_90 = _constants.DATA_PATH / 'EGM2008_90.gfc'
GRAV_MODEL_GGM05C     = _constants.DATA_PATH / 'GGM05C.gfc'

#######################
# Gravity Model Class #
#######################

class GravityModel():
    '''GravityModel stores a spherical harmonic representation of the gravity
    field.

    Attributes:
        data   (:obj:`np.ndarray`): gravity field coefficients stored as square matrix.
            C coeffients are store in the lower triangular corner and diagonal
            S coeffients in the upper triangular corner excluding the diagonal
        n_max  (:obj:`int`): maximum model degree
        m_max  (:obj:`int`): maximum model order
        gm     (:obj:`float`): gravity constant
        radius (:obj:`float`): equatorial raidus of body
        normalization (:obj:`str`): gravity model normalization type
        tides  (:obj:`str`): model tide system
    '''

    # Initialize class variables
    _data        = np.array((1, 1))
    _initialized = False
    n_max        = 0
    m_max        = 0
    gm           = 0
    radius       = 0
    tides        = ''
    modelname    = ''
    errors       = ''
    norm         = ''

    @classmethod
    def initialize(cls, filepath=GRAV_MODEL_EGM2008_90):
        '''Initialize GravityModel if it is not already initialized.'''

        if not cls._initialized:
            cls.load(filepath)

    @classmethod
    def is_initialized(cls):
        '''Returns whether gravity model has been initialized.

        Returns:
            bool: Returns `True` if model has been previously initialized, False otherwise.
        '''

        return cls._initialized

    @classmethod
    def is_normalized(cls):
        '''Returns whether loaded gravity field mode

        Returns:
            bool: Returns `True` if gravity model coefficients are normalized, False otherwise.
        '''

        return cls.normalization == 'fully_normalized'

    @classmethod
    def load(cls, filepath=GRAV_MODEL_EGM2008_90):
        '''Loads spherical harmonic gravity model file into memory.

        Arguments:
            filepath - String path to gravity field model to load.
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

        logger.debug(f'GravityModel loading from file: {filepath}')
        
        # Open file
        with open(filepath) as fp:
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
                    cls.normalization = line.split()[1]
                elif line[0:11] == 'tide_system':
                    cls.tides = line.split()[1]

            # Initialize CS
            cls._data = np.zeros((cls.n_max + 1, cls.m_max + 1))

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
                cls._data[n, m]     = C
                cls._data[m - 1, n] = S  

        cls.initialized = True

######################
# Point Mass Gravity #
######################

@numba.jit(nopython=True, cache=True)
def accel_point_mass(r_sat:np.ndarray, r_body:np.ndarray=np.zeros(3), gm:float=_constants.GM_EARTH) -> np.ndarray:
    """Computes the acceleration of a satellite caused by a point-mass approximation 
    of the central body. Returns the acceleration vector of the satellite.

    Assumes the satellite is much, much less massive than the central body.

    Arguments:
        r_sat (np.ndarray): satellite position in a common inertial frame. Units: *m*
        r_body (np.ndarray, optional): position of body in a common inertial frame. Units: *m*
        gm (float): gravitational coeffient of attracting body. Units: [*m*^3/*s*^2] Default: brahe.constants.GM_EARTH)
    
    Returns:
        np.ndarray: Acceleration in the cartesian inertial frame. Units: [m/s^2]
    """

    # Ensure r_body exists if
    if not np.any(r_body):
        r_body = np.zeros(3)

    # Ensure input is array-like
    r_sat  = np.asarray(r_sat)
    r_body = np.asarray(r_body)

    # Restrict inputs to position only
    r_sat  = r_sat[0:3]
    r_body = r_body[0:3]

    # Relative position vector of satellite w.r.t. the attraching body
    d = r_sat - r_body

    # Acceleration
    if np.any(r_body):
        a = - gm * (d/np.linalg.norm(d)**3 + r_body/np.linalg.norm(r_body)**3)
    else:
        a = - gm * d/np.linalg.norm(d)**3

    return a

##############################
# Spherical Harmonic Gravity #
##############################

@numba.jit(nopython=True, cache=True)
def _facprod(n:int, m:int) -> float:
    '''Helper function to ain in denormalization of gravity field coefficients.
    This method computes the factorial ratio (n-m)!/(n+m)! in an efficient
    manner, without computing the full factorial products.

    Args:
        n (:obj:`int`): Gravity model degree, n.
        m (:obj:`int`): Gravity model degree m.

    Returns:
        p (:obj:`float`) Factorial product
    '''
    p = 1.0

    for i in range(n-m+1, n+m+1): # TODO: Confirm range termination of n+m+1 vs n+m
        p = p/i

    return p

@numba.jit(nopython=True, cache=True)
def _compute_spherical_harmonics(r_bf: np.ndarray, CS: np.ndarray, n_max: int, 
                                   m_max: int, r_ref: float, GM: float,
                                   is_normalized: bool) -> np.ndarray:
    '''Compute spherical harmonic expansion for gravity field. Internal helper
    method used by accel_gravity. To enable JIT compiling of the primary computational
    worload of the 

    Args:
        r_bf (:obj:`np.ndarray`): Position in body-fixed frame
        CS (:obj:`np.ndarray)
    '''

    # Auxiliary quantities
    r_sqr = np.dot(r_bf, r_bf) # Square of distance
    rho   = r_ref * r_ref / r_sqr
    x0    = r_ref * r_bf[0] / r_sqr # Normalized
    y0    = r_ref * r_bf[1] / r_sqr # coordinates
    z0    = r_ref * r_bf[2] / r_sqr

    # Initialize V and W intemetidary matrices
    V = np.zeros((n_max+2, n_max+2))
    W = np.zeros((n_max+2, n_max+2))
    
    # Calculate zonal terms V(n,0); set W(n,0)=0.0
    V[0, 0] = r_ref / math.sqrt(r_sqr)
    W[0, 0] = 0.0

    V[1, 0] = z0 * V[0, 0]
    W[1, 0] = 0.0

    for n in range(2, n_max+2):
        V[n, 0] = ((2 * n - 1) * z0 * V[n - 1, 0] - (n - 1) * rho * V[n - 2, 0]) / n
        W[n, 0] = 0.0

    # Calculate tesseral and sectorial terms
    for m in range(1, m_max+2):
        # Calculate V(m,m) .. V(n_max+1,m)
        V[m, m] = (2 * m - 1) * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])

        if m <= n_max:
            V[m + 1, m] = (2 * m + 1) * z0 * V[m, m]
            W[m + 1, m] = (2 * m + 1) * z0 * W[m, m]

        for n in range(m+2, n_max+2):
            V[n, m] = ((2 * n - 1) * z0 * V[n - 1, m] - (n + m - 1) * rho * V[n - 2, m]) / (n - m)
            W[n, m] = ((2 * n - 1) * z0 * W[n - 1, m] - (n + m - 1) * rho * W[n - 2, m]) / (n - m)

    # Calculate accelerations ax,ay,az
    ax = ay = az = 0.0

    for m in range(m_max+1):
        for n in range(m, n_max+1):
            if m == 0:
                # Denormalize coefficients, if required
                if is_normalized:
                    N = math.sqrt(2*n + 1)
                    C = N * CS[n, 0]
                else:
                    C = CS[n, 0]

                ax -= C * V[n + 1, 1]
                ay -= C * W[n + 1, 1]
                az -= (n + 1) * C * V[n + 1, 0]
            else:
                # Denormalize coefficients, if required
                if is_normalized:
                    N = math.sqrt((2 - kron_delta(0,m)) * (2*n + 1) * _facprod(n, m))
                    C = N * CS[n, m]
                    S = N * CS[m -1, n]
                else:
                    C = CS[n, m]
                    S = CS[m -1, n]

                Fac  = 0.5 * (n - m + 1) * (n - m + 2)
                ax  += + 0.5 * (-C * V[n + 1, m + 1] - S * W[n + 1, m + 1]) \
                       + Fac * (+C * V[n + 1, m - 1] + S * W[n + 1, m - 1])
                ay  += + 0.5 * (-C * W[n + 1, m + 1] + S * V[n + 1, m + 1]) \
                       + Fac * (-C * W[n + 1, m - 1] + S * V[n + 1, m - 1])
                az  += (n - m + 1) * (-C * V[n + 1, m] - S * W[n + 1, m])

    # Body-fixed acceleration
    a_bf = (GM / (r_ref * r_ref)) * np.array([ax, ay, az])

    return a_bf

def accel_gravity(x:np.ndarray, R_i2b:np.ndarray, n_max:int=20, m_max:int=20) -> np.ndarray:
    '''Compute the accceleration due to gravity

    Args:
        x (:obj:`np.ndarray`): Object position in the inertial frame centered at the pirmary body or system barycenter.
        R_i2b (:obj:`np.ndarray`): Rotation matrix that transform _from_ inertial _to_ body frame. That is x_ecef = R_i2b @ x
        n_max (int) Maximum gravity model degree
        m_max (int) Maximum gravity model order

    Returns:
        np.ndarray: Acceleration in the cartesian inertial frame. Units: [m/s^2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.56-68.
    '''

    # Ensure Gravity Model Initialized
    if not GravityModel.is_initialized():
        GravityModel.initialize()

    # Check Limits of Gravity Field
    if n_max > GravityModel.n_max:
        raise RuntimeError(f"Requested gravity model order {n_max} is larger than the maximum order of the model maximum {GravityModel.n_max}.")

    if m_max > GravityModel.m_max:
        raise RuntimeError(f"Requested gravity model order {m_max} is larger than the maximum order of the model maximum {GravityModel.m_max}.")

    # Gravitational parameters of primary body
    GM    = GravityModel.gm
    R_ref = GravityModel.radius

    # Body-fixed position
    r_bf = R_i2b @ x[0:3]

    # Compute spherical harmonic acceleration
    a_ecef = _compute_spherical_harmonics(r_bf, GravityModel._data, n_max, m_max, R_ref, GM, is_normalized=GravityModel.is_normalized())

    # # Inertial acceleration
    a_eci = np.transpose(R_i2b) @ a_ecef

    # # Finished
    return a_eci

######################
# Third-Body Gravity #
######################

def accel_thirdbody(x_eci:np.ndarray, r_body:np.ndarray, gm:float) -> np.ndarray:
    '''Computes the acceleration of an object in the inertial frame due to the
    gravitational attraction of a third-body.

    Args:
        x_eci (:obj:`np.ndarray`): Satellite state in the inertial frame
        r_body (:obj:`np.ndarray`): Position of the third body in the frame of the primary body.

    Returns:
        np.ndarray: Acceleration in the cartesian inertial frame. Units: [m/s^2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.69-70.
    '''

    return accel_point_mass(x_eci[0:3], r_body, gm)

def accel_thirdbody_sun(epc:Epoch, x_eci:np.ndarray) -> np.ndarray:
    '''Computes the acceleration of an object in the inertial frame due to the
    gravitational attraction of the Sun.

    Args:
        epc (:obj:`Epoch`): Time of 
        x_eci (:obj:`np.ndarray`): Satellite state in the inertial frame

    Returns:
        np.ndarray: Acceleration in the cartesian inertial frame. Units: [m/s^2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.69-70.
    '''

    return accel_thirdbody(x_eci, _ephem.sun_position(epc), _constants.GM_SUN)

def accel_thirdbody_moon(epc:Epoch, x_eci:np.ndarray) -> np.ndarray:
    '''Computes the acceleration of an object in the inertial frame due to the
    gravitational attraction of the Moon.

    Args:
        epc (:obj:`Epoch`): Time of 
        x_eci (:obj:`np.ndarray`): Satellite state in the inertial frame

    Returns:
        np.ndarray: Acceleration in the cartesian inertial frame. Units: [m/s^2]

    References:
        1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.69-70.
    '''

    return accel_thirdbody(x_eci, _ephem.moon_position(epc), _constants.GM_MOON)

