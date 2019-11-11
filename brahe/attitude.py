# -*- coding: utf-8 -*-
"""This module provides functions to convert between different attitude
representations, as well as computing 
"""

# Imports
import logging
import copy
import math
import numpy as np

# Brahe Imports
from   brahe.utils import logger
import brahe.constants as _const
from   brahe.epoch import Epoch

#####################
# Rotation Matrices #
#####################

def Rx(angle:float, use_degrees:bool=False) -> np.ndarray:
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
        angle *= math.pi/180.0


    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([[1.0,  0.0,  0.0],
                      [0.0,   +c,   +s],
                      [0.0,   -s,   +c]])

def Ry(angle:float, use_degrees:bool=False) -> np.ndarray:
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
        angle *= math.pi/180.0


    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([[ +c,  0.0,   -s],
                      [0.0, +1.0,  0.0],
                      [ +s,  0.0,   +c]])

def Rz(angle:float, use_degrees:bool=False) -> np.ndarray:
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
        angle *= math.pi/180.0


    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([[ +c,   +s,  0.0],
                      [ -s,   +c,  0.0],
                      [0.0,  0.0,  1.0]])

##############
# Quaternion #
##############

class Quaternion():
    '''Quaternion representation of attitude. 

    Attributes:
        data (:obj:`np.ndarray`): Internal data representation of 

    References:
        1. J. Diebel, _Representing attitude: Euler angles, unit quaternions, and rotation vectors._ Matrix 58(15-16) (2006).
    '''

    def __init__(self, q0, q1:float=None, q2:float=None, q3:float=None, scalar:str='first'):
        '''Initialize Quaternion.

        Args:
            q0 (:obj:`np.ndarray`, :obj:`Quaternion`, :obj:`Euler Angle`, 
                :obj:`EulerAxis`, :obj:`RotationMatrix`, :obj:`float`): Object
                to initialize quaternion from. Can be any of the other data types
                defined in the Attitude module. If it is float, the 3 other
                Quaternion values must be provided.
            q1 (:obj:`float`, optional): Second component.
            q2 (:obj:`float`, optional): Third component.
            q3 (:obj:`float`, optional): Fouth component.
            scalar (:obj:`string`, optional): Indicate whether the scalar part
                is the first or last value of the input data if initializing from
                a :obj:`np.ndarray` or :obj:`float`.
        '''
        
        # Initialize data storage
        self.data = np.zeros(4)
        
        # Matrix and Quaternion Initialization
        if type(q0) == np.ndarray:
            
            # Vector initialization
            if q0.shape == (4,):
                if scalar.lower() == 'first':
                    # Store scalar part 
                    self.data = q0
                    
                    # Normalize input
                    self.normalize()

                elif scalar.lower() == 'last':
                    # Store scalar part
                    self.data = np.array([q0[3], q0[0], q0[1], q0[2]])

                    # Normalize input
                    self.normalize()

                else:
                    raise RuntimeError('Quaternion scalar part must be specifie as "first" or "second"')
            
            # Matrix initialization
            elif q0.shape == (3,3):
                self.data = np.copy(Quaternion(RotationMatrix(q0)).data)
            else:
                raise RuntimeError(f'Incompatible input shape: {q0.shape}')

        # Initialize from Euler Angle
        elif type(q0) == EulerAngle:
            # extract elements out of Euler angle sequence
            phi   = q0.data[0]
            theta = q0.data[1]
            psi   = q0.data[2]

            # Perform single call of each base trig function
            c1 = math.cos(phi/2.0)
            c2 = math.cos(theta/2.0)
            c3 = math.cos(psi/2.0)
            s1 = math.sin(phi/2.0)
            s2 = math.sin(theta/2.0)
            s3 = math.sin(psi/2.0)

            # populate the quaternion
            if q0.seq == 121:
                self.data[0] = c1*c2*c3 - s1*c2*s3
                self.data[1] = c1*c2*s3 + c2*c3*s1
                self.data[2] = c1*c3*s2 + s1*s2*s3
                self.data[3] = c1*s2*s3 - s1*c3*s2

            elif q0.seq == 123:
                self.data[0] =  c1*c2*c3 + s1*s2*s3
                self.data[1] = -c1*s2*s3 + c2*c3*s1
                self.data[2] =  c1*c3*s2 + s1*c2*s3
                self.data[3] =  c1*c2*s3 - s1*c3*s2

            elif q0.seq == 131:
                self.data[0] =  c1*c2*c3 - s1*c2*s3
                self.data[1] =  c1*c2*s3 + c2*c3*s1
                self.data[2] = -c1*s2*s3 + s1*c3*s2
                self.data[3] =  c1*c3*s2 + s1*s2*s3

            elif q0.seq == 132:
                self.data[0] =  c1*c2*c3 - s1*s2*s3
                self.data[1] =  c1*s2*s3 + c2*c3*s1
                self.data[2] =  c1*c2*s3 + s1*c3*s2
                self.data[3] =  c1*c3*s2 - s1*c2*s3

            elif q0.seq == 212:
                self.data[0] =  c1*c2*c3 - s1*c2*s3
                self.data[1] =  c1*c3*s2 + s1*s2*s3
                self.data[2] =  c1*c2*s3 + c2*c3*s1
                self.data[3] = -c1*s2*s3 + s1*c3*s2

            elif q0.seq == 213:
                self.data[0] =  c1*c2*c3 - s1*s2*s3
                self.data[1] =  c1*c3*s2 - s1*c2*s3
                self.data[2] =  c1*s2*s3 + c2*c3*s1
                self.data[3] =  c1*c2*s3 + s1*c3*s2

            elif q0.seq == 231:
                self.data[0] =  c1*c2*c3 + s1*s2*s3
                self.data[1] =  c1*c2*s3 - s1*c3*s2
                self.data[2] = -c1*s2*s3 + c2*c3*s1
                self.data[3] =  c1*c3*s2 + s1*c2*s3

            elif q0.seq == 232:
                self.data[0] =  c1*c2*c3 - s1*c2*s3
                self.data[1] =  c1*s2*s3 - s1*c3*s2
                self.data[2] =  c1*c2*s3 + c2*c3*s1
                self.data[3] =  c1*c3*s2 + s1*s2*s3

            elif q0.seq == 312:
                self.data[0] =  c1*c2*c3 + s1*s2*s3
                self.data[1] =  c1*c3*s2 + s1*c2*s3
                self.data[2] =  c1*c2*s3 - s1*c3*s2
                self.data[3] = -c1*s2*s3 + c2*c3*s1

            elif q0.seq == 313:
                self.data[0] =  c1*c2*c3 - s1*c2*s3
                self.data[1] =  c1*c3*s2 + s1*s2*s3
                self.data[2] =  c1*s2*s3 - s1*c3*s2
                self.data[3] =  c1*c2*s3 + c2*c3*s1

            elif q0.seq == 321:
                self.data[0] =  c1*c2*c3 - s1*s2*s3
                self.data[1] =  c1*c2*s3 + s1*c3*s2
                self.data[2] =  c1*c3*s2 - s1*c2*s3
                self.data[3] =  c1*s2*s3 + c2*c3*s1

            elif q0.seq == 323:
                self.data[0] =  c1*c2*c3 - s1*c2*s3
                self.data[1] = -c1*s2*s3 + s1*c3*s2
                self.data[2] =  c1*c3*s2 + s1*s2*s3
                self.data[3] =  c1*c2*s3 + c2*c3*s1

            else:
                raise RuntimeError(f'Euler angle sequence {q0.seq} invalid for Quaternion initialization')

        # Initialize from Euler Axis
        elif type(q0) == EulerAxis:
            self.data    = np.zeros(4)
            self.data[0] = math.cos(q0.data[0]/2)
            self.data[1] = q0.data[1]*math.sin(q0.data[0]/2.0)
            self.data[2] = q0.data[2]*math.sin(q0.data[0]/2.0)
            self.data[3] = q0.data[3]*math.sin(q0.data[0]/2.0)
            self.normalize()

        # Initialize from Rotation Matrix
        elif type(q0) == RotationMatrix:
            # Find optimal mapping selection indicators
            temp    = np.zeros(4)
            temp[0] = 1 + q0[0, 0] + q0[1, 1] + q0[2, 2]
            temp[1] = 1 + q0[0, 0] - q0[1, 1] - q0[2, 2]
            temp[2] = 1 - q0[0, 0] + q0[1, 1] - q0[2, 2]
            temp[3] = 1 - q0[0, 0] - q0[1, 1] + q0[2, 2]

            ind = np.argmax(temp)
            den = math.sqrt(np.amax(temp))

            # Select optimal inverse mapping of Rotation matrix to Quaternion
            if ind == 0:
                self.data[0] = 0.5 * den
                self.data[1] = 0.5 * (q0.data[1, 2] - q0.data[2, 1]) / den
                self.data[2] = 0.5 * (q0.data[2, 0] - q0.data[0, 2]) / den
                self.data[3] = 0.5 * (q0.data[0, 1] - q0.data[1, 0]) / den

            elif ind == 1:
                self.data[0] = 0.5 * (q0.data[1, 2] - q0.data[2, 1]) / den
                self.data[1] = 0.5 * den
                self.data[2] = 0.5 * (q0.data[0, 1] + q0.data[1, 0]) / den
                self.data[3] = 0.5 * (q0.data[2, 0] + q0.data[0, 2]) / den

            elif ind == 2:
                self.data[0] = 0.5 * (q0.data[2, 0] - q0.data[0, 2]) / den
                self.data[1] = 0.5 * (q0.data[0, 1] + q0.data[1, 0]) / den
                self.data[2] = 0.5 * den
                self.data[3] = 0.5 * (q0.data[1, 2] + q0.data[2, 1]) / den

            elif ind == 3:
                self.data[0] = 0.5 * (q0.data[0, 1] - q0.data[1, 0]) / den
                self.data[1] = 0.5 * (q0.data[2, 0] + q0.data[0, 2]) / den
                self.data[2] = 0.5 * (q0.data[1, 2] + q0.data[2, 1]) / den
                self.data[3] = 0.5 * den

            else:
                raise RuntimeError('Matrix not suited for Quaternion initialization.')

        # Scalar quaternion initialization
        elif (type(q0) == float or type(q0) == int) and \
             (type(q1) == float or type(q1) == int) and \
             (type(q2) == float or type(q2) == int) and \
             (type(q3) == float or type(q3) == int):

            if scalar.lower() == 'first':
                # Store scalar part 
                self.data = np.array([q0, q1, q2, q3])
                
                # Normalize input
                self.normalize()

            elif scalar.lower() == 'last':
                # Store scalar part
                self.data = np.array([q3, q0, q1, q2])
                
                # Normalize input
                self.normalize()

        elif type(q0) == Quaternion:
            self.data = np.copy(q0.data)

        else:
            raise TypeError(f'q0 input type of {str(type(q0))} cannot be used to initialize Quaternion.')


    # Access with [] operators
    def __getitem__(self, key):
        if type(key) == slice:
            return self.data[key]
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 4

            # Check for valid access
            if key < 0 or key >= 4:
                raise IndexError(f'Unable to access element {key} for quaternion of length 4.')

            # Get data from storage object
            return self.data[key]

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __setitem__(self, key, value):
        if type(key) == slice:
            self.data[key] = value
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 4

            # Check for valid access
            if key < 0 or key >= 4:
                raise IndexError(f'Unable to access element {key} for quaternion of length 4.')

            # Get data from storage object
            self.data[key] = value

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __len__(self):
        return 4

    def __str__(self):
        return f'[{self.data[0]}, {self.data[1]}, {self.data[2]}, {self.data[3]}]'

    # Logical operators
    def __eq__(self, other):
        return np.all(self.data == other.data)

    def __ne__(self, other):
        return np.all(self.data != other.data)

    # Arithmetic operators
    def __add__(self, other):
        if type(other) == Quaternion:
            return Quaternion(self.data + other.data)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if type(other) == Quaternion:
            # Normalize data for input quaternions
            self.normalize()
            other.normalize()

            # Quaternion Multiplication
            qcos = self.data[0]*other.data[0] - np.dot(self.data[1:4], other.data[1:4])
            qvec = self.data[0]*other.data[1:4] + other.data[0]*self.data[1:4] + np.cross(self.data[1:4], other.data[1:4])

            # Create output
            quat = Quaternion(qcos, qvec[0], qvec[1], qvec[2], 'first')

            return quat

        elif type(other) == float:
            quat = Quaternion(self)
            quat.data *= other

            return quat
        else:
            raise RuntimeError(f'Quaternion multiplication is not defined for Quaterion * {type(other)}')

    def __rmul__(self, other):
        if type(other) == float:
            quat = Quaternion(self)
            quat.data *= other

            return quat
        else:
            raise RuntimeError(f'Quaternion multiplication is not defined for Quaterion * {type(other)}')

    def __neg__(self):
        return Quaternion(-self.data[0:])

    # Functions
    def normalize(self):
        '''Normalize Quaternion. Operation is run in-place.
        '''
        self.data = self.data/np.linalg.norm(self.data)

    def conj(self):
        '''Compute quaternion conjugate.

        Returns:
            q (:obj:`Quaternion`): Conjugate Quaternion.
        '''

        # Negate all quaternion parts except for scalar part
        return Quaternion(self.data[0], *(-self.data[1:4]))

    def inverse(self):
        '''Compute inverse quaternion.

        Return:
            q (:obj:`Quaternion`): Inverse Quaternion.
        '''

        quat = self.conj()/(np.linalg.norm(self.data)**2)

        return quat

    def to_array(self):
        '''Return quaternion as numpy array.
        '''
        return self.data

    def to_matrix(self):
        '''Convert to rotation matrix represtation.

        Returns:
            data (:obj:`np.ndarray`): Return data as rotation matrix.
        '''
        return RotationMatrix(self).data

# Spherical Linear Interpolation of Quaternions
def slerp(q0, q1, t: float):
    '''Quaternion spherical linear interpolation
    
    Arguments:
        q0 (:obj:`np.ndarray`, :obj:`Quaternion`): Start quaternion
        q1 (:obj:`np.ndarray`, :obj:`Quaternion`): End quaternion
        t (:obj:`float`): Normalized time [0, 1] between time of first and second quaternions
    
    Returns:
        qt (:obj:`np.ndarray`,:obj:`Quaternion`): Quaternion at time of interpolation.

    References:
        1. https://en.wikipedia.org/wiki/Slerp
    '''

    if type(q0) == np.ndarray and type(q1) == np.ndarray:
        q0 = Quaternion(q0)
        q1 = Quaternion(q1)
    elif not type(q0) == Quaternion or not type(q1) == Quaternion:
        raise RuntimeError('slerp only defined for numpy arrays and Quatnernions')

    # Check Range on t
    if t < 0.0 or t > 1.0:
        raise RuntimeError('slerp interpolation is only valid for normalized times. t must be between 0 and 1 (inclusive).')

    # Normalize quatnerions
    q0.normalize()
    q1.normalize()

    # Compute math.comath.sine of the angle between the two vectors
    dot = np.dot(q0.data, q1.data)

    # If the dot product is negative, the quaternions have opposite handed-ness 
    # and slerp won't take the shortest path. Fix by revermath.sing one quaternion.
    if dot < 0.0:
        q1  = -q1
        dot = -dot

    # If the inputs are too close use linear interpolation instead
    if dot > 0.9995:
        qt = Quaternion(q0.data + (q1.data - q0.data)*t)
        qt.normalize()
        return qt

    theta0 = math.acos(dot) # Angle between input vectors
    theta  = theta0*t  # Angle between q0 and result quaternion

    s0 = math.cos(theta) - dot*math.sin(theta)/math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)

    qt = (s0 * q0.data) + (s1 * q1.data)


    return Quaternion(qt)



##############
# Euler Axis #
##############

class EulerAxis():
    '''Euler axis and angle attitude representation 

    Attributes:
        data (:obj:`np.ndarray`): Data vector where the first element is the
            Euler Angle (in radians), and the next 3 are the Euler axis.

    References:
        1. J. Diebel, _Representing attitude: Euler angles, unit quaternions, and rotation vectors._ Matrix 58(15-16) (2006).
    '''

    def __init__(self, ang, v0:float=None, v1:float=None, v2:float=None, use_degrees:bool=False):
        '''Initialize EulerAxis

        Args:
            ang (:obj:`np.ndarray`, :obj:`Quaternion`, :obj:`EulerAngle`, 
                :obj:`EulerAxis`, :obj:`RotationMatrix`, :obj:`float`): Object
                to initialize EulerAxis from. Can be any of the other data types
                defined in the Attitude module. If ang is a :obj:`float`,
                it is assumed to be the scalar magnitude of rotation, and 3 other
                values must be provided.
            v0 (:obj:`float`, optional): First axis component.
            v1 (:obj:`float`, optional): Second axis component.
            v2 (:obj:`float`, optional): Third axis component.
            use_degrees (:obj:`bool`, optional): Indicates the input angle should
                be interpreted as being in degrees.
        '''
        if type(ang) == np.ndarray:
            if len(ang) != 4:
                raise RuntimeError('To intiallize from numpy array first input must be of length 4.')

            self.data = np.copy(ang)
            
            if use_degrees:
                self.data[0] *= math.pi/180.0
        elif type(ang) == Quaternion \
             or type(ang) == EulerAngle \
             or type(ang) == RotationMatrix:
            self.data = np.zeros(4)

            # If input is Euler angle or rotation matrix, first create a quaternion
            if type(ang) == EulerAngle or type(ang) == RotationMatrix:
                ang = Quaternion(ang,)

            # Extract quaternion vector and normalize
            q = ang.data/np.linalg.norm(ang.data)
            
            # Ensure first element is positive
            if q[0] < 0:
                q = -q
            
            angle = q[0]
            vec = q[1:4]
            # Algorithm
            if abs(q[0]) <= 1:
                angle     = 2*math.acos(q[0])
                qvec_norm = np.linalg.norm(q[1:4])

                if qvec_norm > 1e-15:
                    vec = q[1:4]/qvec_norm
            
            # Populate output
            self.data[0] = angle # angle is element 0 [rad]
            self.data[1] = vec[0]
            self.data[2] = vec[1]
            self.data[3] = vec[2]

        elif type(ang) == float or type(ang) == int:

            self.data = np.array([ang, v0, v1, v2])

            # Normalize vector
            self.data[1:4] = self.data[1:4]/np.linalg.norm(self.data[1:4])

            if use_degrees:
                self.data[0] = self.data[0]*math.pi/180.0

        else: 
            raise RuntimeError('Unable to initialize EulerAxis from given inputs.')

    # Access with [] operators
    def __getitem__(self, key):
        if type(key) == slice:
            return self.data[key]
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 4

            # Check for valid access
            if key < 0 or key >= 4:
                raise IndexError(f'Unable to access element {key} for Euler Axis of length 4.')

            # Get data from storage object
            return self.data[key]

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __setitem__(self, key, value):
        if type(key) == slice:
            self.data[key] = value
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 4

            # Check for valid access
            if key < 0 or key >= 4:
                raise IndexError(f'Unable to access element {key} for Euler Axis of length 4.')

            # Get data from storage object
            self.data[key] = value

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __str__(self):
        return self.data.__str__()

    # Logical operators
    def __eq__(self, other):
        return np.all(self.data == other.data)

    def __ne__(self, other):
        return np.all(self.data != other.data)

###############
# Euler Angle #
###############

class EulerAngle():
    '''Euler angle attitude data structure.

    Attributes:
        seq (:obj:`float`): Euler angle sequence as an integer. (e.g. 123)
        data (:obj:`np.ndarray`): Euler angles as radians.

    References:
        1. J. Diebel, _Representing attitude: Euler angles, unit quaternions, and rotation vectors._ Matrix 58(15-16) (2006).
    '''

    def __init__(self, seq, ang1:float=None, ang2:float=None, ang3:float=None, use_degrees:bool=False):
        '''Initialize Euler Angle.

        Args:
            seq (:obj:`int`): Euler angle sequence as an integer.
            ang1 (:obj:`np.ndarray`, :obj:`Quaternion`, :obj:`EulerAngle`, 
                :obj:`EulerAxis`, :obj:`RotationMatrix`, :obj:`float`): Object
                to initialize EulerAngle from. Can be any of the other data types
                defined in the Attitude module. If ang1 is a :obj:`float`,
                the other 2 data members must be provided.
            ang2 (:obj:`float`, optional): Second component.
            ang3 (:obj:`float`, optional): Third component.
            use_degrees (:obj:`bool`, optional): Indicates the input angles should
                be interpreted as being in degrees.
        '''
        if type(seq) == int and type(ang1) == np.ndarray:
            if len(seq) != 4:
                raise RuntimeError('First input must be ndarray of length 4 to initialize EulerAngle.')

            self.seq  = seq[0]
            self.data = seq[1:4]

            if use_degrees:
                self.data[0:3] *= math.pi/180.0

        if type(seq) == int \
            and (type(ang1) == float or type(ang1) == int) \
            and (type(ang2) == float or type(ang2) == int) \
            and (type(ang3) == float or type(ang3) == int):

            self.seq  = seq
            self.data = np.array([ang1, ang2, ang3], dtype=np.float)

            if use_degrees:
                self.data[0:3] *= math.pi/180.0

        elif type(seq) == int and (type(ang1) == Quaternion \
            or type(ang1) == EulerAxis or type(ang1) == RotationMatrix):
            
            # First compute RotationMatrix if necessary
            if type(ang1) == Quaternion or type(ang1) == EulerAxis:
                rot = RotationMatrix(ang1)
            else:
                rot = ang1

            # Extract elements of rotation matrix
            for i in range(0, 3):
                for j in range(0, 3):
                    if abs(rot[i, j]) > 1:
                        rot[i, j] = np.sign(rot[i, j]) 

            r11 = rot[0, 0]
            r12 = rot[0, 1]
            r13 = rot[0, 2]
            r21 = rot[1, 0]
            r22 = rot[1, 1]
            r23 = rot[1, 2]
            r31 = rot[2, 0]
            r32 = rot[2, 1]
            r33 = rot[2, 2]

            # Select euler angle sequence
            self.seq  = seq
            self.data = np.zeros(3)
            if seq == 121:
                self.data[0] = math.atan2(r21, r31)
                self.data[1] = math.acos(r11)
                self.data[2] = math.atan2(r12, -r13)

            elif seq == 123:
                self.data[0] = math.atan2(r23, r33)
                self.data[1] = -math.asin(r13)
                self.data[2] = math.atan2(r12, r11)

            elif seq == 131:
                self.data[0] = math.atan2(r31, -r21)
                self.data[1] = math.acos(r11)
                self.data[2] = math.atan2(r13, r12)

            elif seq == 132:
                self.data[0] = math.atan2(-r32, r22)
                self.data[1] = math.asin(r12)
                self.data[2] = math.atan2(-r13, r11)

            elif seq == 212:
                self.data[0] = math.atan2(r12, -r32) 
                self.data[1] = math.acos(r22)
                self.data[2] = math.atan2(r21, r23)

            elif seq == 213:
                self.data[0] = math.atan2(-r13, r33)
                self.data[1] = math.asin(r23)
                self.data[2] = math.atan2(-r21, r22)

            elif seq == 231:
                self.data[0] = math.atan2(r31, r11)
                self.data[1] = -math.asin(r21)
                self.data[2] = math.atan2(r23, r22)

            elif seq == 232:
                self.data[0] = math.atan2(r32, r12)
                self.data[1] = math.acos(r22)
                self.data[2] = math.atan2(r23, -r21)

            elif seq == 312:
                self.data[0] = math.atan2(r12, r22)
                self.data[1] = -math.asin(r32)
                self.data[2] = math.atan2(r31, r33)

            elif seq == 313:
                self.data[0] = math.atan2(r13, r23)
                self.data[1] = math.acos(r33)
                self.data[2] = math.atan2(r31, -r32)

            elif seq == 321:
                self.data[0] = math.atan2(-r21, r11)
                self.data[1] = math.asin(r31)
                self.data[2] = math.atan2(-r32, r33)

            elif seq == 323:
                self.data[0] = math.atan2(r23, -r13)
                self.data[1] = math.acos(r33)
                self.data[2] = math.atan2(r32, r31)

            else:
                raise RuntimeError(f'Euler angle sequence {seq} is not a valid Euler angle sequence.')

        else:
            raise RuntimeError(f'seq input type of {type(seq)} cannot be used for Quaternion initialization')


    # Access with [] operators
    def __getitem__(self, key):
        if type(key) == slice:
            return self.data[key]
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 3

            # Check for valid access
            if key < 0 or key >= 3:
                raise IndexError(f'Unable to access element {key} for Euler Angle of length 3.')

            # Get data from storage object
            return self.data[key]

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __setitem__(self, key, value):
        if type(key) == slice:
            self.data[key] = value
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 3

            # Check for valid access
            if key < 0 or key >= 3:
                raise IndexError(f'Unable to access element {key} for Euler Angle of length 3.')

            # Get data from storage object
            self.data[key] = value

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __str__(self):
        return '%d, %f, %f, %f'%(self.seq, self.data[0], self.data[1], self.data[2])

    # Logical operators
    def __eq__(self, other):
        if self.seq != other.seq:
            return False
        elif self.seq == other.seq and np.all(self.data == other.data):
            return True
        else:
            return False

    def __ne__(self, other):
        if self.seq != other.seq:
            return True
        elif self.seq == other.seq and not np.all(self.data == other.data):
            return True
        else:
            return False

###################
# Rotation Matrix #
###################

class RotationMatrix():
    '''Rotation matrix attitude represenation data structure.

    Member variables:
        data (:obj:`np.ndarray`): Rotation matrix data (3x3)

    Notes:

    References:
        1. J. Diebel, _Representing attitude: Euler angles, unit quaternions, and rotation vectors._ Matrix 58(15-16) (2006).
    '''

    def __init__(self, mat):
        if type(mat) == np.ndarray:
            '''Initialize RotationMatrix

            Args:
            ang (:obj:`np.ndarray`, :obj:`Quaternion`, :obj:`EulerAngle`, 
                :obj:`EulerAxis`, :obj:`RotationMatrix`): Object to initialize
                RotationMatrix from. Can be any of the other data types
                defined in the Attitude module.
            '''
            if mat.shape != (3,3):
                raise RuntimeError(f'Matrix size {mat.shape} is incompatible with Rotation Matrix initialization.')

            self.data = np.copy(mat)
        
        elif type(mat) == Quaternion \
          or type(mat) == EulerAngle \
          or type(mat) == EulerAxis:

            # First calculate Quaternion from input
            if type(mat) == EulerAngle or type(mat) == EulerAxis:
                quat = Quaternion(mat)
            else:
                quat = mat

            # Compute Rotation matrix from Quaternion
            qs = quat[0]
            q1 = quat[1]
            q2 = quat[2]
            q3 = quat[3]
            
            # Algorithm
            self.data = np.zeros((3, 3))
            self.data[0, 0] = qs*qs + q1*q1 - q2*q2 - q3*q3
            self.data[0, 1] = 2*q1*q2 + 2*qs*q3
            self.data[0, 2] = 2*q1*q3 - 2*qs*q2
            self.data[1, 0] = 2*q1*q2 - 2*qs*q3
            self.data[1, 1] = qs*qs - q1*q1 + q2*q2 - q3*q3
            self.data[1, 2] = 2*q2*q3 + 2*qs*q1
            self.data[2, 0] = 2*q1*q3 + 2*qs*q2
            self.data[2, 1] = 2*q2*q3 - 2*qs*q1
            self.data[2, 2] = qs*qs - q1*q1 - q2*q2 + q3*q3

    # Access with [] operators
    def __getitem__(self, key):
        if type(key) == slice:
            return self.data[key]
        elif type(key) == tuple:
            return self.data[key]
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 3

            # Check for valid access
            if key < 0 or key >= 3:
                raise IndexError('Unable to access element %d for state of length %d.'%(key, 3))

            # Get data from storage object
            return self.data[key]

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __setitem__(self, key, value):
        if type(key) == slice:
            self.data[key] = value
        if type(key) == tuple:
            self.data[key] = value
        elif type(key) == int:
            # Handle negative indices
            if key < 0:
                key += 3

            # Check for valid access
            if key < 0 or key >= 3:
                raise IndexError('Unable to access element %d for state of length %d.'%(key, 3))

            # Get data from storage object
            self.data[key] = value

        else:
            raise TypeError(f'Invalid argument type: {type(key)}.')

    def __str__(self):
        return self.data.__str__()

    # Logical operators
    def __eq__(self, other):
        return np.all(self.data == other.data)

    def __ne__(self, other):
        return np.all(self.data != other.data)

    # Arithmetic operations
    def __matmul__(self, other):
        if type(other) == np.ndarray:
            if other.shape == (3, 3) or other.shape == (3):
                return self.data @ other
            else:
                raise RuntimeError(f'Invalid dimension of right-hand input: {other.shape}. Must be a (3, 3) or (3) array.')

        elif type(other) == RotationMatrix:
            return RotationMatrix(self.data @ other.data)
        else:
            raise RuntimeError(f'Cannot multiple RotationMatrix with object of type {type(other)}')