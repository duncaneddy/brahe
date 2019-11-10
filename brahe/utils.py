import logging as _logging
import typing as _typing
import numpy as np
import numba as numba

# Setup logging
logger = _logging.getLogger(__name__)

# Define common array-like type
AbstractArray = _typing.NewType('AbstractArray', _typing.Union[tuple, list, np.ndarray])

###############
# Mathematics #
###############

@numba.jit(nopython=True, cache=True)
def kron_delta(a:float, b:float) -> int:
    """Cannonical Kronecker Delta function.

    Returns 1 if inputs are equal returns 0 otherwise.

    Args:
        (:obj:`float`): First input argument
        (:obj:`float`): Second input argument 

    Returns:
        (:obj:`int`) Kronecker delta result {0, 1}
    """
    if a == b:
        return 1
    else:
        return 0