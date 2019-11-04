import typing as _typing
import numpy as _np
import logging as _logging

# Setup logging
logger = _logging.getLogger(__name__)

# Define common array-like type
AbstractArray = _typing.NewType('AbstractArray', _typing.Union[tuple, list, _np.ndarray])