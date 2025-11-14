"""
Brahe - Satellite Dynamics and Astrodynamics Library

A high-performance library for orbital mechanics, time systems, coordinate transformations,
and attitude representations. Brahe provides both Rust and Python interfaces for satellite
dynamics computations.

The library is organized into submodules that mirror the Rust core structure:
- time: Time systems, epochs, and conversions
- orbits: Orbital mechanics and TLE handling
- propagators: Orbit propagators (SGP4, Keplerian)
- coordinates: Coordinate system transformations
- frames: Reference frame transformations (ECI/ECEF)
- eop: Earth Orientation Parameters
- attitude: Attitude representations (quaternions, Euler angles, etc.)
- trajectories: Trajectory containers and interpolation
- access: Access window computation and constraints
- plots: Visualization tools (matplotlib/plotly backends)

All functionality is re-exported at the top level for convenience, so you can use either:
    from brahe import Epoch
    from brahe.time import Epoch

    from brahe import plot_groundtrack
    from brahe.plots import plot_groundtrack
"""

# Import core native module
from brahe import _brahe

# Re-export PanicException for testing
from brahe._brahe import PanicException

# Import and create submodules
from brahe import (
    time,
    orbits,
    propagators,
    coordinates,
    frames,
    eop,
    attitude,
    trajectories,
    constants,
    datasets,
    access,
    relative_motion,
    utils,
    plots,
    logging,
)

# Re-export everything from submodules
from brahe.time import *
from brahe.orbits import *
from brahe.propagators import *
from brahe.coordinates import *
from brahe.frames import *
from brahe.eop import *
from brahe.attitude import *
from brahe.trajectories import *
from brahe.constants import *
from brahe.access import *
from brahe.relative_motion import *
from brahe.utils import *
from brahe.plots import *
from brahe.datasets import *

# Define what's available when doing 'from brahe import *'
__all__ = [
    # Submodules
    "time",
    "orbits",
    "propagators",
    "coordinates",
    "frames",
    "eop",
    "attitude",
    "trajectories",
    "constants",
    "datasets",
    "access",
    "relative_motion",
    "utils",
    "plots",
    "logging",
    # Testing
    "PanicException",
]

# Extend __all__ with exports from submodules
__all__.extend(time.__all__)
__all__.extend(orbits.__all__)
__all__.extend(propagators.__all__)
__all__.extend(coordinates.__all__)
__all__.extend(frames.__all__)
__all__.extend(eop.__all__)
__all__.extend(attitude.__all__)
__all__.extend(trajectories.__all__)
__all__.extend(constants.__all__)
__all__.extend(access.__all__)
__all__.extend(relative_motion.__all__)
__all__.extend(utils.__all__)
__all__.extend(plots.__all__)
__all__.extend(datasets.__all__)

# Import version from native module (set from Cargo.toml at build time)
__version__ = _brahe.__version__
