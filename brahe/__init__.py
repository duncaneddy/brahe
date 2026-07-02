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

from typing import TYPE_CHECKING

# Import core native module
from brahe import _brahe

# Re-export PanicException for testing and BraheError for error handling
from brahe._brahe import BraheError, PanicException

# Import and create submodules
from brahe import (
    time,
    orbits,
    propagators,
    coordinates,
    frames,
    eop,
    space_weather,
    attitude,
    trajectories,
    constants,
    datasets,
    access,
    relative_motion,
    math,
    integrators,
    utils,
    logging,
    orbit_dynamics,
    events,
    spacetrack,
    celestrak,
    estimation,
    spice,
)

# Re-export everything from submodules
from brahe.time import *
from brahe.orbits import *
from brahe.propagators import *
from brahe.coordinates import *
from brahe.frames import *
from brahe.eop import *
from brahe.space_weather import *
from brahe.attitude import *
from brahe.trajectories import *
from brahe.constants import *
from brahe.access import *
from brahe.relative_motion import *
from brahe.math import *
from brahe.integrators import *
from brahe.utils import *
from brahe.datasets import *
from brahe.orbit_dynamics import *
from brahe.events import *
from brahe.spacetrack import *
from brahe.estimation import *
from brahe.spice import *

# Define what's available when doing 'from brahe import *'
__all__ = [
    # Submodules
    "time",
    "orbits",
    "propagators",
    "coordinates",
    "frames",
    "eop",
    "space_weather",
    "attitude",
    "trajectories",
    "constants",
    "datasets",
    "access",
    "relative_motion",
    "math",
    "integrators",
    "utils",
    "plots",  # noqa: F405
    "logging",
    "orbit_dynamics",
    "events",
    "spacetrack",
    "celestrak",
    "estimation",
    "spice",
    # Exceptions
    "BraheError",
    "PanicException",
]

# Extend __all__ with exports from submodules
__all__.extend(time.__all__)
__all__.extend(orbits.__all__)
__all__.extend(propagators.__all__)
__all__.extend(coordinates.__all__)
__all__.extend(frames.__all__)
__all__.extend(eop.__all__)
__all__.extend(space_weather.__all__)
__all__.extend(attitude.__all__)
__all__.extend(trajectories.__all__)
__all__.extend(constants.__all__)
__all__.extend(access.__all__)
__all__.extend(relative_motion.__all__)
__all__.extend(math.__all__)
__all__.extend(integrators.__all__)
__all__.extend(utils.__all__)
__all__.extend(datasets.__all__)
__all__.extend(orbit_dynamics.__all__)
__all__.extend(events.__all__)
__all__.extend(spacetrack.__all__)
__all__.extend(estimation.__all__)
__all__.extend(spice.__all__)

# Import version from native module (set from Cargo.toml at build time)
__version__ = _brahe.__version__


# Static re-exports of the lazily-forwarded plot symbols (resolved at runtime by
# __getattr__ below) so type checkers, IDEs, and the docs builder (griffe /
# mkdocstrings) can see ``brahe.plot_*``. Never executed at runtime, so
# ``import brahe`` still never imports the visualization stack. Mirrors the
# runtime forwarding from ``brahe.plots``; keep in sync with brahe.plots.
if TYPE_CHECKING:
    from brahe.plots import (
        plot_access_elevation as plot_access_elevation,
        plot_access_elevation_azimuth as plot_access_elevation_azimuth,
        plot_access_polar as plot_access_polar,
        plot_cartesian_trajectory as plot_cartesian_trajectory,
        plot_estimator_marginal as plot_estimator_marginal,
        plot_estimator_marginal_from_arrays as plot_estimator_marginal_from_arrays,
        plot_estimator_state_error as plot_estimator_state_error,
        plot_estimator_state_error_from_arrays as plot_estimator_state_error_from_arrays,
        plot_estimator_state_error_grid as plot_estimator_state_error_grid,
        plot_estimator_state_error_grid_from_arrays as plot_estimator_state_error_grid_from_arrays,
        plot_estimator_state_value as plot_estimator_state_value,
        plot_estimator_state_value_from_arrays as plot_estimator_state_value_from_arrays,
        plot_estimator_state_value_grid as plot_estimator_state_value_grid,
        plot_estimator_state_value_grid_from_arrays as plot_estimator_state_value_grid_from_arrays,
        plot_gabbard_diagram as plot_gabbard_diagram,
        plot_groundtrack as plot_groundtrack,
        plot_keplerian_trajectory as plot_keplerian_trajectory,
        plot_measurement_residual as plot_measurement_residual,
        plot_measurement_residual_from_arrays as plot_measurement_residual_from_arrays,
        plot_measurement_residual_grid as plot_measurement_residual_grid,
        plot_measurement_residual_grid_from_arrays as plot_measurement_residual_grid_from_arrays,
        plot_measurement_residual_rms as plot_measurement_residual_rms,
        plot_measurement_residual_rms_from_arrays as plot_measurement_residual_rms_from_arrays,
        plot_trajectory_3d as plot_trajectory_3d,
        split_ground_track_at_antimeridian as split_ground_track_at_antimeridian,
    )


def __getattr__(name):
    """Lazily resolve the optional ``plots`` subpackage and its exports.

    Keeping this lazy means ``import brahe`` never imports the visualization
    dependency stack. The documented ``bh.plot_groundtrack(...)`` /
    ``from brahe import plot_groundtrack`` patterns resolve here on first access;
    a missing ``brahe[plots]`` extra surfaces as a clear ImportError from
    ``brahe.plots``. Unknown names raise AttributeError as usual.
    """
    import importlib

    plots = importlib.import_module("brahe.plots")
    if name == "plots":
        return plots
    if name in plots.__all__:
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include the lazily-forwarded plot symbols in ``dir(brahe)`` for discoverability.

    Plot names are deliberately kept out of ``__all__`` (so ``from brahe import *``
    stays free of the visualization stack), but listing them here keeps interactive
    autocomplete and ``dir()`` working when ``brahe[plots]`` is installed. Importing
    ``brahe.plots`` is cheap — it imports no third-party libraries.
    """
    import importlib

    plots = importlib.import_module("brahe.plots")
    return sorted(set(list(globals().keys()) + __all__ + list(plots.__all__)))
