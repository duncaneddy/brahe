"""
Python (Brahe) benchmark implementations.

Dispatches benchmark tasks to the appropriate implementation function.
"""

from benchmarks.comparative.implementations.python.coordinates import (
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from benchmarks.comparative.implementations.python.orbits import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)
from benchmarks.comparative.results import TaskResult

# Task name -> implementation function
_DISPATCH_TABLE: dict = {
    "coordinates.geodetic_to_ecef": geodetic_to_ecef,
    "coordinates.ecef_to_geodetic": ecef_to_geodetic,
    "orbits.keplerian_to_cartesian": keplerian_to_cartesian,
    "orbits.cartesian_to_keplerian": cartesian_to_keplerian,
}


def dispatch(input_data: dict) -> TaskResult:
    """Dispatch a task to the appropriate Python implementation."""
    task_name = input_data["task"]
    func = _DISPATCH_TABLE.get(task_name)
    if func is None:
        raise ValueError(f"No Python implementation for task: {task_name}")
    return func(input_data["params"], input_data["iterations"])
