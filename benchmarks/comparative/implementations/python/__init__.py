"""
Python (Brahe) benchmark implementations.

Dispatches benchmark tasks to the appropriate implementation function.
"""

from benchmarks.comparative.implementations.python.attitude import (
    euler_angle_to_quaternion,
    quaternion_to_euler_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from benchmarks.comparative.implementations.python.coordinates import (
    ecef_to_azel,
    ecef_to_geocentric,
    ecef_to_geodetic,
    geocentric_to_ecef,
    geodetic_to_ecef,
)
from benchmarks.comparative.implementations.python.frames import (
    state_ecef_to_eci,
    state_eci_to_ecef,
)
from benchmarks.comparative.implementations.python.propagation import (
    keplerian_single,
    keplerian_trajectory,
    numerical_twobody,
    sgp4_single,
    sgp4_trajectory,
)
from benchmarks.comparative.implementations.python.orbits import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)
from benchmarks.comparative.implementations.python.time_bench import (
    epoch_creation,
    utc_to_gps,
    utc_to_tai,
    utc_to_tt,
    utc_to_ut1,
)
from benchmarks.comparative.results import TaskResult

# Task name -> implementation function
_DISPATCH_TABLE: dict = {
    "attitude.quaternion_to_rotation_matrix": quaternion_to_rotation_matrix,
    "attitude.rotation_matrix_to_quaternion": rotation_matrix_to_quaternion,
    "attitude.quaternion_to_euler_angle": quaternion_to_euler_angle,
    "attitude.euler_angle_to_quaternion": euler_angle_to_quaternion,
    "coordinates.geodetic_to_ecef": geodetic_to_ecef,
    "coordinates.ecef_to_geodetic": ecef_to_geodetic,
    "coordinates.geocentric_to_ecef": geocentric_to_ecef,
    "coordinates.ecef_to_geocentric": ecef_to_geocentric,
    "coordinates.ecef_to_azel": ecef_to_azel,
    "frames.state_eci_to_ecef": state_eci_to_ecef,
    "frames.state_ecef_to_eci": state_ecef_to_eci,
    "orbits.keplerian_to_cartesian": keplerian_to_cartesian,
    "orbits.cartesian_to_keplerian": cartesian_to_keplerian,
    "time.epoch_creation": epoch_creation,
    "time.utc_to_tai": utc_to_tai,
    "time.utc_to_tt": utc_to_tt,
    "time.utc_to_gps": utc_to_gps,
    "time.utc_to_ut1": utc_to_ut1,
    "propagation.keplerian_single": keplerian_single,
    "propagation.keplerian_trajectory": keplerian_trajectory,
    "propagation.sgp4_single": sgp4_single,
    "propagation.sgp4_trajectory": sgp4_trajectory,
    "propagation.numerical_twobody": numerical_twobody,
}


def dispatch(input_data: dict) -> TaskResult:
    """Dispatch a task to the appropriate Python implementation."""
    task_name = input_data["task"]
    func = _DISPATCH_TABLE.get(task_name)
    if func is None:
        raise ValueError(f"No Python implementation for task: {task_name}")
    return func(input_data["params"], input_data["iterations"])
