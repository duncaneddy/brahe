"""Basilisk (bsk) benchmark implementations.

Basilisk is invoked in-process by the runner (no subprocess). Imports of the
Basilisk package itself happen inside the submodule functions on first
dispatch; the runner is responsible for catching ImportError and emitting the
"not installed" skip message.
"""

from benchmarks.comparative.implementations.basilisk.attitude import (
    euler_angle_to_quaternion,
    quaternion_to_euler_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from benchmarks.comparative.implementations.basilisk.coordinates import (
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from benchmarks.comparative.implementations.basilisk.frames import (
    state_ecef_to_eci,
    state_eci_to_ecef,
)
from benchmarks.comparative.implementations.basilisk.orbits import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)
from benchmarks.comparative.implementations.basilisk.propagation import (
    numerical_rk4_grav5x5,
    numerical_rk4_grav20x20_sun_moon,
    numerical_rk4_grav80x80_full,
    numerical_twobody,
)
from benchmarks.comparative.results import TaskResult

_DISPATCH_TABLE: dict = {
    "attitude.quaternion_to_rotation_matrix": quaternion_to_rotation_matrix,
    "attitude.rotation_matrix_to_quaternion": rotation_matrix_to_quaternion,
    "attitude.quaternion_to_euler_angle": quaternion_to_euler_angle,
    "attitude.euler_angle_to_quaternion": euler_angle_to_quaternion,
    "coordinates.geodetic_to_ecef": geodetic_to_ecef,
    "coordinates.ecef_to_geodetic": ecef_to_geodetic,
    "frames.state_eci_to_ecef": state_eci_to_ecef,
    "frames.state_ecef_to_eci": state_ecef_to_eci,
    "orbits.keplerian_to_cartesian": keplerian_to_cartesian,
    "orbits.cartesian_to_keplerian": cartesian_to_keplerian,
    "propagation.numerical_rk4_grav5x5": numerical_rk4_grav5x5,
    "propagation.numerical_rk4_grav20x20_sun_moon": numerical_rk4_grav20x20_sun_moon,
    "propagation.numerical_rk4_grav80x80_full": numerical_rk4_grav80x80_full,
    "propagation.numerical_twobody": numerical_twobody,
}


def dispatch(input_data: dict) -> TaskResult:
    """Dispatch a task to the appropriate Basilisk implementation."""
    task_name = input_data["task"]
    func = _DISPATCH_TABLE.get(task_name)
    if func is None:
        raise ValueError(f"No Basilisk implementation for task: {task_name}")
    return func(input_data["params"], input_data["iterations"])
