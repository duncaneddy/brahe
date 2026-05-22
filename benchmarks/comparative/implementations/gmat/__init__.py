"""GMAT (gmatpy) benchmark implementations.

GMAT is invoked in-process via the `gmatpy` Python package shipped at
`<GMAT_ROOT_PATH>/bin/gmatpy/`. Detection and initialization live in
`base.py:_ensure_gmat()`. Imports of `gmatpy` happen lazily on first
dispatch; the runner is responsible for catching ImportError and emitting
the "GMAT not ready" skip message.

See `docs/superpowers/specs/2026-05-19-gmat-benchmark-baseline-design.md`.
"""

from benchmarks.comparative.results import TaskResult
from benchmarks.comparative.implementations.gmat.time import (
    epoch_creation,
    utc_to_gps,
    utc_to_tai,
    utc_to_tt,
    utc_to_ut1,
)
from benchmarks.comparative.implementations.gmat.orbits import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)
from benchmarks.comparative.implementations.gmat.attitude import (
    euler_angle_to_quaternion,
    quaternion_to_euler_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from benchmarks.comparative.implementations.gmat.frames import (
    state_eci_to_ecef,
    state_ecef_to_eci,
)
from benchmarks.comparative.implementations.gmat.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    geocentric_to_ecef,
    ecef_to_geocentric,
)
from benchmarks.comparative.implementations.gmat.force_model import (
    accel_point_mass_gravity,
    accel_spherical_harmonics_20,
    accel_spherical_harmonics_80,
    accel_third_body_sun,
    accel_third_body_moon,
)
from benchmarks.comparative.implementations.gmat.propagation import (
    keplerian_single,
    keplerian_trajectory,
    numerical_twobody,
    numerical_rk4_grav5x5,
    numerical_rk4_grav20x20_sun_moon,
    numerical_rk4_grav80x80_full,
    sgp4_single,
    sgp4_trajectory,
)
from benchmarks.comparative.implementations.gmat.access import (
    sgp4_access,
)

_DISPATCH_TABLE: dict = {
    "time.epoch_creation": epoch_creation,
    "time.utc_to_tai": utc_to_tai,
    "time.utc_to_tt": utc_to_tt,
    "time.utc_to_gps": utc_to_gps,
    "time.utc_to_ut1": utc_to_ut1,
    "orbits.keplerian_to_cartesian": keplerian_to_cartesian,
    "orbits.cartesian_to_keplerian": cartesian_to_keplerian,
    "attitude.quaternion_to_rotation_matrix": quaternion_to_rotation_matrix,
    "attitude.rotation_matrix_to_quaternion": rotation_matrix_to_quaternion,
    "attitude.quaternion_to_euler_angle": quaternion_to_euler_angle,
    "attitude.euler_angle_to_quaternion": euler_angle_to_quaternion,
    "frames.state_eci_to_ecef": state_eci_to_ecef,
    "frames.state_ecef_to_eci": state_ecef_to_eci,
    "coordinates.geodetic_to_ecef": geodetic_to_ecef,
    "coordinates.ecef_to_geodetic": ecef_to_geodetic,
    "coordinates.geocentric_to_ecef": geocentric_to_ecef,
    "coordinates.ecef_to_geocentric": ecef_to_geocentric,
    "force_model.accel_point_mass_gravity": accel_point_mass_gravity,
    "force_model.accel_spherical_harmonics_20": accel_spherical_harmonics_20,
    "force_model.accel_spherical_harmonics_80": accel_spherical_harmonics_80,
    "force_model.accel_third_body_sun": accel_third_body_sun,
    "force_model.accel_third_body_moon": accel_third_body_moon,
    "propagation.keplerian_single": keplerian_single,
    "propagation.keplerian_trajectory": keplerian_trajectory,
    "propagation.numerical_twobody": numerical_twobody,
    "propagation.numerical_rk4_grav5x5": numerical_rk4_grav5x5,
    "propagation.numerical_rk4_grav20x20_sun_moon": numerical_rk4_grav20x20_sun_moon,
    "propagation.numerical_rk4_grav80x80_full": numerical_rk4_grav80x80_full,
    "propagation.sgp4_single": sgp4_single,
    "propagation.sgp4_trajectory": sgp4_trajectory,
    "access.sgp4_access": sgp4_access,
}


def dispatch(input_data: dict) -> TaskResult:
    """Dispatch a task to the appropriate GMAT implementation.

    Raises ImportError if GMAT is not available (caught by the runner).
    Raises ValueError if the task name has no GMAT implementation registered.
    """
    from benchmarks.comparative.implementations.gmat.base import _ensure_gmat

    _ensure_gmat()
    task_name = input_data["task"]
    func = _DISPATCH_TABLE.get(task_name)
    if func is None:
        raise ValueError(f"No GMAT implementation for task: {task_name}")
    return func(input_data["params"], input_data["iterations"])
