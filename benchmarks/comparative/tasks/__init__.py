from benchmarks.comparative.tasks.coordinates_tasks import (
    EcefToAzelTask,
    EcefToGeocentricTask,
    EcefToGeodeticTask,
    GeocentricToEcefTask,
    GeodeticToEcefTask,
)
from benchmarks.comparative.tasks.propagation_tasks import (
    KeplerianSingleTask,
    KeplerianTrajectoryTask,
    NumericalTwobodyTask,
    Sgp4SingleTask,
    Sgp4TrajectoryTask,
)
from benchmarks.comparative.tasks.orbits_tasks import (
    CartesianToKeplerianTask,
    KeplerianToCartesianTask,
)
from benchmarks.comparative.tasks.attitude_tasks import (
    EulerAngleToQuaternionTask,
    QuaternionToEulerAngleTask,
    QuaternionToRotationMatrixTask,
    RotationMatrixToQuaternionTask,
)
from benchmarks.comparative.tasks.frames_tasks import (
    StateEcefToEciTask,
    StateEciToEcefTask,
)
from benchmarks.comparative.tasks.time_tasks import (
    EpochCreationTask,
    UtcToGpsTask,
    UtcToTaiTask,
    UtcToTtTask,
    UtcToUt1Task,
)

ALL_TASKS = [
    # Coordinates
    GeodeticToEcefTask(),
    EcefToGeodeticTask(),
    GeocentricToEcefTask(),
    EcefToGeocentricTask(),
    EcefToAzelTask(),
    # Orbits
    KeplerianToCartesianTask(),
    CartesianToKeplerianTask(),
    # Attitude
    QuaternionToRotationMatrixTask(),
    RotationMatrixToQuaternionTask(),
    QuaternionToEulerAngleTask(),
    EulerAngleToQuaternionTask(),
    # Frames
    StateEciToEcefTask(),
    StateEcefToEciTask(),
    # Time
    EpochCreationTask(),
    UtcToTaiTask(),
    UtcToTtTask(),
    UtcToGpsTask(),
    UtcToUt1Task(),
    # Propagation
    KeplerianSingleTask(),
    KeplerianTrajectoryTask(),
    Sgp4SingleTask(),
    Sgp4TrajectoryTask(),
    NumericalTwobodyTask(),
]

__all__ = [
    "ALL_TASKS",
    "GeodeticToEcefTask",
    "EcefToGeodeticTask",
    "GeocentricToEcefTask",
    "EcefToGeocentricTask",
    "EcefToAzelTask",
    "KeplerianToCartesianTask",
    "CartesianToKeplerianTask",
    "QuaternionToRotationMatrixTask",
    "RotationMatrixToQuaternionTask",
    "QuaternionToEulerAngleTask",
    "EulerAngleToQuaternionTask",
    "StateEciToEcefTask",
    "StateEcefToEciTask",
    "EpochCreationTask",
    "UtcToTaiTask",
    "UtcToTtTask",
    "UtcToGpsTask",
    "UtcToUt1Task",
    "KeplerianSingleTask",
    "KeplerianTrajectoryTask",
    "Sgp4SingleTask",
    "Sgp4TrajectoryTask",
    "NumericalTwobodyTask",
]
