from benchmarks.comparative.tasks.access_tasks import (
    Sgp4AccessTask,
)
from benchmarks.comparative.tasks.attitude_tasks import (
    EulerAngleToQuaternionTask,
    QuaternionToEulerAngleTask,
    QuaternionToRotationMatrixTask,
    RotationMatrixToQuaternionTask,
)
from benchmarks.comparative.tasks.coordinates_tasks import (
    EcefToAzelTask,
    EcefToGeocentricTask,
    EcefToGeodeticTask,
    GeocentricToEcefTask,
    GeodeticToEcefTask,
)
from benchmarks.comparative.tasks.force_model_tasks import (
    AccelPointMassGravityTask,
    AccelSphericalHarmonics20Task,
    AccelSphericalHarmonics80Task,
    AccelThirdBodyMoonTask,
    AccelThirdBodySunTask,
)
from benchmarks.comparative.tasks.frames_tasks import (
    StateEcefToEciTask,
    StateEciToEcefTask,
)
from benchmarks.comparative.tasks.orbits_tasks import (
    CartesianToKeplerianTask,
    KeplerianToCartesianTask,
)
from benchmarks.comparative.tasks.propagation_tasks import (
    KeplerianSingleTask,
    KeplerianTrajectoryTask,
    NumericalRk4Grav5x5Task,
    NumericalRk4Grav20x20SunMoonTask,
    NumericalRk4Grav80x80FullTask,
    NumericalTwobodyTask,
    Sgp4SingleTask,
    Sgp4TrajectoryTask,
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
    NumericalRk4Grav5x5Task(),
    NumericalRk4Grav20x20SunMoonTask(),
    NumericalRk4Grav80x80FullTask(),
    # Force model (function-level acceleration)
    AccelPointMassGravityTask(),
    AccelSphericalHarmonics20Task(),
    AccelSphericalHarmonics80Task(),
    AccelThirdBodySunTask(),
    AccelThirdBodyMoonTask(),
    # Access
    Sgp4AccessTask(),
]

__all__ = [
    "ALL_TASKS",
    "AccelPointMassGravityTask",
    "AccelSphericalHarmonics20Task",
    "AccelSphericalHarmonics80Task",
    "AccelThirdBodyMoonTask",
    "AccelThirdBodySunTask",
    "CartesianToKeplerianTask",
    "EcefToAzelTask",
    "EcefToGeocentricTask",
    "EcefToGeodeticTask",
    "EpochCreationTask",
    "EulerAngleToQuaternionTask",
    "GeocentricToEcefTask",
    "GeodeticToEcefTask",
    "KeplerianSingleTask",
    "KeplerianToCartesianTask",
    "KeplerianTrajectoryTask",
    "NumericalRk4Grav5x5Task",
    "NumericalRk4Grav20x20SunMoonTask",
    "NumericalRk4Grav80x80FullTask",
    "NumericalTwobodyTask",
    "QuaternionToEulerAngleTask",
    "QuaternionToRotationMatrixTask",
    "RotationMatrixToQuaternionTask",
    "Sgp4AccessTask",
    "Sgp4SingleTask",
    "Sgp4TrajectoryTask",
    "StateEcefToEciTask",
    "StateEciToEcefTask",
    "UtcToGpsTask",
    "UtcToTaiTask",
    "UtcToTtTask",
    "UtcToUt1Task",
]
