from benchmarks.comparative.tasks.coordinates_tasks import (
    EcefToGeodeticTask,
    GeodeticToEcefTask,
)
from benchmarks.comparative.tasks.orbits_tasks import (
    CartesianToKeplerianTask,
    KeplerianToCartesianTask,
)

ALL_TASKS = [
    GeodeticToEcefTask(),
    EcefToGeodeticTask(),
    KeplerianToCartesianTask(),
    CartesianToKeplerianTask(),
]

__all__ = [
    "ALL_TASKS",
    "GeodeticToEcefTask",
    "EcefToGeodeticTask",
    "KeplerianToCartesianTask",
    "CartesianToKeplerianTask",
]
