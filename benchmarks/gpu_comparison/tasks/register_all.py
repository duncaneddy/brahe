"""Importing this module registers every task with the global registry.

Kept separate from ``tasks/__init__.py`` to avoid a circular import between
``registry`` (which depends on ``tasks.base``) and the task implementations
(which depend on ``registry``).
"""

from benchmarks.gpu_comparison.registry import register
from benchmarks.gpu_comparison.tasks.coordinates_tasks import (
    EnzToAzelTask,
    GeodeticToEcefTask,
    KeplerianToCartesianTask,
)
from benchmarks.gpu_comparison.tasks.propagation_tasks import (
    NumericalTwobodyJ2Task,
    Sgp4IssSweepTask,
)
from benchmarks.gpu_comparison.tasks.force_model_tasks import ForceModelGrav5x5Task
from benchmarks.gpu_comparison.tasks.frames_tasks import GcrfToItrfStateTask
from benchmarks.gpu_comparison.tasks.time_tasks import UtcMjdToTtMjdTask

register(GeodeticToEcefTask)
register(KeplerianToCartesianTask)
register(EnzToAzelTask)
register(Sgp4IssSweepTask)
register(NumericalTwobodyJ2Task)
register(UtcMjdToTtMjdTask)
register(GcrfToItrfStateTask)
register(ForceModelGrav5x5Task)
