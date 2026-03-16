"""
Task registry: discovers and filters benchmark tasks.
"""

from benchmarks.comparative.tasks import ALL_TASKS
from benchmarks.comparative.tasks.base import BenchmarkTask


def get_all_tasks() -> list[BenchmarkTask]:
    """Return all registered benchmark tasks."""
    return list(ALL_TASKS)


def filter_tasks(
    module: str | None = None,
    task_name: str | None = None,
    language: str | None = None,
) -> list[BenchmarkTask]:
    """Filter tasks by module, name, or language support."""
    tasks = get_all_tasks()

    if module:
        tasks = [t for t in tasks if t.module == module]

    if task_name:
        tasks = [t for t in tasks if t.name == task_name]

    if language:
        tasks = [t for t in tasks if language in t.languages]

    return tasks
