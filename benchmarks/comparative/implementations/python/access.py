"""
Python (Brahe) access computation benchmarks.
"""

import brahe

from benchmarks.comparative.implementations.python.base import (
    ensure_eop,
    time_iterations,
)
from benchmarks.comparative.results import TaskResult


def sgp4_access(params: dict, iterations: int) -> TaskResult:
    """Benchmark SGP4 access computation for 100 ground locations."""
    ensure_eop()

    line1 = params["line1"]
    line2 = params["line2"]
    locations_data = params["locations"]
    min_el = params["min_elevation_deg"]
    duration = params["search_duration_seconds"]

    prop = brahe.SGPPropagator.from_tle(line1, line2, 60.0)
    search_start = prop.epoch
    search_end = search_start + duration

    constraint = brahe.ElevationConstraint(min_elevation_deg=min_el)
    config = brahe.AccessSearchConfig(parallel=False)

    # Build PointLocation objects
    point_locations = []
    for loc in locations_data:
        point_locations.append(
            brahe.PointLocation(lon=loc["lon"], lat=loc["lat"], alt=loc["alt"])
        )

    def run():
        all_windows = []
        for pl in point_locations:
            windows = brahe.location_accesses(
                pl,
                prop,
                search_start,
                search_end,
                constraint,
                config=config,
            )
            loc_windows = []
            for w in windows:
                loc_windows.append(
                    {
                        "start_jd": w.window_open.jd(),
                        "end_jd": w.window_close.jd(),
                    }
                )
            all_windows.append(loc_windows)
        return all_windows

    times, results = time_iterations(run, iterations)

    return TaskResult(
        task_name="access.sgp4_access",
        language="python",
        library="brahe",
        iterations=iterations,
        times_seconds=times,
        results=results,
        metadata={
            "library": "brahe",
            "language": "python",
            "version": getattr(brahe, "__version__", "unknown"),
        },
    )
