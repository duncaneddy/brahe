"""Profile: SGP4 + access window computation.

ISS-like TLE, San Francisco ground location, 7-day window, 10 deg elevation
constraint. Python mirror of profiles/rust/src/bin/sgp4_access.rs.
"""

from __future__ import annotations

import brahe as bh

from _common import (
    DEFAULT_ISS_TLE_LINE1,
    DEFAULT_ISS_TLE_LINE2,
    duration_from_env,
    run_until_elapsed,
    setup_providers,
)


def main() -> None:
    setup_providers()
    duration_s = duration_from_env()

    # Build once outside the hot loop: TLE parsing and constraint construction
    # are negligible relative to the 7-day access search.
    propagator = bh.SGPPropagator.from_tle(
        DEFAULT_ISS_TLE_LINE1, DEFAULT_ISS_TLE_LINE2, 60.0
    ).with_name("ISS")

    location = bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco")
    epoch_start = bh.Epoch.from_datetime(
        2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
    )
    epoch_end = epoch_start + 7.0 * 86400.0
    constraint = bh.ElevationConstraint(min_elevation_deg=10.0, max_elevation_deg=None)

    def workload() -> None:
        windows = bh.location_accesses(
            location, propagator, epoch_start, epoch_end, constraint
        )
        _ = windows

    iters = run_until_elapsed(duration_s, workload)
    print(f"sgp4_access: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
