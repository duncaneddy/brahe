"""Profile: SGP4 propagation of an ISS-like TLE over 24h at 60s steps.

Python mirror of profiles/rust/src/bin/sgp4_trajectory.rs. Run via:
    just profile-python sgp4_trajectory
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

    def workload() -> None:
        prop = bh.SGPPropagator.from_tle(
            DEFAULT_ISS_TLE_LINE1,
            DEFAULT_ISS_TLE_LINE2,
            60.0,
        )
        target = prop.current_epoch() + 86400.0
        prop.propagate_to(target)
        # Reference the result so the optimizer can't elide the call.
        _ = prop.current_state()

    iters = run_until_elapsed(duration_s, workload)
    print(f"sgp4_trajectory: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
