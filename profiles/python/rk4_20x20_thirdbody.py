"""Profile: RK4 numerical propagation, 20x20 spherical-harmonic gravity +
sun/moon third-body.

Python mirror of profiles/rust/src/bin/rk4_20x20_thirdbody.rs. Mid-complexity
baseline.
"""

from __future__ import annotations

import brahe as bh

from _common import (
    default_leo_state,
    duration_from_env,
    run_until_elapsed,
    setup_providers,
)


def main() -> None:
    setup_providers()
    epoch, state = default_leo_state()
    duration_s = duration_from_env()

    force = bh.ForceModelConfig(
        gravity=bh.GravityConfiguration(degree=20, order=20),
        third_body=bh.ThirdBodyConfiguration(
            ephemeris_source=bh.EphemerisSource.DE440s,
            bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
        ),
    )
    prop_cfg = bh.NumericalPropagationConfig.default()

    def workload() -> None:
        prop = bh.NumericalOrbitPropagator(epoch, state, prop_cfg, force)
        prop.propagate_to(epoch + 86400.0)
        _ = prop.current_state()

    iters = run_until_elapsed(duration_s, workload)
    print(f"rk4_20x20_thirdbody: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
