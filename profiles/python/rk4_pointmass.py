"""Profile: RK4 numerical propagation, point-mass gravity only.

Python mirror of profiles/rust/src/bin/rk4_pointmass.rs. No perturbations
(no third body, no drag, no SRP, no relativity). CPU baseline — the
flamegraph should be dominated by integrator stage evaluations and the
point-mass force evaluation.
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
        gravity=bh.GravityConfiguration(),
    )
    prop_cfg = bh.NumericalPropagationConfig.default()

    def workload() -> None:
        prop = bh.NumericalOrbitPropagator(epoch, state, prop_cfg, force)
        prop.propagate_to(epoch + 86400.0)
        _ = prop.current_state()

    iters = run_until_elapsed(duration_s, workload)
    print(f"rk4_pointmass: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
