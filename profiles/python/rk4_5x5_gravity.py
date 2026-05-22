"""Profile: RK4 numerical propagation, 5x5 spherical-harmonic gravity, no
other perturbations.

Python mirror of profiles/rust/src/bin/rk4_5x5_gravity.rs. Useful for
isolating the cost of low-degree spherical harmonic evaluation.
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
        gravity=bh.GravityConfiguration(degree=5, order=5),
    )
    prop_cfg = bh.NumericalPropagationConfig.default()

    def workload() -> None:
        prop = bh.NumericalOrbitPropagator(epoch, state, prop_cfg, force)
        prop.propagate_to(epoch + 86400.0)
        _ = prop.current_state()

    iters = run_until_elapsed(duration_s, workload)
    print(f"rk4_5x5_gravity: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
