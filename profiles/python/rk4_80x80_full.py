"""Profile: RK4 numerical propagation, full force model.

80x80 spherical-harmonic gravity + sun/moon third-body + NRLMSISE-00 drag +
SRP. Python mirror of profiles/rust/src/bin/rk4_80x80_full.rs. Heaviest
baseline; matches the conservative-forces case in
benchmarks/propagator_benchmarks.rs but adds drag and SRP for full fidelity.
"""

from __future__ import annotations

import numpy as np

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

    # Parameter vector layout (DefaultParameterLayout):
    #   [mass, drag_area, Cd, srp_area, Cr]
    params = np.array([1000.0, 10.0, 2.2, 10.0, 1.3])

    force = bh.ForceModelConfig(
        gravity=bh.GravityConfiguration(degree=80, order=80),
        drag=bh.DragConfiguration(
            model=bh.AtmosphericModel.NRLMSISE00,
            area=bh.ParameterSource.parameter_index(1),
            cd=bh.ParameterSource.parameter_index(2),
        ),
        srp=bh.SolarRadiationPressureConfiguration(
            area=bh.ParameterSource.parameter_index(3),
            cr=bh.ParameterSource.parameter_index(4),
            eclipse_model=bh.EclipseModel.CONICAL,
        ),
        third_body=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
        mass=bh.ParameterSource.parameter_index(0),
    )
    prop_cfg = bh.NumericalPropagationConfig.default()

    def workload() -> None:
        prop = bh.NumericalOrbitPropagator(epoch, state, prop_cfg, force, params)
        prop.propagate_to(epoch + 86400.0)
        _ = prop.current_state()

    iters = run_until_elapsed(duration_s, workload)
    print(f"rk4_80x80_full: {iters} iterations in ~{duration_s:.1f}s")


if __name__ == "__main__":
    main()
