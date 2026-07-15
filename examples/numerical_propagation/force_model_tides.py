# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Solid Earth tidal corrections to the geopotential.
Propagates one LEO orbit with tides ON and OFF, and reports the peak
position difference to show the tidal perturbation magnitude.
"""

import math
import numpy as np
import brahe as bh

# EOP is required for the ITRF frame transformations used inside the tidal model.
bh.initialize_eop()
bh.initialize_sw()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# LEO satellite: 500 km altitude, slightly elliptical, sun-synchronous inclination.
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)


def make_propagator(tides_config):
    """Build a gravity+third-body propagator with the given tides config."""
    force_config = bh.ForceModelConfig.earth_gravity()
    # Add Sun+Moon third-body perturbations. Note: tidal accelerations use their
    # own internal low-precision ephemeris and do NOT depend on this setting.
    force_config.third_body = [bh.ThirdBody.SUN, bh.ThirdBody.MOON]
    force_config.tides = tides_config
    return bh.NumericalOrbitPropagator(
        epoch,
        state0.copy(),
        bh.NumericalPropagationConfig.default(),
        force_config,
    )


# Tides-ON configuration: IERS Step 1 + Step 2 (frequency-dependent), Auto
# permanent-tide handling (converts the model's C̄20 to conventional tide-free).
solid = bh.SolidTideConfig(frequency_dependent=True)
tides_on = bh.TidesConfiguration(permanent=bh.PermanentTideConfig.AUTO, solid=solid)

prop_on = make_propagator(tides_on)
prop_off = make_propagator(None)

# Propagate one full orbital period, sampling every ~minute.
sma = bh.R_EARTH + 500e3
period = 2.0 * math.pi * math.sqrt(sma**3 / bh.GM_EARTH)
n_steps = 90
dt = period / n_steps

max_diff_m = 0.0
t = epoch
for _ in range(n_steps):
    t = t + dt
    prop_on.propagate_to(t)
    prop_off.propagate_to(t)

    pos_on = np.array(prop_on.current_state()[:3])
    pos_off = np.array(prop_off.current_state()[:3])
    diff = np.linalg.norm(pos_on - pos_off)
    if diff > max_diff_m:
        max_diff_m = diff

print("Solid Earth tides example")
print(f"  Orbital period:               {period / 60.0:.1f} min")
print(
    f"  Max tidal position difference: {max_diff_m:.3f} m  ({max_diff_m / 1e3:.3f} km)"
)
assert 0.0 < max_diff_m < 1000.0, f"Unexpected tidal difference: {max_diff_m} m"
print("Example validated successfully!")
