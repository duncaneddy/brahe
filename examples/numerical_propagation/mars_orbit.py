# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["IGNORE"]
# ///
"""
Propagating a spacecraft orbit about Mars with ForceModelConfig.mars_default().

mars_default() configures a CentralBody.Mars force model: 50x50 GGM2B Mars
gravity, exponential atmospheric drag, solar radiation pressure (occulted by
Mars), and Sun third-body perturbations from the DE440s ephemeris. The
propagator integrates in the Mars-Centered Inertial (MCI) frame; state_in_frame
converts the result into the Mars-fixed MCMF frame for reporting a body-fixed
ground track position.

First run downloads the ggm2bc80 gravity model, caching it under
$BRAHE_CACHE (~/.cache/brahe by default).
"""

import numpy as np
import brahe as bh

# Initialize EOP data and the DE440s planetary ephemeris used for third-body
# perturbations and ECI <-> MCI frame conversions.
bh.initialize_eop()
bh.load_common_spice_kernels()

# Initial epoch
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Circular low Mars orbit at 400 km altitude, expressed directly in the
# Mars-Centered Inertial (MCI) frame the propagator integrates in.
a = bh.R_MARS + 400e3
v = bh.periapsis_velocity(a, 0.0, gm=bh.GM_MARS)
state = np.array([a, 0.0, 0.0, 0.0, v, 0.0])

# Spacecraft parameters, indexed per mars_default()'s ParameterSource
# assignments: [mass, drag_area, Cd, srp_area, Cr].
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

force_config = bh.ForceModelConfig.mars_default()
force_config.validate()

prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    force_config,
    params,
)

# Propagate for 6 hours
final_epoch = epoch + 6.0 * 3600.0
prop.propagate_to(final_epoch)

# state_in_frame routes the propagator's native MCI state through the
# reference frame router into any other supported frame. MCMF is the
# Mars-fixed, IAU/WGCCRE body-fixed frame.
x0_mcmf = prop.state_in_frame(bh.ReferenceFrame.MCMF, epoch)
xf_mcmf = prop.state_in_frame(bh.ReferenceFrame.MCMF, final_epoch)

print(f"Initial epoch: {epoch}")
print(f"Final epoch:   {final_epoch}")
print("\nInitial state (MCMF, Mars-fixed):")
print(
    f"  Position (km): [{x0_mcmf[0] / 1e3:.3f}, {x0_mcmf[1] / 1e3:.3f}, {x0_mcmf[2] / 1e3:.3f}]"
)
print(f"  Velocity (m/s): [{x0_mcmf[3]:.3f}, {x0_mcmf[4]:.3f}, {x0_mcmf[5]:.3f}]")
print("\nFinal state (MCMF, Mars-fixed):")
print(
    f"  Position (km): [{xf_mcmf[0] / 1e3:.3f}, {xf_mcmf[1] / 1e3:.3f}, {xf_mcmf[2] / 1e3:.3f}]"
)
print(f"  Velocity (m/s): [{xf_mcmf[3]:.3f}, {xf_mcmf[4]:.3f}, {xf_mcmf[5]:.3f}]")

# Validate propagation completed and the orbit remains bound to Mars
assert prop.current_epoch() == final_epoch
r_final = np.linalg.norm(xf_mcmf[:3])
assert bh.R_MARS < r_final < bh.R_MARS + 1000e3
print("\nExample validated successfully!")
