# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Initialize a NumericalOrbitPropagator from an OPM state vector — extract
position, velocity, and epoch to create initial conditions for propagation.
"""

import brahe as bh
import numpy as np
from brahe.ccsds import OPM

bh.initialize_eop()
bh.initialize_sw()

# Parse OPM — use Example1 which has spacecraft mass
opm = OPM.from_file("test_assets/ccsds/opm/OPMExample1.txt")
print(f"Object: {opm.object_name} ({opm.object_id})")
print(f"Epoch:  {opm.epoch}")
print(f"Frame:  {opm.ref_frame}")

# Extract initial conditions from OPM via .state property
initial_state = opm.state  # numpy array [x, y, z, vx, vy, vz]
print("\nInitial state (ECI):")
print(
    f"  Position: [{initial_state[0] / 1e3:.3f}, {initial_state[1] / 1e3:.3f}, {initial_state[2] / 1e3:.3f}] km"
)
print(
    f"  Velocity: [{initial_state[3]:.3f}, {initial_state[4]:.3f}, {initial_state[5]:.3f}] m/s"
)

# Build spacecraft parameters from OPM
mass = opm.mass or 500.0
drag_area = opm.drag_area or 2.0
drag_coeff = opm.drag_coeff or 2.2
srp_area = opm.solar_rad_area or 2.0
srp_coeff = opm.solar_rad_coeff or 1.3
params = np.array([mass, drag_area, drag_coeff, srp_area, srp_coeff])
print(f"\nSpacecraft params: mass={mass}kg, Cd={drag_coeff}, Cr={srp_coeff}")

# Initialize propagator from OPM state
# Note: OPM frame is ITRF2000; we convert to ECI for propagation
# The propagator expects ECI coordinates
state_eci = bh.state_ecef_to_eci(opm.epoch, initial_state)
prop = bh.NumericalOrbitPropagator(
    opm.epoch,
    state_eci,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Propagate for 1 orbit period (approximately)
r = np.linalg.norm(initial_state[:3])
period = 2 * np.pi * np.sqrt(r**3 / bh.GM_EARTH)
print(f"\nEstimated period: {period:.0f}s ({period / 60:.1f} min)")

target_epoch = opm.epoch + period
prop.propagate_to(target_epoch)

# Check final state
final_state = prop.current_state()
print("\nAfter 1 orbit:")
print(f"  Epoch: {prop.current_epoch()}")
print(
    f"  Position: [{final_state[0] / 1e3:.3f}, {final_state[1] / 1e3:.3f}, {final_state[2] / 1e3:.3f}] km"
)
print(
    f"  Velocity: [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}] m/s"
)
