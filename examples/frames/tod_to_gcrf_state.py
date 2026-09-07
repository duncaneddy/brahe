# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform a state vector from the true equator and equinox of date (TOD) to GCRF
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

# Hard-coded TOD state vector
state_tod = np.array([6878137.0, 0.0, 0.0, 0.0, 7612.0, 0.0])

print("\nTOD state vector:")
print(f"  Position: [{state_tod[0]:.3f}, {state_tod[1]:.3f}, {state_tod[2]:.3f}] m")
print(f"  Velocity: [{state_tod[3]:.6f}, {state_tod[4]:.6f}, {state_tod[5]:.6f}] m/s\n")

# Transform to GCRF at the given epoch
state_gcrf = bh.state_tod_to_gcrf(epc, state_tod)

print("GCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

# Round trip back to TOD
state_tod_roundtrip = bh.state_gcrf_to_tod(epc, state_gcrf)

pos_err = np.linalg.norm(state_tod[0:3] - state_tod_roundtrip[0:3])
vel_err = np.linalg.norm(state_tod[3:6] - state_tod_roundtrip[3:6])
print("Round-trip error (TOD -> GCRF -> TOD):")
print(f"  Position: {pos_err:.6e} m")
print(f"  Velocity: {vel_err:.6e} m/s")
