# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform a state vector from TOD to ITRF and compare with the direct GCRF to ITRF path
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

# Transform directly from TOD to ITRF
state_itrf_direct = bh.state_tod_to_itrf(epc, state_tod)

print("ITRF state vector (direct TOD -> ITRF):")
print(
    f"  Position: [{state_itrf_direct[0]:.3f}, {state_itrf_direct[1]:.3f}, {state_itrf_direct[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_itrf_direct[3]:.6f}, {state_itrf_direct[4]:.6f}, {state_itrf_direct[5]:.6f}] m/s\n"
)

# Transform via GCRF: TOD -> GCRF -> ITRF
state_gcrf = bh.state_tod_to_gcrf(epc, state_tod)
state_itrf_via_gcrf = bh.state_gcrf_to_itrf(epc, state_gcrf)

print("ITRF state vector (via GCRF: TOD -> GCRF -> ITRF):")
print(
    f"  Position: [{state_itrf_via_gcrf[0]:.3f}, {state_itrf_via_gcrf[1]:.3f}, {state_itrf_via_gcrf[2]:.3f}] m"
)
print(
    f"  Velocity: [{state_itrf_via_gcrf[3]:.6f}, {state_itrf_via_gcrf[4]:.6f}, {state_itrf_via_gcrf[5]:.6f}] m/s\n"
)

pos_diff = np.linalg.norm(state_itrf_direct[0:3] - state_itrf_via_gcrf[0:3])
vel_diff = np.linalg.norm(state_itrf_direct[3:6] - state_itrf_via_gcrf[3:6])
print("Difference between the direct and GCRF-mediated paths:")
print(f"  Position: {pos_diff:.6e} m")
print(f"  Velocity: {vel_diff:.6e} m/s")
