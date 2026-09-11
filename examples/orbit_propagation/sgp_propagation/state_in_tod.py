# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Convert SGP4 TEME output into TOD with state_in_frame
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

epc = prop.epoch + 600.0
state_tod = prop.state_in_frame(bh.CelestialFrame.TOD, epc)
state_gcrf = prop.state_gcrf(epc)

print(f"Epoch: {epc}")
print("TOD state vector:")
print(f"  Position: [{state_tod[0]:.3f}, {state_tod[1]:.3f}, {state_tod[2]:.3f}] m")
print(f"  Velocity: [{state_tod[3]:.6f}, {state_tod[4]:.6f}, {state_tod[5]:.6f}] m/s\n")

print("GCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

pos_diff = np.linalg.norm(state_tod[0:3] - state_gcrf[0:3])
print(f"Position difference norm: {pos_diff:.3f} m")
