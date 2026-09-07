# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform a TEME state vector from an SGP4 propagator into the ITRF
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

epc = prop.epoch + 600.0
state_teme = prop.state(epc)
print(f"Epoch: {epc}")
print("TEME state vector:")
print(f"  Position: [{state_teme[0]:.3f}, {state_teme[1]:.3f}, {state_teme[2]:.3f}] m")
print(
    f"  Velocity: [{state_teme[3]:.6f}, {state_teme[4]:.6f}, {state_teme[5]:.6f}] m/s\n"
)

state_itrf = bh.state_teme_to_itrf(epc, state_teme)
print("ITRF state vector:")
print(f"  Position: [{state_itrf[0]:.3f}, {state_itrf[1]:.3f}, {state_itrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_itrf[3]:.6f}, {state_itrf[4]:.6f}, {state_itrf[5]:.6f}] m/s\n"
)

speed_teme = np.linalg.norm(state_teme[3:6])
speed_itrf = np.linalg.norm(state_itrf[3:6])
print(f"Inertial speed: {speed_teme:.3f} m/s, Earth-fixed speed: {speed_itrf:.3f} m/s")
