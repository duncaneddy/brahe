# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize EOP Providers with simpliest way possible
"""

import brahe as bh

bh.initialize_eop()

# Get current time
epc = bh.Epoch.now()

xp, yp, dut1, lod, dX, dY = bh.get_global_eop(epc.mjd())

print(f"At epoch {epc}:")
print(f"  x_pole: {xp} arcseconds")
print(f"  y_pole: {yp} arcseconds")
print(f"  dut1: {dut1} seconds")
print(f"  length of day: {lod} seconds")
print(f"  dX: {dX} arcseconds")
print(f"  dY: {dY} arcseconds")
