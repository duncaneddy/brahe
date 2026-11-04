# /// script
# dependencies = ["brahe"]
# ///
"""
Read an SP3-d precise orbit file and inspect the first record of satellite G01.
"""

import numpy as np

import brahe as bh

sp3 = bh.SP3.from_file("test_assets/sp3/example-d-1.sp3")
header = sp3.header

print(f"Version:     {header.version}")
print(f"Agency:      {header.agency}")
print(f"Time system: {header.time_system}")
print(f"Satellites:  {sp3.num_satellites}")
print(f"Epochs:      {sp3.num_epochs}")
print(f"Interval:    {header.epoch_interval:.1f} s")
print(f"Span:        {sp3.start_epoch} to {sp3.end_epoch}")

g01 = bh.SP3SatelliteId.parse("G01")
record = sp3.record(g01, 0)
print(f"G01 position (m):    {np.array2string(record.position, precision=3)}")
print(f"G01 clock (us):      {record.clock * 1e6:.6f}")
print(f"G01 velocity:        {record.velocity}")

# A zero position is the format's bad-value marker and is read as absent
print(f"G01 at epoch 1:      {sp3.record(g01, 1)}")
