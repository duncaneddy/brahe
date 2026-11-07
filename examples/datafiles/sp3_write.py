# /// script
# dependencies = ["brahe"]
# ///
"""
Build an SP3-d file in memory, write it to text, and read it back.
"""

import numpy as np

import brahe as bh

g01 = bh.SP3SatelliteId(bh.GNSSSystem.GPS, 1)
g02 = bh.SP3SatelliteId(bh.GNSSSystem.GPS, 2)

sp3 = bh.SP3([g01, g02], 900.0, agency="BRHE", comments=["Example orbit file"])

t0 = bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
for k in range(3):
    index = sp3.push_epoch(t0 + 900.0 * k)
    sp3.set_record(
        g01,
        index,
        bh.SP3Record(
            np.array([2.0e7 + 1.0e3 * k, 1.0e7, 1.0e7]),
            velocity=np.array([100.0, 2.0e3, -5.0e2]),
            clock=2.5e-5,
        ),
    )

# G02 is left absent at every epoch except the first
sp3.set_record(g02, 0, bh.SP3Record(np.array([-1.5e7, 2.0e7, 5.0e6])))

text = sp3.to_string()
print("\n".join(text.splitlines()[:3]))
print("...")
print("\n".join(text.splitlines()[-6:]))

parsed = bh.SP3.from_str(text)
print(f"Read back {parsed.num_satellites} satellites and {parsed.num_epochs} epochs")
print(f"G02 at epoch 1: {parsed.record(g02, 1)}")
