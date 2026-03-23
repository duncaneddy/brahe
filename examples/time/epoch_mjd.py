# /// script
# dependencies = ["brahe"]
# ///
"""
Create Epoch instances from Modified Julian Date
"""

import brahe as bh

bh.initialize_eop()

# Create epoch from MJD
mjd = 61041.5
epc2 = bh.Epoch.from_mjd(mjd, bh.UTC)
print(f"MJD {mjd}: {epc2}")

# Verify round-trip conversion
mjd_out = epc2.mjd()
print(f"Round-trip MJD: {mjd_out:.6f}")
