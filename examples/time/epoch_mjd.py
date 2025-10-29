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
# MJD 61041.5: 2026-01-01 12:00:00.000 UTC

# Verify round-trip conversion
mjd_out = epc2.mjd()
print(f"Round-trip MJD: {mjd_out:.6f}")
# Round-trip MJD: 61041.500000
