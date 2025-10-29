# /// script
# dependencies = ["brahe"]
# ///
"""
Create Epoch instances from Julian Date
"""

import brahe as bh

bh.initialize_eop()

# Create epoch from JD
jd = 2460310.5
epc = bh.Epoch.from_jd(jd, bh.UTC)
print(f"JD {jd}: {epc}")
# JD 2460310.5: 2024-01-01 00:00:00.000 UTC

# Verify round-trip conversion
jd_out = epc.jd()
print(f"Round-trip JD: {jd_out:.10f}")
# Round-trip JD: 2460310.5000000000
