# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between eccentric anomaly and mean anomaly.

This example demonstrates the conversion between eccentric and mean anomaly
for a given eccentricity, and validates that the round-trip conversion
returns to the original value.
"""

import brahe as bh

bh.initialize_eop()

ecc = 45.0  # Starting eccentric anomaly (degrees)
e = 0.01  # Eccentricity

# Convert to mean anomaly
mean_anomaly = bh.anomaly_eccentric_to_mean(ecc, e, angle_format=bh.AngleFormat.DEGREES)
print(f"Eccentric anomaly: {ecc:.3f} deg")
print(f"Mean anomaly:      {mean_anomaly:.3f} deg")

# Convert back from mean to eccentric anomaly
ecc_2 = bh.anomaly_mean_to_eccentric(
    mean_anomaly, e, angle_format=bh.AngleFormat.DEGREES
)
print(f"Round-trip result: {ecc_2:.3f} deg")

# Verify round-trip accuracy
print(f"Difference:        {abs(ecc - ecc_2):.2e} deg")

# Expected output:
# Eccentric anomaly: 45.000 deg
# Mean anomaly:      44.595 deg
# Round-trip result: 45.000 deg
# Difference:        0.00e0 deg
