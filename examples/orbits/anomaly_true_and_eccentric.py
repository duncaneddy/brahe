# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between true anomaly and eccentric anomaly.

This example demonstrates the conversion between true and eccentric anomaly
for a given eccentricity, and validates that the round-trip conversion
returns to the original value.
"""

import brahe as bh

bh.initialize_eop()

nu = 45.0  # Starting true anomaly (degrees)
e = 0.01  # Eccentricity

# Convert to eccentric anomaly
ecc_anomaly = bh.anomaly_true_to_eccentric(nu, e, bh.AngleFormat.DEGREES)
print(f"True anomaly:      {nu:.3f} deg")
print(f"Eccentric anomaly: {ecc_anomaly:.3f} deg")

# Convert back from eccentric to true anomaly
nu_2 = bh.anomaly_eccentric_to_true(ecc_anomaly, e, bh.AngleFormat.DEGREES)
print(f"Round-trip result: {nu_2:.3f} deg")

# Verify round-trip accuracy
print(f"Difference:        {abs(nu - nu_2):.2e} deg")

# Expected output:
# True anomaly:      45.000 deg
# Eccentric anomaly: 44.944 deg
# Round-trip result: 45.000 deg
# Difference:        0.00e+00 deg
