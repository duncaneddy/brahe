# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between true anomaly and mean anomaly.

This example demonstrates the conversion between true and mean anomaly
for a given eccentricity, and validates that the round-trip conversion
returns to the original value.
"""

import brahe as bh

bh.initialize_eop()

nu = 45.0  # Starting true anomaly (degrees)
e = 0.01  # Eccentricity

# Convert to mean anomaly
mean_anomaly = bh.anomaly_true_to_mean(nu, e, bh.AngleFormat.DEGREES)
print(f"True anomaly:      {nu:.3f} deg")
print(f"Mean anomaly:      {mean_anomaly:.3f} deg")

# Convert back from mean to true anomaly
nu_2 = bh.anomaly_mean_to_true(mean_anomaly, e, bh.AngleFormat.DEGREES)
print(f"Round-trip result: {nu_2:.3f} deg")

# Verify round-trip accuracy
print(f"Difference:        {abs(nu - nu_2):.2e} deg")

# Expected output:
# True anomaly:      45.000 deg
# Mean anomaly:      44.194 deg
# Round-trip result: 45.000 deg
# Difference:        0.00e0 deg
