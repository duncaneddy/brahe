# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Determine if a satellite is in Earth's shadow using eclipse models
"""

import brahe as bh
import numpy as np

# Initialize EOP data
bh.initialize_eop()

# Define satellite position and get Sun position
epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
r_sat = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
r_sun = bh.sun_position(epc)

# Check eclipse using conical model (accounts for penumbra)
nu_conical = bh.eclipse_conical(r_sat, r_sun)
print(f"Conical illumination fraction: {nu_conical:.4f}")

# Check eclipse using cylindrical model (binary: 0 or 1)
nu_cyl = bh.eclipse_cylindrical(r_sat, r_sun)
print(f"Cylindrical illumination: {nu_cyl:.1f}")

if nu_conical == 0.0:
    print("Satellite in full shadow (umbra)")
elif nu_conical == 1.0:
    print("Satellite in full sunlight")
else:
    print(f"Satellite in penumbra ({nu_conical * 100:.1f}% illuminated)")

# Expected output:
# Conical illumination fraction: 1.0000
# Cylindrical illumination: 1.0
# Satellite in full sunlight
