# /// script
# dependencies = ["brahe"]
# ///
"""
Convert ENZ (East-North-Zenith) position to azimuth-elevation-range coordinates
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define several relative positions in ENZ coordinates
test_cases = [
    ("Directly overhead", np.array([0.0, 0.0, 100e3])),
    ("North horizon", np.array([0.0, 100e3, 0.0])),
    ("East horizon", np.array([100e3, 0.0, 0.0])),
    ("South horizon", np.array([0.0, -100e3, 0.0])),
    ("West horizon", np.array([-100e3, 0.0, 0.0])),
    ("Northeast at 45° elevation", np.array([50e3, 50e3, 70.7e3])),
]

print("Converting ENZ coordinates to Azimuth-Elevation-Range:\n")

for name, enz in test_cases:
    # Convert to azimuth-elevation-range
    azel = bh.position_enz_to_azel(enz, bh.AngleFormat.DEGREES)

    print(f"{name}:")
    print(
        f"  ENZ:   E={enz[0] / 1000:.1f}km, N={enz[1] / 1000:.1f}km, Z={enz[2] / 1000:.1f}km"
    )
    print(
        f"  Az/El: Az={azel[0]:.1f}°, El={azel[1]:.1f}°, Range={azel[2] / 1000:.1f}km\n"
    )

# Expected outputs:
# Directly overhead: Az=0.0°, El=90.0°, Range=100.0km
# North horizon: Az=0.0°, El=0.0°, Range=100.0km
# East horizon: Az=90.0°, El=0.0°, Range=100.0km
# South horizon: Az=180.0°, El=0.0°, Range=100.0km
# West horizon: Az=270.0°, El=0.0°, Range=100.0km
