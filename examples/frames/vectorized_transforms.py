# /// script
# dependencies = ["brahe"]
# ///
"""
Transform batches of states and positions between ECI and ECEF in one call
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)

# Build a batch of ECI states: one row per satellite, columns [x, y, z, vx, vy, vz]
raan = np.linspace(0.0, 360.0, 6, endpoint=False)
states_eci = np.array(
    [
        bh.state_koe_to_eci(
            np.array([bh.R_EARTH + 500e3, 0.001, 97.8, r, 0.0, 0.0]),
            bh.AngleFormat.DEGREES,
        )
        for r in raan
    ]
)
print(f"ECI states shape: {states_eci.shape}")

# One epoch, many states: the rotation matrices are computed once
states_ecef = bh.state_eci_to_ecef(epc, states_eci)
print(f"ECEF states shape: {states_ecef.shape}")
print(
    f"First ECEF state: {np.array2string(states_ecef[0], precision=3, suppress_small=True, max_line_width=120)}"
)

# The same call with the components along the first axis
states_ecef_t = bh.state_eci_to_ecef(epc, states_eci.T, axis=0)
print(f"Transposed layout shape: {states_ecef_t.shape}")

# Many epochs, one position: a ground station tracked through inertial space
epochs = [epc + 600.0 * i for i in range(6)]
station_ecef = bh.position_geodetic_to_ecef(
    np.array([-122.4, 37.8, 0.0]), bh.AngleFormat.DEGREES
)
station_eci = bh.position_ecef_to_eci(epochs, station_ecef)
print(f"Station ECI positions shape: {station_eci.shape}")
for e, r in zip(epochs, station_eci):
    print(f"  {e}: [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}] m")

# Many epochs, many states: one epoch per row
states_ecef_series = bh.state_eci_to_ecef(epochs, states_eci)
print(f"Per-epoch ECEF states shape: {states_ecef_series.shape}")

# A sequence of epochs also vectorizes the rotation matrices
rotations = bh.rotation_eci_to_ecef(epochs)
print(f"Rotation matrices shape: {rotations.shape}")
