# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Query lunar orientation from a binary PCK kernel.

This example demonstrates pck_euler_angles and pck_rotation_matrix using the
moon_pa_de440 kernel, whose principal-axis lunar frame is registered under
NAIF frame class ID 31008 (MOON_PA_DE440). Downloads the PCK kernel on first
run.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# PCKs are never auto-initialized; they must be loaded explicitly.
bh.load_spice_kernel("moon_pa_de440")

epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)

angles, rates = bh.pck_euler_angles(bh.FrameId.MOON_PA_DE440, epc)
print(f"Euler angles [phi, delta, w] (rad): {angles}")
print(f"Euler angle rates (rad/s): {rates}")

R = bh.pck_rotation_matrix(bh.FrameId.MOON_PA_DE440, epc).to_matrix()
print(f"\nICRF -> Moon principal-axis rotation matrix:\n{R}")
print(f"Orthogonality check |R @ R^T - I|: {np.linalg.norm(R @ R.T - np.eye(3)):.3e}")
