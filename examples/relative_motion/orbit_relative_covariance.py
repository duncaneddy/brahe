# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform a covariance into and out of an orbit-relative frame from the frame's rotation and rate, and through the frame graph.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Satellite in a 700 km, e = 0.01, sun-synchronous orbit
oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# ECI covariance: 100 m position and 0.1 m/s velocity standard deviations
p_eci = np.diag([1.0e4, 1.0e4, 1.0e4, 1.0e-2, 1.0e-2, 1.0e-2])

# Rotating LVLH: Jacobian from the frame's rotation and angular velocity
r = bh.rotation_eci_to_lvlh(x_eci)
omega = bh.omega_lvlh(x_eci)
p_lvlh = bh.rotate_covariance(p_eci, bh.jacobian_inertial_to_rotating(r, omega))
sigma_v = np.sqrt(np.diag(p_lvlh)[3:])
print(
    f"Rotating LVLH velocity sigmas (m/s): [{sigma_v[0]:.4f}, {sigma_v[1]:.4f}, {sigma_v[2]:.4f}]"
)

# Inertial snapshot: zero rate gives the block-diagonal Jacobian
p_snapshot = bh.rotate_covariance(
    p_eci, bh.jacobian_inertial_to_rotating(r, np.zeros(3))
)
sigma_v = np.sqrt(np.diag(p_snapshot)[3:])
print(
    f"Snapshot LVLH velocity sigmas (m/s): [{sigma_v[0]:.4f}, {sigma_v[1]:.4f}, {sigma_v[2]:.4f}]"
)

# Back to ECI with the inverse Jacobian
p_back = bh.rotate_covariance(p_lvlh, bh.jacobian_rotating_to_inertial(r, omega))
print(
    "Round trip recovers the ECI covariance:",
    np.linalg.norm(p_back - p_eci) / np.linalg.norm(p_eci) < 1e-12,
)

# Frame graph: the same transform for a registered object
epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
bh.register_object("SC", lambda epoch: x_eci, bh.CelestialFrame.GCRF)
p_graph = bh.covariance_frame_to_frame(
    bh.CelestialFrame.GCRF, bh.ReferenceFrame.LVLH("SC"), epc, p_eci
)
print(
    "Frame graph matches the Jacobian route:",
    np.linalg.norm(p_graph - p_lvlh) / np.linalg.norm(p_lvlh) < 1e-12,
)
bh.clear_object_registry()
