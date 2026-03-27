# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Demonstrate the different ways to specify measurement noise covariance.

Shows scalar, per-axis, full covariance, and upper-triangular initialization
for measurement models.
"""

import numpy as np
import brahe as bh

# --- Scalar sigma: same noise on all axes ---
model = bh.ECEFPositionMeasurementModel(5.0)
print("Scalar (5 m isotropic):")
print(model.noise_covariance())

# --- Per-axis sigma: different noise per component ---
model = bh.ECEFPositionMeasurementModel.per_axis(3.0, 3.0, 8.0)
print("\nPer-axis (3, 3, 8 m):")
print(model.noise_covariance())

# --- Full covariance: captures cross-axis correlations ---
cov = np.array(
    [
        [9.0, 1.0, 0.0],
        [1.0, 9.0, 0.0],
        [0.0, 0.0, 64.0],
    ]
)
model = bh.ECEFPositionMeasurementModel.from_covariance(cov)
print("\nFull covariance (with correlations):")
print(model.noise_covariance())

# --- Upper-triangular: compact packed form ---
# Elements: [c00, c01, c02, c11, c12, c22]
upper = np.array([9.0, 1.0, 0.0, 9.0, 0.0, 64.0])
model = bh.ECEFPositionMeasurementModel.from_upper_triangular(upper)
print("\nUpper-triangular packed:")
print(model.noise_covariance())

# --- Standalone covariance helpers ---
r = bh.isotropic_covariance(3, 10.0)
print(f"\nisotropic_covariance(3, 10.0) diagonal: {np.diag(r)}")

r = bh.diagonal_covariance(np.array([5.0, 10.0, 15.0]))
print(f"diagonal_covariance([5, 10, 15]) diagonal: {np.diag(r)}")
