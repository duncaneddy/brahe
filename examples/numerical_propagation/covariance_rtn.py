# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Covariance propagation in RTN (Radial-Tangential-Normal) frame.
Demonstrates frame-specific covariance retrieval and physical interpretation.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Enable STM for covariance propagation
prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()

# Define initial covariance in ECI frame
# Position uncertainty: 10 m in each axis
# Velocity uncertainty: 0.01 m/s in each axis
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

# Create propagator with initial covariance
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.two_body(),
    None,
    initial_covariance=P0,
)

print("=== Covariance in RTN Frame ===\n")
print("Initial position std (ECI): 10.0 m in each axis")

# Propagate for one orbital period
orbital_period = bh.orbital_period(oe[0])
prop.propagate_to(epoch + orbital_period)

# Get covariance in different frames
target = epoch + orbital_period
P_gcrf = prop.covariance_gcrf(target)
P_rtn = prop.covariance_rtn(target)

# Extract position covariances (3x3 upper-left block)
pos_cov_gcrf = P_gcrf[:3, :3]
pos_cov_rtn = P_rtn[:3, :3]

print("\n--- GCRF Frame Results ---")
print("Position std (X, Y, Z):")
print(f"  X: {np.sqrt(pos_cov_gcrf[0, 0]):.1f} m")
print(f"  Y: {np.sqrt(pos_cov_gcrf[1, 1]):.1f} m")
print(f"  Z: {np.sqrt(pos_cov_gcrf[2, 2]):.1f} m")

print("\n--- RTN Frame Results ---")
print("Position std (R, T, N):")
print(f"  Radial (R):     {np.sqrt(pos_cov_rtn[0, 0]):.1f} m  <- Altitude uncertainty")
print(f"  Tangential (T): {np.sqrt(pos_cov_rtn[1, 1]):.1f} m  <- Along-track timing")
print(f"  Normal (N):     {np.sqrt(pos_cov_rtn[2, 2]):.1f} m  <- Cross-track offset")

# Physical interpretation
print("\n--- Physical Interpretation ---")
print("RTN frame aligns with the orbit:")
print("  R (Radial): Points from Earth center to satellite")
print("  T (Tangential): Points along velocity direction")
print("  N (Normal): Completes right-hand system (cross-track)")
print()
print("Key insight: Along-track (T) uncertainty grows fastest because")
print("velocity uncertainty causes timing errors that accumulate.")
print(
    f"After one orbit: T/R ratio = {np.sqrt(pos_cov_rtn[1, 1]) / np.sqrt(pos_cov_rtn[0, 0]):.1f}x"
)

# Show correlation structure
print("\n--- Position Correlation Matrix (RTN) ---")
pos_std_rtn = np.sqrt(np.diag(pos_cov_rtn))
corr_rtn = pos_cov_rtn / np.outer(pos_std_rtn, pos_std_rtn)
print("       R      T      N")
for i, name in enumerate(["R", "T", "N"]):
    print(
        f"  {name}  {corr_rtn[i, 0]:6.3f} {corr_rtn[i, 1]:6.3f} {corr_rtn[i, 2]:6.3f}"
    )

# Validate
assert P_gcrf.shape == (6, 6)
assert P_rtn.shape == (6, 6)
assert np.sqrt(pos_cov_rtn[1, 1]) > np.sqrt(pos_cov_rtn[0, 0])  # T > R

print("\nExample validated successfully!")
