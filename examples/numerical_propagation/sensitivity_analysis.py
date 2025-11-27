# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Parameter sensitivity analysis using NumericalOrbitPropagator.
Demonstrates computing sensitivity of orbital state to configuration parameters.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 400e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagation config with sensitivity enabled
prop_config = (
    bh.NumericalPropagationConfig.default()
    .with_sensitivity()
    .with_sensitivity_history()
)

# Define spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator with full force model (needed for parameter sensitivity)
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.default(),
    params=params,
)

print("Spacecraft Parameters:")
print(f"  Mass: {params[0]:.1f} kg")
print(f"  Drag area: {params[1]:.1f} m²")
print(f"  Drag coefficient (Cd): {params[2]:.1f}")
print(f"  SRP area: {params[3]:.1f} m²")
print(f"  SRP coefficient (Cr): {params[4]:.1f}")

# Propagate for one orbital period
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
prop.propagate_to(epoch + orbital_period)

# Get the sensitivity matrix (6 x 5)
sens = prop.sensitivity()

if sens is not None:
    print(f"\nSensitivity Matrix shape: {sens.shape}")
    print(
        "(Rows: state components [x,y,z,vx,vy,vz], Cols: params [mass,A_d,Cd,A_s,Cr])"
    )

    # Analyze position sensitivity to each parameter
    pos_sens = sens[:3, :]  # First 3 rows
    param_names = ["mass", "drag_area", "Cd", "srp_area", "Cr"]

    print("\nPosition sensitivity magnitude to each parameter:")
    for i, name in enumerate(param_names):
        # Position sensitivity magnitude for this parameter
        mag = np.linalg.norm(pos_sens[:, i])
        print(f"  {name:10s}: {mag:.3e} m per unit param")

    # Compute impact of 1% parameter uncertainties
    print("\nPosition error from 1% parameter uncertainty:")
    param_uncertainties = params * 0.01  # 1% of each parameter
    for i, name in enumerate(param_names):
        # dpos = sensitivity * dparam
        pos_error = np.linalg.norm(pos_sens[:, i]) * param_uncertainties[i]
        print(f"  {name:10s}: {pos_error:.1f} m")

    # Total position error (RSS)
    total_pos_error = 0.0
    for i in range(len(params)):
        pos_error = np.linalg.norm(pos_sens[:, i]) * param_uncertainties[i]
        total_pos_error += pos_error**2
    total_pos_error = np.sqrt(total_pos_error)
    print(f"\n  Total (RSS): {total_pos_error:.1f} m")

    # Validate
    assert sens.shape == (6, 5)
    print("\nExample validated successfully!")
else:
    print("\nSensitivity not available (may require full force model)")
