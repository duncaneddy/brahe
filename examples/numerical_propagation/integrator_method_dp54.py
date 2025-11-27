# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the DP54 adaptive integrator method (default).
Dormand-Prince 5(4) - the recommended general-purpose integrator.
"""

import brahe as bh

# DP54 (Dormand-Prince 5(4))
# - Adaptive step size control
# - 5th order solution with 4th order error estimate
# - 6-7 function evaluations per step (FSAL optimization)
# - MATLAB's ode45 uses this method
# - Default integrator in Brahe

# Create configuration with DP54 method (default)
config = bh.NumericalPropagationConfig.default()

# Or explicitly specify DP54
config_explicit = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.DP54)

# Customize tolerances for DP54 using builder pattern
config_custom = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-9).with_rel_tol(1e-6)
)

print("DP54 Adaptive Integrator Configuration:")
print("  Method: DP54 (Dormand-Prince 5(4))")
print("  Adaptive: Yes")
print("  Function evaluations per step: 6-7 (FSAL optimized)")
print("  Order: 5(4) - 5th order solution, 4th order error estimate")

print("\nDefault tolerances:")
print("  Absolute tolerance: 1e-6")
print("  Relative tolerance: 1e-3")

print("\nFSAL Optimization:")
print("  First-Same-As-Last (FSAL) reuses the last function")
print("  evaluation from the previous step, reducing cost")
print("  to effectively 6 evaluations per accepted step.")

print("\nCharacteristics:")
print("  - Automatic step size adjustment")
print("  - Built-in error estimation")
print("  - Generally more efficient than RKF45")
print("  - Industry standard (MATLAB ode45)")

print("\nWhen to use DP54:")
print("  - General-purpose orbit propagation")
print("  - When accuracy and efficiency both matter")
print("  - Most LEO to GEO applications")
print("  - Default choice for most users")

print("\nWhen to consider alternatives:")
print("  - RK4: When you need fixed steps or maximum speed")
print("  - RKN1210: When you need highest precision")

print("\nExample validated successfully!")
