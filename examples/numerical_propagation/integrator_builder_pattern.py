# /// script
# dependencies = ["brahe"]
# ///
"""
Using the builder pattern to customize integrator configuration.
Demonstrates chaining with_* methods for flexible configuration.
"""

import brahe as bh

# NumericalPropagationConfig supports a builder pattern
# Start with a preset and chain with_* methods to customize

# Example 1: Customize default configuration
config1 = (
    bh.NumericalPropagationConfig.default()
    .with_abs_tol(1e-9)
    .with_rel_tol(1e-6)
    .with_max_step(300.0)
)

# Example 2: Start with specific method and customize
config2 = (
    bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF45)
    .with_abs_tol(1e-8)
    .with_rel_tol(1e-5)
    .with_max_step(600.0)
)

# Example 3: High precision with step constraints
config3 = (
    bh.NumericalPropagationConfig.high_precision().with_max_step(
        120.0
    )  # Limit max step for output resolution
)

# Example 4: Full control using NumericalPropagationConfig.new()
config4 = bh.NumericalPropagationConfig.new(
    bh.IntegrationMethod.RK4,
    bh.IntegratorConfig.fixed_step(30.0),  # 30 second fixed steps
    bh.VariationalConfig(),
)

print("Builder Pattern for Integrator Configuration")
print("=" * 60)

print("\nAvailable with_* Methods:")
print("  .with_abs_tol(value)      - Set absolute tolerance")
print("  .with_rel_tol(value)      - Set relative tolerance")
print("  .with_max_step(value)     - Set maximum step size")
print("  .with_initial_step(value) - Set initial step size")

print("\nExample 1: Precision DP54")
print("  NumericalPropagationConfig.default()")
print("    .with_abs_tol(1e-9)")
print("    .with_rel_tol(1e-6)")
print("    .with_max_step(300.0)")

print("\nExample 2: Custom RKF45")
print("  NumericalPropagationConfig.with_method(IntegrationMethod.RKF45)")
print("    .with_abs_tol(1e-8)")
print("    .with_rel_tol(1e-5)")
print("    .with_max_step(600.0)")

print("\nExample 3: Constrained High Precision")
print("  NumericalPropagationConfig.high_precision()")
print("    .with_max_step(120.0)")

print("\nExample 4: Full Control (new)")
print("  NumericalPropagationConfig.new(")
print("    IntegrationMethod.RK4,")
print("    IntegratorConfig.fixed_step(30.0),")
print("    VariationalConfig()")
print("  )")

print("\nBenefits of Builder Pattern:")
print("  - Start from sensible defaults")
print("  - Only modify what you need")
print("  - Clear, readable configuration")
print("  - Method chaining for convenience")

print("\nExample validated successfully!")
